# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native-readable Zarr-v2 checkpoints without TensorStore's write/OCDBT path.

Reuse the native disjoint shard/chunk plans. Encode chunks in Python, upload
bounded immutable objects, then use the native all-host completion barrier.
The unchanged native legacy checkpoint reader provides independent readback.
"""
import gc
import itertools
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch

import fsspec
import jax
import numpy as np
import zstandard
from jax.experimental.array_serialization import serialization as array_ser
from levanter import checkpoint
from levanter import tensorstore_serialization as native

logger = logging.getLogger(__name__)


def detached_host_snapshot(data):
    """Copy one shard without caching all TPU buffers in host memory."""
    cpu = jax.local_devices(backend="cpu")[0]
    staged = jax.device_put(data, jax.sharding.SingleDeviceSharding(cpu), may_alias=False)
    try:
        return np.array(staged, copy=True)
    finally:
        staged.delete()


def shard_snapshot(shard, local_slice=None):
    if local_slice is None:
        return detached_host_snapshot(shard.data)
    sliced = native._slice_shard_on_device(shard.data, *local_slice)
    try:
        return detached_host_snapshot(sliced)
    finally:
        sliced.delete()


def intervals(index, shape):
    return tuple((s.start or 0, dim if s.stop is None else s.stop) for s, dim in zip(index, shape, strict=True))


def check_coverage(array, plan):
    """Verify complete, disjoint global rectangles before writing any chunk."""
    shape = array.shape
    rects = [
        intervals(r.index, shape) for s in array.global_shards if (r := native._shard_write_region(s, plan)) is not None
    ]
    assert sum(math.prod(b - a for a, b in r) for r in rects) == array.size
    for i, r in enumerate(rects):
        assert all(0 <= a < b <= dim for (a, b), dim in zip(r, shape, strict=True))
        for other in rects[:i]:
            assert not all(
                max(a, c) < min(b, d) for (a, b), (c, d) in zip(r, other, strict=True)
            ), "Overlapping checkpoint writers"


def serialize_python_zarr(
    checkpoint_dir,
    pytree,
    manager=None,
    *,
    commit_callback=None,
    on_local_commit=None,
    on_staged=None,
    write_config=None,
    **unused,
):
    config = write_config or native.TensorStoreWriteConfig()
    manager = manager or array_ser.GlobalAsyncCheckpointManager()
    manager.wait_until_finished()
    paths, arrays = native._flatten_serializable_leaves(pytree)
    fs, root = fsspec.core.url_to_fs(str(checkpoint_dir))
    pending = []
    pending_bytes = peak = writes = 0
    codec = zstandard.ZstdCompressor(level=1)

    def upload(path, content):
        fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        fs.pipe_file(path, content)

    def drain():
        nonlocal pending_bytes
        for future, _size in pending:
            future.result()
        pending.clear()
        pending_bytes = 0

    with ThreadPoolExecutor(max_workers=4) as pool:

        def enqueue(path, content):
            nonlocal pending_bytes, peak, writes
            if pending and (pending_bytes + len(content) > config.max_staged_host_bytes or len(pending) >= 8):
                drain()
            pending.append((pool.submit(upload, path, content), len(content)))
            pending_bytes += len(content)
            peak = max(peak, pending_bytes)
            writes += 1

        for i, (path, array) in enumerate(zip(paths, arrays, strict=True)):
            plan = native.plan_array_write(path, array, config)
            check_coverage(array, plan)
            shape = tuple(array.shape)
            chunks = tuple(plan.chunk_shape)
            dtype = np.dtype(array.dtype)
            meta = {
                "zarr_format": 2,
                "shape": list(shape),
                "chunks": list(chunks),
                "dtype": "bfloat16" if dtype.name == "bfloat16" else dtype.str,
                "compressor": {"id": "zstd", "level": 1},
                "fill_value": None,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
            if jax.process_index() == 0:
                enqueue(root + "/" + path + "/.zarray", json.dumps(meta, sort_keys=True).encode())
            for shard in array.addressable_shards:
                region = native._shard_write_region(shard, plan)
                if region is None:
                    continue
                bounds = intervals(region.index, shape)
                assert all(
                    a % c == 0 and (b % c == 0 or b == d) for (a, b), c, d in zip(bounds, chunks, shape, strict=True)
                ), "Chunk ownership crosses a writer boundary"
                host = shard_snapshot(shard, region.local_slice)
                ranges = [range(a // c, (b + c - 1) // c) for (a, b), c in zip(bounds, chunks, strict=True)]
                for coord in itertools.product(*ranges):
                    local = tuple(
                        slice(k * c - a, min((k + 1) * c, b) - a)
                        for k, c, (a, b) in zip(coord, chunks, bounds, strict=True)
                    )
                    block = np.ascontiguousarray(host[local]) if shape else host.reshape(())
                    if block.shape != chunks:
                        padded = np.zeros(chunks, dtype=dtype)
                        padded[tuple(slice(0, n) for n in block.shape)] = block
                        block = padded
                    content = codec.compress(block.tobytes(order="C"))
                    key = ".".join(map(str, coord)) if coord else "0"
                    enqueue(root + "/" + path + "/" + key, content)
                    del content, block
                del host
            if i % 32 == 0:
                logger.info("Python Zarr upload array=%d objects=%d pending_bytes=%d", i, writes, pending_bytes)
        drain()
    if on_staged is not None:
        on_staged(peak)
    if on_local_commit is not None:
        on_local_commit("local_completed")
    manager._add_futures([])
    manager._start_async_commit(commit_callback or (lambda: None))
    manager.wait_until_finished()
    logger.info(
        "Python Zarr checkpoint committed process=%d arrays=%d objects=%d peak_compressed_bytes=%d",
        jax.process_index(),
        len(arrays),
        writes,
        peak,
    )
    gc.collect()
    return peak


@contextmanager
def python_zarr_checkpoint_writes():
    with (
        patch.object(checkpoint, "tree_serialize_leaves_tensorstore", serialize_python_zarr),
        patch.object(native, "tree_serialize_leaves_tensorstore", serialize_python_zarr),
    ):
        yield
