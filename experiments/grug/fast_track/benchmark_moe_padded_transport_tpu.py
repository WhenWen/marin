# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check and time transport candidates with original accepted expert intervals."""

import argparse
import hashlib
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import ml_dtypes
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.moe_padded_transport_tpu import (
    overflowed,
    packed_transport,
    padded_transport,
    peer_capacity,
)
from experiments.grug.fast_track.moe_transport_tpu import preshaped_transport, uint32_transport

DISTRIBUTIONS = ("uniform", "half-valid", "concentrated", "empty", "at-capacity", "over-capacity")


def transport_fixture(seed, distribution, peers, rows, experts, chunk):
    """Build transport metadata independently from an expert-count matrix."""
    local_experts = experts // peers
    capacity = math.ceil(1.05 * rows / 2)
    padded = peer_capacity(rows, capacity, peers)
    rng = np.random.default_rng(seed)
    count = rows // 2 if distribution == "half-valid" else rows
    counts = rng.multinomial(count, np.full(experts, 1 / experts), size=peers).astype(np.int32)
    if distribution == "concentrated":
        counts.fill(0)
        counts[:, 0] = rows
    elif distribution == "empty":
        counts.fill(0)
    elif distribution in ("at-capacity", "over-capacity"):
        counts[:] = rows // experts
        counts[:, : rows % experts] += 1
        target = padded + (distribution == "over-capacity")
        counts[0] = 0
        counts[0, 0] = target
        remaining_experts = experts - local_experts // 2
        counts[0, local_experts // 2 :] = (rows - target) // remaining_experts
        counts[0, local_experts // 2 : local_experts // 2 + (rows - target) % remaining_experts] += 1
    clipped = np.zeros((peers, peers, local_experts), np.int32)
    limit = math.ceil(1.05 * count / 2)
    for destination in range(peers):
        remaining = limit
        for expert in range(chunk * local_experts // 2, (chunk + 1) * local_experts // 2):
            for sender in range(peers):
                accepted = min(remaining, int(counts[sender, destination * local_experts + expert]))
                clipped[sender, destination, expert] = accepted
                remaining -= accepted
    starts = np.cumsum(counts, axis=-1, dtype=np.int32) - counts
    remote = np.zeros_like(clipped)
    for destination in range(peers):
        cursor = 0
        for expert in range(local_experts):
            for sender in range(peers):
                remote[sender, destination, expert] = cursor
                cursor += int(clipped[sender, destination, expert])
    dispatch = (
        starts,
        clipped.reshape(peers, experts),
        remote.reshape(peers, experts),
        clipped.transpose(1, 0, 2).reshape(peers, experts),
    )
    returned = (
        remote.transpose(1, 0, 2).reshape(peers, experts),
        dispatch[3],
        starts.reshape(peers, peers, local_experts).transpose(1, 0, 2).reshape(peers, experts),
        dispatch[1],
    )
    expected_overflow = bool(clipped.sum(axis=-1).max() > padded)
    assert expected_overflow == (chunk == 0 and distribution in ("concentrated", "over-capacity"))
    return counts, clipped, dispatch, returned, capacity, expected_overflow


def payload(seed, peer, start, stop, width, dtype, *, initial=False):
    """Generate only the requested interval; transport oracle uses the same payload definition."""
    rows = np.arange(start, stop, dtype=np.uint32)[:, None]
    if initial:
        return np.broadcast_to((-97.5 - peer - (rows % 3)).astype(dtype), (stop - start, width)).copy()
    if dtype == np.int32:
        return (peer * 1048576 + rows).astype(np.int32)
    channels = np.arange(width, dtype=np.uint32)[None, :]
    hashed = rows * np.uint32(1664525) + channels * np.uint32(1013904223)
    hashed ^= np.uint32((peer * 22695477 + seed * 1103515245) % 2**32)
    hashed ^= hashed >> np.uint32(13)
    values = ((hashed % 1024).astype(np.float32) - 512) / 16
    values = np.where(rows % 17 == 0, np.float32(-0.0), values)
    return values.astype(dtype)


def output_oracle(seed, peer, width, dtype, counts, clipped, capacity, sender_rows, direction, *, payload_fn=payload):
    """Copy accepted intervals in expert/sender order; never use JAX metadata offsets."""
    peers, experts = counts.shape
    local_experts = experts // peers
    result = payload_fn(seed, peer, 0, capacity if direction == "dispatch" else sender_rows, width, dtype, initial=True)
    for destination in range(peers):
        cursor = 0
        for expert in range(local_experts):
            global_expert = destination * local_experts + expert
            for sender in range(peers):
                size = int(clipped[sender, destination, expert])
                offset = int(counts[sender, :global_expert].sum())
                if direction == "dispatch" and destination == peer:
                    result[cursor : cursor + size] = payload_fn(seed, sender, offset, offset + size, width, dtype)
                elif direction == "return" and sender == peer:
                    result[offset : offset + size] = payload_fn(seed, destination, cursor, cursor + size, width, dtype)
                cursor += size
    return result


def mapped_transport(*arrays, mesh, implementation):
    def local(*values):
        values = tuple(x[0] for x in values)
        if implementation == "reference":
            result = jax.lax.ragged_all_to_all(*values, axis_name="expert")
            overflow = jnp.asarray(False)
        elif implementation == "pre-shaped":
            result = preshaped_transport(*values, axis_name="expert")
            overflow = jnp.asarray(False)
        elif implementation == "uint32":
            result = uint32_transport(*values, axis_name="expert")
            overflow = jnp.asarray(False)
        elif implementation == "packed-only":
            capacity = peer_capacity(values[0].shape[0], values[1].shape[0], jax.lax.axis_size("expert"))
            overflow = overflowed(
                values[3], values[5], peers=jax.lax.axis_size("expert"), capacity=capacity, axis_name="expert"
            )
            result = packed_transport(*values, axis_name="expert")
        else:
            result, overflow = padded_transport(*values, axis_name="expert")
        return result[None], overflow[None]

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("expert", None, None), P("expert", None, None), *(P("expert", None),) * 4),
        out_specs=(P("expert", None, None), P("expert")),
    )(*arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--candidate", choices=("padded", "pre-shaped", "uint32"), default="padded")
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--cpu-packed-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_packed_only and args.candidate != "padded":
        raise ValueError("The packed-only CPU preflight applies to padded transport")
    if args.cpu_packed_only and not args.cpu_check:
        raise ValueError("Packed-only preflight is CPU-only; TPU must validate the ragged fallback")
    if not args.cpu_check:
        require_tpu_rounding()
        DistributedConfig().initialize()
    devices = jax.devices()
    peers = 4 if args.cpu_check else 16
    if len(devices) != peers or (not args.cpu_check and any(d.device_kind != "TPU v4" for d in devices)):
        raise ValueError("Use four CPU devices for preflight or a TPU v4-32 slice")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Keep artifacts in us-central2")
    rank = jax.process_index()
    report_name = {
        "padded": "padded-transport",
        "pre-shaped": "preshaped-transport",
        "uint32": "uint32-transport",
    }[args.candidate]
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / f"{report_name}-rank-{rank}"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(f"{args.artifact_prefix}/rank-{rank}")
    report = dict(
        revision=args.revision,
        candidate=args.candidate,
        rank=rank,
        status="running",
        cpu_packed_only=args.cpu_packed_only,
        backend=devices[0].platform,
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        numerics=[],
        compilation=[],
        timings=[],
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primitive_sha256=hashlib.sha256(
            Path(__file__).with_name("moe_padded_transport_tpu.py").read_bytes()
        ).hexdigest(),
        transport_helper_sha256=hashlib.sha256(
            Path(__file__).with_name("moe_transport_tpu.py").read_bytes()
        ).hexdigest(),
    )

    def save():
        path = folder / f"{report_name}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    mesh = Mesh(np.asarray(devices), ("expert",), axis_types=(AxisType.Explicit,))
    specs = (P("expert", None, None), P("expert", None, None), *(P("expert", None),) * 4)
    cache = {}
    shapes = ((1024, peers * 4),) if args.cpu_check else ((1024, 64), (262144, 512))
    save()
    with jax.set_mesh(mesh):
        for sender_rows, experts in shapes:
            full = sender_rows == 262144
            widths = (
                ((256, ml_dtypes.bfloat16), (512, ml_dtypes.bfloat16), (1, np.float32))
                if full
                else ((8, ml_dtypes.bfloat16), (16, ml_dtypes.bfloat16), (1, np.float32), (1, np.int32))
            )
            for seed in (0, 1, 2):
                for distribution in DISTRIBUTIONS:
                    if full and seed != 0 and distribution != "uniform":
                        continue
                    for chunk in (0, 1):
                        counts, clipped, dispatch, returned, capacity, overflow = transport_fixture(
                            seed, distribution, peers, sender_rows, experts, chunk
                        )
                        for direction, metadata in (("dispatch", dispatch), ("return", returned)):
                            source_rows, target_rows = (
                                (sender_rows, capacity) if direction == "dispatch" else (capacity, sender_rows)
                            )
                            for width, dtype in widths:
                                case = dict(
                                    rows=sender_rows,
                                    experts=experts,
                                    width=width,
                                    dtype=np.dtype(dtype).name,
                                    seed=seed,
                                    distribution=distribution,
                                    chunk=chunk,
                                    direction=direction,
                                )

                                def callback(
                                    index,
                                    *,
                                    initial,
                                    seed=seed,
                                    target_rows=target_rows,
                                    source_rows=source_rows,
                                    width=width,
                                    dtype=dtype,
                                ):
                                    return np.stack(
                                        [
                                            payload(
                                                seed,
                                                peer,
                                                index[1].start or 0,
                                                index[1].stop or (target_rows if initial else source_rows),
                                                width,
                                                dtype,
                                                initial=initial,
                                            )
                                            for peer in range(index[0].start or 0, index[0].stop or peers)
                                        ]
                                    )

                                arrays = [
                                    jax.make_array_from_callback(
                                        (peers, source_rows, width),
                                        NamedSharding(mesh, specs[0]),
                                        partial(callback, initial=False),
                                    ),
                                    jax.make_array_from_callback(
                                        (peers, target_rows, width),
                                        NamedSharding(mesh, specs[1]),
                                        partial(callback, initial=True),
                                    ),
                                ]
                                arrays.extend(
                                    jax.make_array_from_callback(
                                        a.shape, NamedSharding(mesh, spec), lambda index, a=a: a[index]
                                    )
                                    for a, spec in zip(metadata, specs[2:], strict=True)
                                )
                                result_row = {
                                    **case,
                                    "expected_overflow": overflow,
                                    "implementations": {},
                                    "metadata_sha256": (
                                        hashlib.sha256(
                                            b"".join(a.tobytes() for a in (counts, clipped, *metadata))
                                        ).hexdigest()
                                    ),
                                }
                                implementations = (
                                    ("packed-only",) if args.cpu_packed_only else ("reference", args.candidate)
                                )
                                for implementation in implementations:
                                    key = (sender_rows, width, np.dtype(dtype).name, direction, implementation)
                                    if key not in cache:
                                        start = time.monotonic()
                                        fn = jax.jit(partial(mapped_transport, mesh=mesh, implementation=implementation))
                                        cache[key] = fn.lower(*arrays).compile()
                                        report["compilation"].append(
                                            dict(
                                                key=key,
                                                seconds=time.monotonic() - start,
                                                temporary_bytes=cache[key].memory_analysis().temp_size_in_bytes,
                                            )
                                        )
                                        if rank == 0 and not args.cpu_check:
                                            path = folder / ("-".join(map(str, key)) + "-hlo.txt")
                                            path.write_text(cache[key].as_text())
                                            fs.put_file(str(path), f"{prefix}/{path.name}")
                                    actual, branches = jax.block_until_ready(cache[key](*arrays))
                                    output_checked = not (args.cpu_packed_only and overflow)
                                    exact = True if output_checked else None
                                    mismatched_elements = 0
                                    if output_checked:
                                        for shard in actual.addressable_shards:
                                            peer = shard.index[0].start
                                            expected = output_oracle(
                                                seed,
                                                peer,
                                                width,
                                                dtype,
                                                counts,
                                                clipped,
                                                capacity,
                                                sender_rows,
                                                direction,
                                            )
                                            value = np.ascontiguousarray(np.asarray(shard.data)[0])
                                            bit_dtype = np.dtype(f"u{value.dtype.itemsize}")
                                            mismatches = int(
                                                np.count_nonzero(value.view(bit_dtype) != expected.view(bit_dtype))
                                            )
                                            mismatched_elements += mismatches
                                            exact &= mismatches == 0
                                    branch_ok = all(
                                        bool(np.asarray(shard.data)[0])
                                        == (False if implementation in ("reference", "pre-shaped", "uint32") else overflow)
                                        for shard in branches.addressable_shards
                                    )
                                    result_row["implementations"][implementation] = dict(
                                        exact=bool(exact) if output_checked else None,
                                        mismatched_elements=mismatched_elements if output_checked else None,
                                        output_checked=output_checked,
                                        branch_correct=branch_ok,
                                    )
                                result_row["passed"] = all(
                                    (m["exact"] or not m["output_checked"]) and m["branch_correct"]
                                    for m in result_row["implementations"].values()
                                )
                                report["numerics"].append(result_row)
                                if not result_row["passed"]:
                                    save()
                                    raise ValueError(f"Transport mismatch: {result_row}")
                                if full and distribution == "uniform":
                                    for name in ("reference", args.candidate):
                                        for _ in range(2):
                                            jax.block_until_ready(cache[(*key[:-1], name)](*arrays))
                                    for repeat in range(8):
                                        for name in (
                                            ("reference", args.candidate)
                                            if repeat % 2 == 0
                                            else (args.candidate, "reference")
                                        ):
                                            multihost_utils.sync_global_devices(
                                                f"{sender_rows}-{width}-{seed}-{chunk}-{direction}-{repeat}-{name}"
                                            )
                                            start = time.monotonic()
                                            jax.block_until_ready(cache[(*key[:-1], name)](*arrays))
                                            report["timings"].append(
                                                {
                                                    **case,
                                                    "implementation": name,
                                                    "repeat": repeat,
                                                    "seconds": time.monotonic() - start,
                                                }
                                            )
                                if not full and rank == 0:
                                    path = folder / ("fixture-" + "-".join(map(str, case.values())) + ".npz")
                                    np.savez_compressed(
                                        path,
                                        counts=counts,
                                        clipped=clipped,
                                        source=np.stack(
                                            [payload(seed, p, 0, source_rows, width, dtype) for p in range(peers)]
                                        ),
                                        initial=np.stack(
                                            [
                                                payload(seed, p, 0, target_rows, width, dtype, initial=True)
                                                for p in range(peers)
                                            ]
                                        ),
                                        **dict(
                                            zip(
                                                ("input_offsets", "send_sizes", "output_offsets", "recv_sizes"),
                                                metadata,
                                                strict=True,
                                            )
                                        ),
                                    )
                                    if not args.cpu_check:
                                        fs.put_file(str(path), f"{prefix}/{path.name}")
                            save()
                    print(
                        json.dumps(
                            dict(rows=sender_rows, seed=seed, distribution=distribution, checked=len(report["numerics"]))
                        ),
                        flush=True,
                    )
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
