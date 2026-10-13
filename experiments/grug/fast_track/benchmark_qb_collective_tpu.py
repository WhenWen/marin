# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare exact distributed QB quantiles with replicated or partitioned counts."""

import argparse
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def quantile(counts, target_rank):
    counts = counts.astype(jnp.float32)
    bins = counts.shape[-1]
    cumulative = jnp.cumsum(counts[:, ::-1], axis=-1)[:, ::-1]
    crossing = jnp.clip(jnp.sum((cumulative >= target_rank[:, None]).astype(jnp.int32), axis=-1) - 1, 0, bins - 1)
    ct = jnp.take_along_axis(cumulative, crossing[:, None], axis=-1)[:, 0]
    h = jnp.take_along_axis(counts, crossing[:, None], axis=-1)[:, 0]
    width = jnp.float32(4095.0) / bins
    edge = jnp.float32(-2048.0) + crossing.astype(jnp.float32) * width
    return edge + width * (ct - target_rank) / jnp.maximum(h, 1.0)


def distributed_quantile(counts, target_rank, *, axes, scatter, shards):
    if not scatter:
        return quantile(jax.lax.psum(counts, axes), target_rank)
    reduced = jax.lax.psum_scatter(counts, axes, scatter_dimension=0, tiled=True)
    rank = jax.lax.dynamic_slice_in_dim(target_rank, jax.lax.axis_index(axes) * reduced.shape[0], reduced.shape[0])
    assert reduced.shape[0] * shards == target_rank.shape[0]
    return jax.lax.all_gather(quantile(reduced, rank), axes, axis=0, tiled=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    devices = jax.local_devices()
    if len(devices) != 4 or any(d.device_kind != "TPU v4" for d in devices):
        raise RuntimeError("This benchmark requires all four chips of one v4-8 host")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        parser.error("Artifacts must stay in us-central2")
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    records = []
    for mesh_shape in ((4,), (2, 2)):
        axes = tuple(f"batch_{i}" for i in range(len(mesh_shape)))
        mesh = Mesh(np.asarray(devices).reshape(mesh_shape), axes, axis_types=(AxisType.Explicit,) * len(axes))
        for experts, bins in ((16, 31), (512, 10000)):
            for distribution in ("spread", "concentrated", "constant"):
                counts = jax.random.randint(jax.random.key(0), (4 * experts, bins - 1), 0, 4, dtype=jnp.int32)
                counts = jnp.concatenate((counts, 32768 - counts.sum(axis=1, keepdims=True)), axis=1)
                if distribution != "spread":
                    counts = jnp.zeros_like(counts).at[:, 0].set(32768 if distribution == "constant" else 16384)
                    if distribution == "concentrated":
                        counts = counts.at[:, -1].set(16384)
                np.testing.assert_array_equal(np.asarray(counts.sum(axis=1)), np.full(4 * experts, 32768))
                rank = jnp.float32(4 * 32768) * jnp.linspace(1 / experts, 8 / experts, experts)
                with jax.set_mesh(mesh):
                    counts = jax.device_put(counts, NamedSharding(mesh, P(axes, None)))
                    rank = jax.device_put(rank, NamedSharding(mesh, P()))
                    reference = None
                    for scatter in (False, True):
                        name = "reduce_scatter" if scatter else "all_reduce"
                        row = {
                            "implementation": name,
                            "shape": [4, experts, bins],
                            "mesh_shape": mesh_shape,
                            "distribution": distribution,
                            "seed": 0,
                            "git_sha": args.revision,
                            "device_type": devices[0].device_kind,
                            "device_count": 4,
                            "jax_version": jax.__version__,
                            "jaxlib_version": jaxlib.__version__,
                            "xla_flags": os.environ.get("XLA_FLAGS", ""),
                            "error": None,
                        }
                        try:
                            fn = partial(distributed_quantile, axes=axes, scatter=scatter, shards=4)
                            run = jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=(P(axes, None), P()), out_specs=P()))
                            start = time.monotonic()
                            compiled = run.lower(counts, rank).compile()
                            row["compile_time"] = time.monotonic() - start
                            actual = np.asarray(jax.block_until_ready(compiled(counts, rank)))
                            if reference is None:
                                reference = actual
                            row["max_abs_error"] = float(np.max(np.abs(actual - reference)))
                            np.testing.assert_array_equal(actual, reference)
                            row["temporary_bytes"] = compiled.memory_analysis().temp_size_in_bytes
                            for _ in range(2):
                                jax.block_until_ready(compiled(counts, rank))
                            samples = []
                            for _ in range(5):
                                start = time.monotonic()
                                jax.block_until_ready(compiled(counts, rank))
                                samples.append(time.monotonic() - start)
                            row.update(steady_state_time=statistics.median(samples), steady_state_samples=samples)
                            if experts == 512 and distribution == "spread":
                                hlo = output / f"mesh-{len(axes)}-{name}-hlo.txt"
                                hlo.write_text(compiled.as_text())
                                fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
                        except Exception as exc:
                            row["error"] = repr(exc)
                        records.append(row)
                        print(json.dumps(row), flush=True)
                        report = output / "collective-benchmark.json"
                        report.write_text(json.dumps(records, indent=2) + "\n")
                        fs.put_file(str(report), f"{prefix}/{report.name}")
    if any(row["error"] for row in records):
        raise RuntimeError("A collective candidate failed; see retained report")


if __name__ == "__main__":
    main()
