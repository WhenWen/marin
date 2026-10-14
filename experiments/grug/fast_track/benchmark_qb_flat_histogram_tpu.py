# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare grouped and flat loop outputs for the distributed QB histogram pipeline."""

import argparse
import hashlib
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

from experiments.grug.fast_track.qb_balancing import factorized_expert_histogram, histogram_quantile


def flat_histogram(indices, n_bins, group_size, radix, *, axes):
    experts = indices.shape[1]
    high_bins = (n_bins + radix - 1) // radix
    groups = indices.T

    def update(index, output):
        group = jax.lax.dynamic_slice_in_dim(groups, index * group_size, group_size, axis=0)
        high = jax.nn.one_hot(group // radix, high_bins, dtype=jnp.int8)
        low = jax.nn.one_hot(group % radix, radix, dtype=jnp.int8)
        counts = jax.lax.dot_general(high, low, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.int32)
        counts = counts.reshape(group_size, high_bins * radix)[:, :n_bins]
        return jax.lax.dynamic_update_slice(output, counts, (index * group_size, 0))

    initial = jax.lax.pcast(jnp.zeros((experts, n_bins), jnp.int32), axes, to="varying")
    return jax.lax.fori_loop(0, experts // group_size, update, initial)


def pipeline(indices, rank, *, axes, n_bins, group_size, radix, implementation):
    histogram = partial(flat_histogram, axes=axes) if implementation == "flat" else factorized_expert_histogram
    counts = jax.lax.psum(histogram(indices, n_bins, group_size, radix), axes).astype(jnp.float32)
    return histogram_quantile(counts, rank, jnp.float32(-2048), jnp.float32(4095) / n_bins)


def fixture(seed, tokens, experts, bins, distribution, shards=4):
    rng = np.random.default_rng(seed)
    indices = rng.integers(bins, size=(shards, tokens, experts), dtype=np.int32)
    if distribution == "edges":
        indices %= 2
        indices *= bins - 1
    elif distribution == "constant":
        indices.fill(seed % bins)
    counts = np.empty((shards, experts, bins), np.int32)
    for shard in range(shards):
        for expert in range(experts):
            counts[shard, expert] = np.bincount(indices[shard, :, expert], minlength=bins)
    rank = np.float32(shards * tokens) * np.linspace(1 / experts, 8 / experts, experts, dtype=np.float32)
    global_counts = counts.sum(axis=0, dtype=np.int32).astype(np.float32)
    cumulative = np.cumsum(global_counts[:, ::-1], axis=-1, dtype=np.float32)[:, ::-1]
    crossing = np.clip((cumulative >= rank[:, None]).sum(axis=-1) - 1, 0, bins - 1)
    ct = cumulative[np.arange(experts), crossing]
    height = global_counts[np.arange(experts), crossing]
    width = np.float32(4095) / np.float32(bins)
    expected = np.float32(-2048) + crossing.astype(np.float32) * width + width * (ct - rank) / np.maximum(height, 1)
    return indices, counts, rank, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    devices = jax.local_devices(backend="tpu")
    if len(devices) != 4 or any(d.device_kind != "TPU v4" for d in devices):
        raise ValueError("Use all four chips of one TPU v4-8 host")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Keep artifacts in us-central2")
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "flat-histogram"
    output.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "status": "running",
        "revision": args.revision,
        "backend": "tpu",
        "device_type": devices[0].device_kind,
        "device_count": len(devices),
        "dtype": "int32 counts, float32 thresholds",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "numpy_version": np.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "records": [],
    }

    def save():
        path = output / "flat-histogram.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    compiled = {}
    checkers = {}
    save()
    for mesh_shape in ((4,), (2, 2)):
        axes = tuple(f"batch_{i}" for i in range(len(mesh_shape)))
        mesh = Mesh(np.asarray(devices).reshape(mesh_shape), axes, axis_types=(AxisType.Explicit,) * len(axes))
        with jax.set_mesh(mesh):
            for tokens, experts, bins, group, radix in ((2048, 16, 127, 16, 16), (32768, 512, 10000, 64, 64)):
                for seed in (0, 1, 2):
                    for distribution in ("uniform", "edges", "constant"):
                        host, expected_counts, host_rank, expected = fixture(seed, tokens, experts, bins, distribution)
                        digest = hashlib.sha256(host.tobytes()).hexdigest()
                        if tokens == 2048 and mesh_shape == (4,):
                            path = output / f"fixture-{seed}-{distribution}.npz"
                            np.savez(path, indices=host, counts=expected_counts, rank=host_rank, thresholds=expected)
                            fs.put_file(str(path), f"{prefix}/{path.name}")
                        inputs = jax.device_put(host.reshape(-1, experts), NamedSharding(mesh, P(axes, None)))
                        rank = jax.device_put(host_rank, NamedSharding(mesh, P()))
                        row = {
                            "mesh_shape": mesh_shape,
                            "local_shape": [tokens, experts, bins],
                            "group_size": group,
                            "radix": radix,
                            "seed": seed,
                            "distribution": distribution,
                            "input_sha256": digest,
                            "implementations": {},
                            "samples": [],
                        }
                        reference = None
                        for name in ("reference", "flat"):
                            key = (mesh_shape, tokens, name)
                            if key not in compiled:
                                kwargs = dict(n_bins=bins, group_size=group, radix=radix)
                                fn = partial(pipeline, axes=axes, implementation=name, **kwargs)
                                run = jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=(P(axes, None), P()), out_specs=P()))
                                start = time.monotonic()
                                executable = run.lower(inputs, rank).compile()
                                compiled[key] = (executable, time.monotonic() - start)
                                histogram = partial(flat_histogram, axes=axes) if name == "flat" else factorized_expert_histogram
                                checker = jax.jit(
                                    jax.shard_map(
                                        partial(histogram, **kwargs),
                                        mesh=mesh,
                                        in_specs=P(axes, None),
                                        out_specs=P(axes, None),
                                    )
                                )
                                checkers[key] = checker.lower(inputs).compile()
                                path = output / f"hlo-{len(axes)}-{tokens}-{name}.txt"
                                path.write_text(executable.as_text())
                                fs.put_file(str(path), f"{prefix}/{path.name}")
                            actual_counts = np.asarray(jax.block_until_ready(checkers[key](inputs)))
                            executable, compile_time = compiled[key]
                            actual = np.asarray(jax.block_until_ready(executable(inputs, rank)))
                            if name == "reference":
                                reference = actual
                            assert reference is not None
                            row["implementations"][name] = {
                                "counts_exact": bool(np.array_equal(actual_counts, expected_counts.reshape(-1, bins))),
                                "finite": bool(np.isfinite(actual).all()),
                                "thresholds_exact": bool(
                                    np.array_equal(actual.view(np.uint32), reference.view(np.uint32))
                                ),
                                "oracle_close": bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-5)),
                                "oracle_max_abs_error": float(np.max(np.abs(actual - expected))),
                                "compile_seconds": compile_time,
                                "compiler_temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
                            }
                            for _ in range(2):
                                jax.block_until_ready(executable(inputs, rank))
                        for repeat in range(8):
                            names = ("reference", "flat") if repeat % 2 == 0 else ("flat", "reference")
                            for name in names:
                                executable, _ = compiled[(mesh_shape, tokens, name)]
                                start = time.monotonic()
                                jax.block_until_ready(executable(inputs, rank))
                                row["samples"].append(
                                    {"implementation": name, "repeat": repeat, "seconds": time.monotonic() - start}
                                )
                        row["median_seconds"] = {
                            name: statistics.median(r["seconds"] for r in row["samples"] if r["implementation"] == name)
                            for name in ("reference", "flat")
                        }
                        row["passed_numerics"] = all(
                            r["counts_exact"] and r["finite"] and r["thresholds_exact"] and r["oracle_close"]
                            for r in row["implementations"].values()
                        )
                        report["records"].append(row)
                        save()
                        print(json.dumps(row), flush=True)
    report["status"] = "completed"
    save()
    if not all(r["passed_numerics"] for r in report["records"]):
        raise RuntimeError("Histogram pipeline failed numerical checks")


if __name__ == "__main__":
    main()
