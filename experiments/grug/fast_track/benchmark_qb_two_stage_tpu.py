# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare full and exact two-stage distributed QB histograms."""

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

from experiments.grug.fast_track.benchmark_qb_flat_histogram_tpu import fixture
from experiments.grug.fast_track.qb_balancing import factorized_expert_histogram


def crossing_details(counts, rank, last_valid):
    cumulative = jnp.cumsum(counts[:, ::-1], axis=-1)[:, ::-1]
    crossing = jnp.clip(jnp.sum(cumulative >= rank[:, None], axis=-1) - 1, 0, last_valid)
    ct = jnp.take_along_axis(cumulative, crossing[:, None], axis=-1)[:, 0]
    height = jnp.take_along_axis(counts, crossing[:, None], axis=-1)[:, 0]
    return crossing, ct, height


def details(indices, rank, *, axes, bins, group, radix, implementation):
    if implementation == "reference":
        counts = jax.lax.psum(factorized_expert_histogram(indices, bins, group, radix), axes).astype(jnp.float32)
        crossing, ct, height = crossing_details(counts, rank, bins - 1)
    else:
        block_bins = 64
        coarse_bins = (bins + block_bins - 1) // block_bins
        coarse_indices = indices // block_bins
        coarse_counts = jax.lax.psum(factorized_expert_histogram(coarse_indices, coarse_bins, group, 16), axes).astype(
            jnp.float32
        )
        block, coarse_ct, coarse_height = crossing_details(coarse_counts, rank, coarse_bins - 1)
        above = coarse_ct - coarse_height
        selected = jnp.where(coarse_indices == block[None, :], indices % block_bins, -1)
        # Negative high digits produce all-zero one-hots, excluding other blocks.
        fine_counts = jax.lax.psum(factorized_expert_histogram(selected, block_bins, group, 8), axes).astype(jnp.float32)
        cumulative = jnp.cumsum(fine_counts[:, ::-1], axis=-1)[:, ::-1] + above[:, None]
        last_valid = jnp.minimum(block_bins - 1, bins - 1 - block * block_bins)
        local_bin = jnp.clip(jnp.sum(cumulative >= rank[:, None], axis=-1) - 1, 0, last_valid)
        ct = jnp.take_along_axis(cumulative, local_bin[:, None], axis=-1)[:, 0]
        height = jnp.take_along_axis(fine_counts, local_bin[:, None], axis=-1)[:, 0]
        crossing = block * block_bins + local_bin
    width = jnp.float32(4095) / bins
    lower = jnp.float32(-2048) + crossing.astype(jnp.float32) * width
    threshold = lower + width * (ct - rank) / jnp.maximum(height, 1.0)
    return threshold, crossing, ct, height


def threshold_only(indices, rank, **kwargs):
    return details(indices, rank, **kwargs)[0]


def oracle(counts, rank, bins):
    cumulative = np.cumsum(counts[:, ::-1], axis=-1, dtype=np.int32)[:, ::-1]
    crossing = np.clip((cumulative >= rank[:, None]).sum(axis=-1) - 1, 0, bins - 1)
    rows = np.arange(len(rank))
    ct = cumulative[rows, crossing]
    height = counts[rows, crossing]
    width = np.float32(4095) / np.float32(bins)
    lower = np.float32(-2048) + crossing.astype(np.float32) * width
    threshold = lower + width * (ct.astype(np.float32) - rank) / np.maximum(height.astype(np.float32), 1)
    return threshold, crossing, ct, height


def boundary_ranks(counts, tokens, bins):
    experts = counts.shape[0]
    cumulative = np.cumsum(counts[:, ::-1], axis=-1, dtype=np.int32)[:, ::-1]
    coarse_start = (np.arange(experts) % ((bins + 63) // 64)) * 64
    suffix = cumulative[np.arange(experts), coarse_start].astype(np.float32)
    options = np.stack(
        (
            np.zeros(experts),
            np.ones(experts),
            np.full(experts, tokens),
            np.full(experts, tokens + 1),
            suffix,
            suffix + np.float32(0.5),
        )
    )
    return options[np.arange(experts) % 6, np.arange(experts)].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    devices = jax.local_devices(backend="cpu" if args.cpu_check else "tpu")
    if len(devices) != 4 or (not args.cpu_check and any(d.device_kind != "TPU v4" for d in devices)):
        raise ValueError("Use four CPU devices for non-training checks or all four chips of a TPU v4-8 host")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Keep artifacts in us-central2")
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "two-stage-histogram"
    output.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = dict(
        status="running",
        revision=args.revision,
        training=False,
        backend=devices[0].platform,
        device_type=devices[0].device_kind,
        device_count=4,
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        numpy_version=np.__version__,
        records=[],
    )
    report["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    report["libtpu_init_args"] = os.environ.get("LIBTPU_INIT_ARGS", "")
    name = "cpu-check.json" if args.cpu_check else "two-stage-histogram.json"

    def save():
        path = output / name
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{name}")

    shapes = (
        ((128, 32, 127, 16, 16), (64, 32, 1031, 16, 32))
        if args.cpu_check
        else ((2048, 16, 127, 16, 16), (32768, 512, 10000, 64, 64))
    )
    compiled = {}
    save()
    for mesh_shape in ((4,), (2, 2)):
        axes = tuple(f"batch_{i}" for i in range(len(mesh_shape)))
        mesh = Mesh(np.asarray(devices).reshape(mesh_shape), axes, axis_types=(AxisType.Explicit,) * len(axes))
        with jax.set_mesh(mesh):
            for tokens, experts, bins, group, radix in shapes:
                for seed in (0, 1, 2):
                    for distribution in ("uniform", "edges", "constant"):
                        host, local_counts, ordinary, _ = fixture(seed, tokens, experts, bins, distribution)
                        counts = local_counts.sum(axis=0, dtype=np.int32)
                        digest = hashlib.sha256(host.tobytes()).hexdigest()
                        inputs = jax.device_put(host.reshape(-1, experts), NamedSharding(mesh, P(axes, None)))
                        for rank_mode, host_rank in (
                            ("ordinary", ordinary),
                            ("boundary", boundary_ranks(counts, 4 * tokens, bins)),
                        ):
                            expected = oracle(counts, host_rank, bins)
                            rank = jax.device_put(host_rank, NamedSharding(mesh, P()))
                            label = f"{len(axes)}-{tokens}-{seed}-{distribution}-{rank_mode}"
                            if args.cpu_check or tokens == 2048:
                                path = output / f"fixture-{label}.npz"
                                np.savez(path, indices=host, counts=counts, rank=host_rank, threshold=expected[0])
                                fs.put_file(str(path), f"{prefix}/{path.name}")
                            row = dict(
                                mesh_shape=mesh_shape,
                                local_shape=[tokens, experts, bins],
                                seed=seed,
                                distribution=distribution,
                                rank_mode=rank_mode,
                                input_sha256=digest,
                                implementations={},
                                samples=[],
                            )
                            reference = None
                            for implementation in ("reference", "two-stage"):
                                key = (mesh_shape, tokens, implementation)
                                if key not in compiled:
                                    kwargs = dict(
                                        axes=axes, bins=bins, group=group, radix=radix, implementation=implementation
                                    )

                                    def executable(function, out_specs):
                                        mapped = jax.shard_map(
                                            partial(function, **kwargs),
                                            mesh=mesh,
                                            in_specs=(P(axes, None), P()),
                                            out_specs=out_specs,
                                        )
                                        return jax.jit(mapped).lower(inputs, rank).compile()

                                    start = time.monotonic()
                                    timed = executable(threshold_only, P())
                                    compile_time = time.monotonic() - start
                                    checker = executable(details, (P(), P(), P(), P()))
                                    compiled[key] = (timed, checker, compile_time)
                                    if not args.cpu_check:
                                        path = output / f"hlo-{len(axes)}-{tokens}-{implementation}.txt"
                                        path.write_text(timed.as_text())
                                        fs.put_file(str(path), f"{prefix}/{path.name}")
                                timed, checker, compile_time = compiled[key]
                                actual = tuple(np.asarray(x) for x in jax.block_until_ready(checker(inputs, rank)))
                                timed_threshold = np.asarray(jax.block_until_ready(timed(inputs, rank)))
                                if reference is None:
                                    reference = actual[0]
                                row["implementations"][implementation] = dict(
                                    finite=bool(np.isfinite(actual[0]).all()),
                                    discrete_exact=all(np.array_equal(a, b) for a, b in zip(actual[1:], expected[1:])),
                                    reference_exact=bool(
                                        np.array_equal(actual[0].view(np.uint32), reference.view(np.uint32))
                                    ),
                                    timed_threshold_exact=bool(
                                        np.array_equal(timed_threshold.view(np.uint32), actual[0].view(np.uint32))
                                    ),
                                    oracle_close=bool(np.allclose(actual[0], expected[0], atol=1e-5, rtol=1e-5)),
                                    oracle_max_abs_error=float(np.max(np.abs(actual[0] - expected[0]))),
                                    compile_seconds=compile_time,
                                    compiler_temporary_bytes=timed.memory_analysis().temp_size_in_bytes,
                                )
                                if not args.cpu_check:
                                    for _ in range(2):
                                        jax.block_until_ready(timed(inputs, rank))
                            if not args.cpu_check:
                                for repeat in range(8):
                                    order = ("reference", "two-stage") if repeat % 2 == 0 else ("two-stage", "reference")
                                    for implementation in order:
                                        timed = compiled[(mesh_shape, tokens, implementation)][0]
                                        start = time.monotonic()
                                        jax.block_until_ready(timed(inputs, rank))
                                        row["samples"].append(
                                            dict(
                                                implementation=implementation,
                                                repeat=repeat,
                                                seconds=time.monotonic() - start,
                                            )
                                        )
                                row["median_seconds"] = {
                                    variant: statistics.median(
                                        x["seconds"] for x in row["samples"] if x["implementation"] == variant
                                    )
                                    for variant in ("reference", "two-stage")
                                }
                            row["passed_numerics"] = all(
                                m["finite"]
                                and m["discrete_exact"]
                                and m["reference_exact"]
                                and m["oracle_close"]
                                and m["timed_threshold_exact"]
                                for m in row["implementations"].values()
                            )
                            report["records"].append(row)
                            save()
                            print(json.dumps(row), flush=True)
    report["status"] = "completed"
    save()
    if not all(row["passed_numerics"] for row in report["records"]):
        raise RuntimeError("Two-stage histogram failed numerical checks")


if __name__ == "__main__":
    main()
