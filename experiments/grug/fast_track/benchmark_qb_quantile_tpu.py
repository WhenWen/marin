# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure exact two-level QB histogram searches on one local TPU v4 chip."""

import argparse
import functools
import json
import os
import statistics
import time
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.benchmark_qb_collective_tpu import quantile
from experiments.grug.fast_track.qb_balancing import hierarchical_histogram_crossing, histogram_quantile


def hierarchical_quantile(counts, target_rank, *, block_bins):
    width = jnp.float32(4095.0) / counts.shape[-1]
    return histogram_quantile(counts.astype(jnp.float32), target_rank, jnp.float32(-2048.0), width, block_bins=block_bins)


def fixture(seed, experts, bins, distribution):
    rng = np.random.default_rng(seed)
    counts = np.zeros((experts, bins), np.int32)
    total = 524288
    if distribution == "multinomial":
        counts = rng.multinomial(total, np.full(bins, 1 / bins), size=experts).astype(np.int32)
    elif distribution == "single":
        counts[np.arange(experts), rng.integers(0, bins, size=experts)] = total
    elif distribution == "edges":
        counts[:, 0] += total // 2
        counts[:, -1] += total // 2
    elif distribution != "empty":
        raise ValueError(distribution)
    ranks = np.resize(np.array([0, 1, 8192, total, total + 1, 8192.5], np.float32), experts)
    # Independent integer search, including the clipped out-of-range cases.
    suffix = np.flip(np.cumsum(np.flip(counts.astype(np.int64), axis=-1), axis=-1), axis=-1)
    crossing = np.array(
        [
            int(indices[-1]) if len(indices := np.flatnonzero(row >= rank)) else 0
            for row, rank in zip(suffix, ranks, strict=True)
        ]
    )
    oracle = (crossing, suffix[np.arange(experts), crossing], counts[np.arange(experts), crossing])
    return counts, ranks, oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    require_tpu_rounding()
    device = jax.local_devices()[0]
    if device.device_kind != "TPU v4":
        raise ValueError("This probe requires a local TPU v4 chip")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Keep artifacts in us-central2")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "kernel": "qb_histogram_quantile",
        "git_sha": args.revision,
        "dtype": "int32 counts and float32 beta",
        "backend": "tpu",
        "device_type": device.device_kind,
        "device_count": 1,
        "jax_version": jax.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
        "status": "running",
        "numerics": [],
        "timings": [],
        "implementations": [],
    }

    def save():
        path = folder / "quantile.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    save()
    reference = jax.jit(quantile)
    candidates = {b: jax.jit(functools.partial(hierarchical_quantile, block_bins=b)) for b in (128, 256)}
    discrete = {b: jax.jit(functools.partial(hierarchical_histogram_crossing, block_bins=b)) for b in (128, 256)}
    with jax.default_device(device):
        for experts, bins in ((8, 1), (8, 31), (8, 129), (8, 257), (512, 10000)):
            for seed in (0, 1, 2):
                for distribution in ("multinomial", "single", "edges", "empty"):
                    counts, ranks, oracle = fixture(seed, experts, bins, distribution)
                    fixture_path = folder / f"fixture-{experts}-{bins}-{seed}-{distribution}.npz"
                    np.savez_compressed(fixture_path, counts=counts, ranks=ranks, crossing=oracle[0])
                    fs.put_file(str(fixture_path), f"{prefix}/fixtures/{fixture_path.name}")
                    x, target = jax.device_put(counts, device), jax.device_put(ranks, device)
                    expected = np.asarray(reference(x, target))
                    for block_bins in (128, 256):
                        actual = np.asarray(candidates[block_bins](x, target))
                        indices = tuple(np.asarray(a) for a in discrete[block_bins](x, target))
                        row = {
                            "shape": [experts, bins],
                            "seed": seed,
                            "distribution": distribution,
                            "block_bins": block_bins,
                            "beta_mismatches": int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
                            "max_abs_error": float(np.max(np.abs(actual - expected))),
                            "discrete_mismatches": [
                                int(np.count_nonzero(a != b)) for a, b in zip(indices, oracle, strict=True)
                            ],
                            "finite": bool(np.all(np.isfinite(actual))),
                        }
                        report["numerics"].append(row)
                    save()
            print(f"Completed numerical shape {(experts, bins)}", flush=True)
        counts, ranks, _ = fixture(0, 512, 10000, "multinomial")
        ranks.fill(8192)
        x, target = jax.device_put(counts, device), jax.device_put(ranks, device)
        jax.clear_caches()
        compiled = {}
        for name, fn in (("reference", reference), *((f"block{b}", f) for b, f in candidates.items())):
            start = time.monotonic()
            compiled[name] = fn.lower(x, target).compile()
            report["implementations"].append(
                {
                    "implementation": name,
                    "shape": [512, 10000],
                    "block_sizes": {"bins": None if name == "reference" else int(name[5:])},
                    "compile_time": time.monotonic() - start,
                    "temporary_bytes": compiled[name].memory_analysis().temp_size_in_bytes,
                    "error": None,
                }
            )
            hlo = folder / f"{name}-hlo.txt"
            hlo.write_text(compiled[name].as_text())
            fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
            for _ in range(2):
                jax.block_until_ready(compiled[name](x, target))
        names = list(compiled)
        for repeat in range(8):
            order = names[repeat % len(names) :] + names[: repeat % len(names)]
            for name in order:
                start = time.monotonic()
                jax.block_until_ready(compiled[name](x, target))
                report["timings"].append({"implementation": name, "repeat": repeat, "seconds": time.monotonic() - start})
            save()
        for row in report["implementations"]:
            row["steady_state_time"] = statistics.median(
                t["seconds"] for t in report["timings"] if t["implementation"] == row["implementation"]
            )
        report["numerical_passed"] = all(
            r["finite"] and r["beta_mismatches"] == 0 and not any(r["discrete_mismatches"]) for r in report["numerics"]
        )
        report["status"] = "completed"
        save()
        print(json.dumps({k: v for k, v in report.items() if k not in ("numerics", "timings")}), flush=True)


if __name__ == "__main__":
    main()
