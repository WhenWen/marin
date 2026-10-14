# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure batching of independent Inkling query blocks on one TPU v4 chip."""

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
from levanter.grug.attention._core import AttentionMask, _blocked_inkling_math

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.benchmark_inkling_flash_tpu import fixture, reference64, values_and_gradients
from experiments.grug.fast_track.diagnose_kda_state_precision import errors


def batched_attention(q, k, v, band, segments, *, query_blocks):
    return _blocked_inkling_math(
        q,
        k,
        v,
        AttentionMask.causal().with_segment_ids(segments),
        band,
        block_size=128,
        query_block_batch_size=query_blocks,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    require_tpu_rounding()
    device = jax.local_devices(backend="tpu")[0]
    cpu = jax.local_devices(backend="cpu")[0]
    if device.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Use TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "inkling-query-batch"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    implementations = {"reference": None, "batch2": 2, "batch4": 4, "batch8": 8}
    functions = {
        name: jax.jit(partial(values_and_gradients, attention=partial(batched_attention, query_blocks=batch)))
        for name, batch in implementations.items()
    }
    report = {
        "revision": args.revision,
        "status": "running",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "device_type": device.device_kind,
        "device_count_used": 1,
        "dtype": "bfloat16 inputs; reference FP64 dense attention oracle",
        "block_size": 128,
        "query_block_batches": implementations,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "numerics": [],
        "compilation": [],
        "warmups": [],
        "timings": [],
    }

    def save():
        path = folder / "query-batches.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    save()
    for seed in (0, 1, 2):
        for length in (256, 512, 1024):
            for scenario in ("one-document", "packed", "strong-logits"):
                host = fixture(seed, length, scenario)
                path = folder / f"fixture-{seed}-{length}-{scenario}.npz"
                np.savez(path, **dict(zip(("q", "k", "v", "band", "segments", "cotangent"), host, strict=True)))
                fs.put_file(str(path), f"{prefix}/{path.name}")
                with jax.enable_x64(True), jax.default_device(cpu):
                    inputs64 = tuple(
                        jax.device_put(x.astype(np.int32 if i == 4 else np.float64), cpu) for i, x in enumerate(host)
                    )
                    oracle = jax.jit(partial(values_and_gradients, attention=reference64))(*inputs64)
                    oracle = tuple(np.asarray(x) for x in jax.device_get(oracle))
                with jax.default_device(device):
                    inputs = tuple(jax.device_put(x, device) for x in host)
                    row = {"seed": seed, "length": length, "scenario": scenario, "implementations": {}, "passed": {}}
                    for name, fn in functions.items():
                        values = jax.block_until_ready(fn(*inputs))
                        row["implementations"][name] = {
                            key: errors(actual, expected)
                            for key, actual, expected in zip(
                                ("output", "dq", "dk", "dv", "dband"), values, oracle, strict=True
                            )
                        }
                    for name in ("batch2", "batch4", "batch8"):
                        row["passed"][name] = all(
                            baseline["finite"]
                            and candidate["finite"]
                            and candidate[metric] <= 1.25 * baseline[metric] + 1e-3
                            for key, baseline in row["implementations"]["reference"].items()
                            for candidate in (row["implementations"][name][key],)
                            for metric in ("max_abs_error", "relative_l2")
                        )
                    report["numerics"].append(row)
                    save()
                    print(
                        json.dumps({"seed": seed, "length": length, "scenario": scenario, "passed": row["passed"]}),
                        flush=True,
                    )
    with jax.default_device(device):
        for batch, length, heads, kv_heads in ((1, 1024, 2, 1), (8, 4096, 6, 6)):
            shape = [batch, length, heads, kv_heads, 128]
            q, k, v, band, ct = (
                jax.random.normal(jax.random.key(seed), dims, jnp.bfloat16)
                for seed, dims in (
                    (10, (batch, length, heads, 128)),
                    (11, (batch, length, kv_heads, 128)),
                    (12, (batch, length, kv_heads, 128)),
                    (13, (batch, heads, length, 1152)),
                    (14, (batch, length, heads, 128)),
                )
            )
            segments = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32) // 1024, (batch, length))
            inputs = q, k, v, band, segments, ct
            compiled = {}
            for name, fn in functions.items():
                report["phase"] = {"stage": "compiling", "shape": shape, "implementation": name}
                save()
                start = time.monotonic()
                executable = fn.lower(*inputs).compile()
                compiled[name] = executable
                report["compilation"].append(
                    {
                        "shape": shape,
                        "implementation": name,
                        "seconds": time.monotonic() - start,
                        "temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
                    }
                )
                path = folder / f"b{batch}-s{length}-{name}-hlo.txt"
                path.write_text(executable.as_text())
                fs.put_file(str(path), f"{prefix}/{path.name}")
                for repeat in range(2):
                    values = jax.block_until_ready(executable(*inputs))
                    finite = bool(jax.jit(lambda xs: jnp.stack([jnp.isfinite(x).all() for x in xs]).all())(values))
                    report["warmups"].append(
                        {"shape": shape, "implementation": name, "repeat": repeat, "finite": finite}
                    )
                    save()
                    if not finite:
                        raise ValueError(f"Nonfinite target outputs/gradients: {name}, {shape}")
            for repeat in range(8):
                names = list(implementations)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    start = time.monotonic()
                    jax.block_until_ready(compiled[name](*inputs))
                    report["timings"].append(
                        {"shape": shape, "implementation": name, "repeat": repeat, "seconds": time.monotonic() - start}
                    )
                save()
            medians = {
                name: statistics.median(
                    r["seconds"] for r in report["timings"] if r["shape"] == shape and r["implementation"] == name
                )
                for name in implementations
            }
            print(json.dumps({"shape": shape, "medians": medians}), flush=True)
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
