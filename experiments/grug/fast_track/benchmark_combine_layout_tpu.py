# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen route-unrolled and token-tiled layouts for the TPU MoE output combine."""

import argparse
import functools
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from levanter.grug._moe.ep_ragged_all_to_all import _unpermute_from_global_expert

from experiments.grug.fast_track.benchmark_combine_tpu import fixture, value_and_gradients
from experiments.grug.fast_track.diagnose_kda_state_precision import errors


def unrolled_combine(intermediate, sorted_indices, weights, accepted, *, tokens_per_shard, topk):
    if topk != 8:
        raise ValueError("The route-unrolled screen requires topk=8")
    positions = jnp.argsort(sorted_indices).reshape(tokens_per_shard, topk)
    rounded_weights = weights.astype(intermediate.dtype).astype(jnp.float32)
    terms = []
    for route in range(topk):
        rows = jnp.where(accepted[:, route, None], intermediate[positions[:, route]], 0)
        terms.append(rows.astype(jnp.float32) * rounded_weights[:, route, None])
    return ((terms[0] + terms[1]) + (terms[2] + terms[3])) + ((terms[4] + terms[5]) + (terms[6] + terms[7]))


def tiled_combine(intermediate, sorted_indices, weights, accepted, *, tokens_per_shard, topk):
    tile = min(tokens_per_shard, 256)
    if tokens_per_shard % tile:
        raise ValueError("The token-tiled screen requires whole tiles")
    positions = jnp.argsort(sorted_indices).reshape(-1, tile, topk)
    tiled_weights = weights.astype(intermediate.dtype).astype(jnp.float32).reshape(-1, tile, topk)
    tiled_accepted = accepted.reshape(-1, tile, topk)

    def combine_tile(inputs):
        indices, weight, keep = inputs
        rows = jnp.where(keep[..., None], intermediate[indices], 0).astype(jnp.float32)
        return jnp.sum(rows * weight[..., None], axis=1, dtype=jnp.float32)

    return jax.lax.map(combine_tile, (positions, tiled_weights, tiled_accepted)).reshape(tokens_per_shard, -1)


IMPLEMENTATIONS = {
    "production": _unpermute_from_global_expert,
    "unrolled": unrolled_combine,
    "tiled": tiled_combine,
}


def numerical_case(seed, tokens, width, scenario, functions, device):
    host_inputs, oracle = fixture(seed, width, scenario, tokens=tokens)
    inputs = tuple(jax.device_put(x, device) for x in host_inputs)
    values = {name: jax.block_until_ready(fn(*inputs)) for name, fn in functions.items()}
    measurements = {
        name: {
            key: errors(actual, expected)
            for key, actual, expected in zip(("output", "dx", "dw"), result, oracle, strict=True)
        }
        for name, result in values.items()
    }
    passed = {}
    direct = {}
    for name in ("unrolled", "tiled"):
        passed[name] = all(
            measurements["production"][key]["finite"]
            and measurements[name][key]["finite"]
            and measurements[name][key]["max_abs_error"] <= measurements["production"][key]["max_abs_error"] + 1e-5
            and measurements[name][key]["relative_l2"] <= measurements["production"][key]["relative_l2"] + 1e-6
            for key in ("output", "dx", "dw")
        )
        direct[name] = {
            key: errors(actual, reference)
            for key, actual, reference in zip(("output", "dx", "dw"), values[name], values["production"], strict=True)
        }
    return {
        "seed": seed,
        "tokens": tokens,
        "width": width,
        "scenario": scenario,
        "input_sha256": [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in host_inputs],
        "implementations": measurements,
        "direct_vs_production": direct,
        "passed": passed,
    }


def timing_inputs(seed, width):
    tokens, topk = 32768, 8
    selected = jax.lax.top_k(jax.random.uniform(jax.random.key(seed), (tokens, 512)), topk)[1]
    order = jnp.argsort(selected.reshape(-1), stable=True)
    x = jax.random.normal(jax.random.key(seed + 1), (tokens * topk, width), jnp.bfloat16)
    weights = jax.nn.softmax(jax.random.normal(jax.random.key(seed + 2), (tokens, topk)), axis=-1)
    accepted = jax.random.uniform(jax.random.key(seed + 3), (tokens, topk)) > 0.1
    return jax.block_until_ready((x, order, weights, accepted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    device = jax.local_devices()[0]
    if device.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("This screen requires TPU v4 and us-central2 artifacts")
    if jax.__version__ != "0.11.1" or jaxlib.__version__ != "0.11.1":
        raise ValueError("The frozen screen requires JAX/JAXLIB 0.11.1")
    if jax.config.jax_enable_pgle or os.environ.get("XLA_FLAGS") != "--xla_allow_excess_precision=false":
        raise ValueError("The frozen screen requires PGLE off and excess precision disabled")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "combine-layout"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "revision": args.revision,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "device_type": device.device_kind,
        "device_count_used": 1,
        "jax_enable_pgle": jax.config.jax_enable_pgle,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "numerics": [],
        "compilation": [],
        "timings": [],
        "status": "running",
    }

    def save():
        path = folder / "combine-layout.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    with jax.default_device(device):
        functions = {
            name: jax.jit(functools.partial(value_and_gradients, combine=fn)) for name, fn in IMPLEMENTATIONS.items()
        }
        for seed in (0, 1, 2):
            for tokens in (32, 512):
                for width in (128, 256, 512):
                    for scenario in ("random", "all-dropped", "zero-weight", "dropped-nan"):
                        row = numerical_case(seed, tokens, width, scenario, functions, device)
                        report["numerics"].append(row)
                        save()
                        print(json.dumps({"numerical_fixture": len(report["numerics"]), "passed": row["passed"]}), flush=True)
        qualified = [name for name in ("unrolled", "tiled") if all(r["passed"][name] for r in report["numerics"])]
        report["numerically_qualified"] = qualified
        save()
        for width in (256, 512):
            if not qualified:
                break
            inputs = timing_inputs(10, width)
            compiled = {}
            names = ["production", *qualified]
            for name in names:
                call = functools.partial(IMPLEMENTATIONS[name], tokens_per_shard=32768, topk=8)
                started = time.monotonic()
                executable = jax.jit(call).lower(*inputs).compile()
                compiled[name] = executable
                report["compilation"].append(
                    {
                        "width": width,
                        "implementation": name,
                        "seconds": time.monotonic() - started,
                        "temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
                    }
                )
                path = folder / f"width-{width}-{name}-hlo.txt"
                path.write_text(executable.as_text())
                fs.put_file(str(path), f"{prefix}/{path.name}")
            for seed in (10, 20, 30):
                inputs = timing_inputs(seed, width)
                for name in names:
                    for _ in range(2):
                        output = jax.block_until_ready(compiled[name](*inputs))
                        if not bool(jnp.all(jnp.isfinite(output))):
                            raise ValueError(f"Nonfinite output: {name}, seed={seed}, width={width}")
                for repeat in range(8):
                    for name in names if repeat % 2 == 0 else names[::-1]:
                        started = time.monotonic()
                        output = jax.block_until_ready(compiled[name](*inputs))
                        elapsed = time.monotonic() - started
                        if not bool(jnp.all(jnp.isfinite(output))):
                            raise ValueError(f"Nonfinite output: {name}, seed={seed}, width={width}")
                        report["timings"].append(
                            {"seed": seed, "width": width, "implementation": name, "repeat": repeat, "seconds": elapsed}
                        )
                save()
                print(json.dumps({"timed_seed": seed, "width": width}), flush=True)
        report["comparisons"] = []
        for seed in (10, 20, 30):
            for width in (256, 512):
                if not qualified:
                    break
                medians = {
                    name: statistics.median(
                        r["seconds"]
                        for r in report["timings"]
                        if r["seed"] == seed and r["width"] == width and r["implementation"] == name
                    )
                    for name in ("production", *qualified)
                }
                report["comparisons"].append(
                    {
                        "seed": seed,
                        "width": width,
                        "medians": medians,
                        "passed": {name: medians[name] <= 0.90 * medians["production"] for name in qualified},
                    }
                )
        report["eligible_for_native_trial"] = [
            name for name in qualified if all(row["passed"][name] for row in report["comparisons"])
        ]
        report["status"] = "completed"
        save()
        print(json.dumps({"qualified": qualified, "eligible_for_native_trial": report["eligible_for_native_trial"]}), flush=True)


if __name__ == "__main__":
    main()
