# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare TPU MoE gather/mask/reduce with the production gather/dot combine."""

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
import ml_dtypes
import numpy as np
from levanter.grug._moe.ep_common import _sort_activations
from levanter.grug._moe.ep_ragged_all_to_all import _unpermute_from_global_expert

from experiments.grug.fast_track.diagnose_kda_state_precision import errors


def reduced_combine(intermediate, sorted_indices, combine_weights_local, accepted, *, tokens_per_shard, topk):
    """Expose gather, mask, multiply and reduction to XLA fusion."""
    positions = jnp.argsort(sorted_indices)
    unsorted = _sort_activations(intermediate, positions)
    rows = jnp.where(accepted[..., None], unsorted.reshape(tokens_per_shard, topk, -1), 0)
    weights = combine_weights_local.astype(rows.dtype).astype(jnp.float32)
    return jnp.sum(rows.astype(jnp.float32) * weights[..., None], axis=1, dtype=jnp.float32)


def value_and_gradients(x, order, weights, accepted, cotangent, *, combine):
    output, pullback = jax.vjp(
        lambda a, b: combine(a, order, b, accepted, tokens_per_shard=weights.shape[0], topk=weights.shape[1]),
        x,
        weights,
    )
    return output, *pullback(cotangent)


def fixture(seed, width, scenario):
    rng = np.random.default_rng(seed)
    tokens, topk = 32, 8
    order = rng.permutation(tokens * topk).astype(np.int32)
    rows = rng.normal(size=(tokens, topk, width)).astype(ml_dtypes.bfloat16)
    weights = rng.uniform(size=(tokens, topk)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    accepted = rng.uniform(size=(tokens, topk)) > 0.2
    if scenario == "all-dropped":
        accepted[:] = False
    if scenario == "zero-weight":
        accepted[0] = True
        weights[0] = 0
    if scenario == "dropped-nan":
        rows[~accepted] = np.nan
    cotangent = rng.normal(size=(tokens, width)).astype(ml_dtypes.bfloat16).astype(np.float32)
    x = rows.reshape(tokens * topk, width)[order]
    clean64 = np.where(accepted[..., None], rows.astype(np.float64), 0)
    w64 = weights.astype(ml_dtypes.bfloat16).astype(np.float64)
    ct64 = cotangent.astype(np.float64)
    output = np.sum(clean64 * w64[..., None], axis=1)
    dx = np.where(accepted[..., None], ct64[:, None, :] * w64[..., None], 0)
    dx = dx.reshape(tokens * topk, width)[order].astype(ml_dtypes.bfloat16)
    dw = np.sum(clean64 * ct64[:, None, :], axis=-1).astype(ml_dtypes.bfloat16).astype(np.float32)
    return (x, order, weights, accepted, cotangent), (output, dx, dw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    device = jax.local_devices()[0]
    if device.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("This probe requires TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "combine"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    implementations = {"production": _unpermute_from_global_expert, "reduce": reduced_combine}
    report = {
        "revision": args.revision,
        "jax_version": jax.__version__,
        "device_type": device.device_kind,
        "device_count_used": 1,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "oracle": "NumPy FP64 weighted sum and analytical gradients; BF16 input and weight-cast transposes",
        "numerics": [],
        "compilation": [],
        "timings": [],
        "status": "running",
    }

    def save():
        path = folder / "combine.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    with jax.default_device(device):
        functions = {
            name: jax.jit(functools.partial(value_and_gradients, combine=fn)) for name, fn in implementations.items()
        }
        for seed in (0, 1, 2):
            for width in (128, 512):
                for scenario in ("random", "all-dropped", "zero-weight", "dropped-nan"):
                    host_inputs, oracle = fixture(seed, width, scenario)
                    inputs = tuple(jax.device_put(x, device) for x in host_inputs)
                    row = {"seed": seed, "width": width, "scenario": scenario, "implementations": {}}
                    for name, fn in functions.items():
                        values = jax.block_until_ready(fn(*inputs))
                        row["implementations"][name] = {
                            key: errors(actual, expected)
                            for key, actual, expected in zip(("output", "dx", "dw"), values, oracle, strict=True)
                        }
                    passed = True
                    for key in ("output", "dx", "dw"):
                        reference = row["implementations"]["production"][key]
                        candidate = row["implementations"]["reduce"][key]
                        passed &= reference["finite"] and candidate["finite"]
                        passed &= candidate["max_abs_error"] <= reference["max_abs_error"] + 1e-5
                        passed &= candidate["relative_l2"] <= reference["relative_l2"] + 1e-6
                    row["passed"] = passed
                    report["numerics"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
        tokens, topk, width = 32768, 8, 512
        selected = jax.lax.top_k(jax.random.uniform(jax.random.key(1), (tokens, 512)), topk)[1]
        order = jnp.argsort(selected.reshape(-1), stable=True)
        x = jax.random.normal(jax.random.key(2), (tokens * topk, width), jnp.bfloat16)
        weights = jax.nn.softmax(jax.random.normal(jax.random.key(3), (tokens, topk)), axis=-1)
        accepted = jax.random.uniform(jax.random.key(4), (tokens, topk)) > 0.1
        ct = jax.random.normal(jax.random.key(5), (tokens, width), jnp.bfloat16).astype(jnp.float32)
        inputs = (x, order, weights, accepted, ct)
        for mode in ("forward", "forward-backward"):
            compiled = {}
            for name, fn in implementations.items():
                if mode == "forward":
                    call = functools.partial(fn, tokens_per_shard=tokens, topk=topk)
                    call_inputs = inputs[:4]
                else:
                    call = functools.partial(value_and_gradients, combine=fn)
                    call_inputs = inputs
                start = time.monotonic()
                executable = jax.jit(call).lower(*call_inputs).compile()
                compile_seconds = time.monotonic() - start
                compiled[name] = executable
                report["compilation"].append(
                    {
                        "mode": mode,
                        "implementation": name,
                        "compile_seconds": compile_seconds,
                        "temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
                    }
                )
                path = folder / f"{mode}-{name}-hlo.txt"
                path.write_text(executable.as_text())
                fs.put_file(str(path), f"{prefix}/{path.name}")
                for _ in range(2):
                    jax.block_until_ready(executable(*call_inputs))
            for repeat in range(8):
                names = ("production", "reduce") if repeat % 2 == 0 else ("reduce", "production")
                for name in names:
                    start = time.monotonic()
                    jax.block_until_ready(compiled[name](*call_inputs))
                    report["timings"].append(
                        {"mode": mode, "implementation": name, "repeat": repeat, "seconds": time.monotonic() - start}
                    )
            save()
        report["medians"] = {
            mode: {
                name: statistics.median(
                    r["seconds"] for r in report["timings"] if r["mode"] == mode and r["implementation"] == name
                )
                for name in implementations
            }
            for mode in ("forward", "forward-backward")
        }
        report["numerical_gate_passed"] = all(r["passed"] for r in report["numerics"])
        forward = report["medians"]["forward"]
        report["forward_time_reduction"] = 1 - forward["reduce"] / forward["production"]
        report["eligible_for_native_trial"] = report["numerical_gate_passed"] and report["forward_time_reduction"] >= 0.1
        report["status"] = "completed"
        save()
        print(json.dumps({k: v for k, v in report.items() if k not in ("numerics", "timings")}), flush=True)


if __name__ == "__main__":
    main()
