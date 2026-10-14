# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare TPU KDA implementations with an independent CPU FP64 recurrence.

The CPU computation is a numerical oracle, with no optimizer or training loop.
This probe reports errors without changing any acceptance tolerance.
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import fsspec
import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.benchmark_kda_state_tpu import evaluate, tpu_recurrence
from experiments.grug.moe import kda


def recurrence(inputs, segments):
    """Evaluate the tokenwise delta rule in the supplied floating-point dtype."""
    q, k, v, g, beta, initial = inputs
    q = q / jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k / jnp.sqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    q = q / jnp.sqrt(jnp.asarray(q.shape[-1], q.dtype))
    keep = jnp.concatenate((jnp.ones(1, dtype=bool), segments[1:] == segments[:-1]))

    def step(state, token):
        qt, kt, vt, gt, bt, retain = token
        decayed = state * (jnp.exp(gt) * retain)[..., :, None]
        prediction = jnp.sum(kt[..., :, None] * decayed, axis=-2)
        update = bt[..., None] * (vt - prediction)
        following = decayed + kt[..., :, None] * update[..., None, :]
        output = jnp.sum(qt[..., :, None] * following, axis=-2)
        return following, output

    tokens = tuple(jnp.moveaxis(x, -2, 0) for x in (q, k, v, g))
    final, output = jax.lax.scan(step, initial, (*tokens, jnp.moveaxis(beta, -1, 0), keep))
    return jnp.moveaxis(output, 0, -2), final


def oracle(inputs, cotangent, final_cotangent, segments):
    def objective(xs):
        output, final = recurrence(xs, segments)
        value = jnp.sum(output * cotangent) + jnp.sum(final * final_cotangent)
        return value, (output, final)

    (value, outputs), gradients = jax.value_and_grad(objective, has_aux=True)(inputs)
    return (*outputs, value, *gradients)


def errors(actual, reference):
    actual = np.asarray(jax.device_get(actual), dtype=np.float64)
    reference = np.asarray(jax.device_get(reference), dtype=np.float64)
    delta = actual - reference
    norm = np.linalg.norm(reference.ravel())
    actual_norm = np.linalg.norm(actual.ravel())
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / max(norm, 1e-300)),
        "cosine": (
            float(np.vdot(actual.ravel(), reference.ravel()) / (norm * actual_norm))
            if norm > 0 and actual_norm > 0
            else None
        ),
        "reference_l2": float(norm),
        "finite": bool(np.isfinite(actual).all() and np.isfinite(reference).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tpu = jax.devices("tpu")[0]
    cpu = jax.devices("cpu")[0]
    if tpu.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("This probe requires TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "revision": args.revision,
        "oracle": "CPU FP64 tokenwise recurrence",
        "seed": args.seed,
        "final_cotangent_seed": args.seed + 8,
        "jax_version": jax.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "records": [],
    }
    names = ("output", "final_state", "objective", "dq", "dk", "dv", "dg", "dbeta", "dinitial")
    for fixture, shape, document_length in (
        ("random", (1, 2, 128, 128), 37),
        ("repeated", (1, 2, 128, 128), 37),
        ("strong-decay", (1, 2, 128, 128), 37),
        ("long-packed", (1, 1, 1024, 128), 83),
        ("uninterrupted", (1, 1, 256, 128), 256),
    ):
        with jax.default_device(tpu):
            keys = jax.random.split(jax.random.key(args.seed), 7)
            q, k, v = (jax.random.normal(key, shape, jnp.bfloat16) for key in keys[:3])
            if fixture == "repeated":
                k = jnp.broadcast_to(k[..., :1, :], shape)
            scale = 5.0 if fixture == "strong-decay" else 0.05
            inputs = (
                q,
                k,
                v,
                -scale * jax.nn.sigmoid(jax.random.normal(keys[3], shape)),
                jax.nn.sigmoid(jax.random.normal(keys[4], shape[:-1])),
                0.01 * jax.random.normal(keys[5], (*shape[:2], 128, 128)),
            )
            ct = jax.random.normal(keys[6], shape)
            ft = jax.random.normal(jax.random.key(args.seed + 8), inputs[-1].shape)
            segments = jnp.arange(shape[-2]) // document_length
            values = {}
            for implementation in ("production", "pallas_state"):
                compiled = []
                for mode in ("forward", "value_and_grad"):
                    fn = jax.jit(lambda xs, c, f, s, mode=mode: evaluate(xs, c, f, s, mode=mode))
                    if implementation == "pallas_state":
                        with patch.object(kda, "_parallel_chunk_recurrence", tpu_recurrence):
                            executable = fn.lower(inputs, ct, ft, segments).compile()
                    else:
                        executable = fn.lower(inputs, ct, ft, segments).compile()
                    compiled.append(jax.block_until_ready(executable(inputs, ct, ft, segments)))
                outputs, (value, gradients) = compiled
                values[implementation] = (*outputs, value, *gradients)
        with jax.enable_x64(True), jax.default_device(cpu):
            cpu_inputs = tuple(jnp.asarray(np.asarray(jax.device_get(x), dtype=np.float64)) for x in inputs)
            cpu_ct = jnp.asarray(np.asarray(jax.device_get(ct), dtype=np.float64))
            cpu_ft = jnp.asarray(np.asarray(jax.device_get(ft), dtype=np.float64))
            cpu_segments = jax.device_put(segments, cpu)
            reference = jax.block_until_ready(jax.jit(oracle)(cpu_inputs, cpu_ct, cpu_ft, cpu_segments))
            assert all(x.dtype == jnp.float64 for x in reference)
            row = {"fixture": fixture, "shape": shape, "document_length": document_length, "implementations": {}}
            for implementation, actual in values.items():
                row["implementations"][implementation] = {
                    name: errors(a, b) for name, a, b in zip(names, actual, reference, strict=True)
                }
                row["implementations"][implementation]["rounded_oracle_gradients"] = {
                    name: errors(a, b.astype(a.dtype))
                    for name, a, b in zip(names[3:], actual[3:], reference[3:], strict=True)
                }
            row["candidate_vs_production"] = {
                name: errors(a, b)
                for name, a, b in zip(names, values["pallas_state"], values["production"], strict=True)
            }
        report["records"].append(row)
        print(json.dumps(row), flush=True)
        path = folder / "precision.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")


if __name__ == "__main__":
    main()
