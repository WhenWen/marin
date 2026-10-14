# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test the local TPU state pass through Belay's complete KDA parameterization."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import fsspec
import jax
import jax.numpy as jnp
import jaxlib

from experiments.grug.fast_track.benchmark_kda_parameter_scan_tpu import compare
from experiments.grug.moe import kda
from experiments.grug.moe.kda_state_tpu import state_pass


def tpu_recurrence(prep, state, mm, lead, dk, dv, *, scan_impl):
    """Diagnostic adapter; the caller places all arrays on one explicit device."""
    n, c = prep.kw.shape[-3:-1]

    def flatten(x):
        return x.reshape(-1, n, *x.shape[-2:])

    out, final = state_pass(
        flatten(prep.q_inflate),
        flatten(prep.attn),
        flatten(prep.kw),
        flatten(prep.k_cumdecay),
        flatten(prep.v_pseudo),
        prep.decay.reshape(-1, n, 1, dk),
        state.reshape(-1, dk, dv),
    )
    return out.reshape(*lead, n, c, dv), final.reshape(*lead, dk, dv)


def evaluate(inputs, cotangent, final_cotangent, segments, *, mode):
    def outputs(xs):
        return kda.chunk_kda(
            *xs[:5],
            initial_state=xs[5],
            chunk_size=16,
            segment_ids=segments,
            scan_impl="parallel_stacked_split",
        )

    def objective(xs):
        output, final = outputs(xs)
        return jnp.sum(output * cotangent) + jnp.sum(final * final_cotangent)

    return outputs(inputs) if mode == "forward" else jax.value_and_grad(objective)(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    device = jax.local_devices()[0]
    if device.device_kind != "TPU v4":
        raise RuntimeError("This experiment requires TPU v4")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        parser.error("Artifacts must stay in us-central2")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    records = []
    # No SPMD inference: the prototype accepts fully device-local arrays only.
    with jax.default_device(device):
        for fixture, shape, document_length in (
            ("random", (1, 2, 128, 128), 37),
            ("repeated", (1, 2, 128, 128), 37),
            ("strong-decay", (1, 2, 128, 128), 37),
            ("medium", (2, 4, 1024, 128), 83),
            ("belay-local", (8, 4, 4096, 128), 1024),
        ):
            keys = jax.random.split(jax.random.key(0), 7)
            q, k, v = (jax.random.normal(key, shape, jnp.bfloat16) for key in keys[:3])
            if fixture == "repeated":
                k = jnp.broadcast_to(k[..., :1, :], shape)
            gate_scale = 5.0 if fixture == "strong-decay" else 0.05
            g = -gate_scale * jax.nn.sigmoid(jax.random.normal(keys[3], shape))
            beta = jax.nn.sigmoid(jax.random.normal(keys[4], shape[:-1]))
            initial = 0.01 * jax.random.normal(keys[5], (*shape[:2], 128, 128))
            inputs = (q, k, v, g, beta, initial)
            cotangent = jax.random.normal(keys[6], shape)
            final_cotangent = jax.random.normal(jax.random.key(8), initial.shape)
            segments = jnp.arange(shape[-2]) // document_length
            for mode in ("forward", "value_and_grad"):
                reference = None
                for name in ("production", "pallas_state"):
                    row = {
                        "kernel": "chunk_kda",
                        "implementation": name,
                        "fixture": fixture,
                        "shape": shape,
                        "pass_mode": mode,
                        "dtype": "qkv=bf16, gates/state=fp32",
                        "backend": "tpu",
                        "device_type": device.device_kind,
                        "device_count": 1,
                        "block_sizes": {"chunk": 16, "state": [128, 128]},
                        "git_sha": args.revision,
                        "seed": 0,
                        "document_length": document_length,
                        "jax_version": jax.__version__,
                        "jaxlib_version": jaxlib.__version__,
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                        "compile_time": None,
                        "steady_state_time": None,
                        "error": None,
                        "atol": 1e-5,
                        "rtol": 1e-5,
                        "leaf_order": (
                            ["output", "final_state"]
                            if mode == "forward"
                            else ["objective", "dq", "dk", "dv", "dg", "dbeta", "dinitial"]
                        ),
                    }
                    try:
                        fn = jax.jit(lambda xs, ct, ft, seg, mode=mode: evaluate(xs, ct, ft, seg, mode=mode))
                        start = time.monotonic()
                        if name == "pallas_state":
                            with patch.object(kda, "_parallel_chunk_recurrence", tpu_recurrence):
                                compiled = fn.lower(inputs, cotangent, final_cotangent, segments).compile()
                        else:
                            compiled = fn.lower(inputs, cotangent, final_cotangent, segments).compile()
                        row["compile_time"] = time.monotonic() - start
                        hlo = folder / f"{fixture}-{mode}-{name}.hlo.txt"
                        hlo.write_text(compiled.as_text())
                        fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
                        actual = jax.block_until_ready(compiled(inputs, cotangent, final_cotangent, segments))
                        if reference is None:
                            reference = actual
                        errors = compare(actual, reference)
                        row["errors"] = errors
                        row["parity_pass"] = all(e["finite"] and e["bad_count"] == 0 for e in errors)
                        row["temporary_bytes"] = compiled.memory_analysis().temp_size_in_bytes
                        samples = []
                        for i in range(7):
                            start = time.monotonic()
                            jax.block_until_ready(compiled(inputs, cotangent, final_cotangent, segments))
                            if i >= 2:
                                samples.append(time.monotonic() - start)
                        row["steady_state_samples"] = samples
                        row["steady_state_time"] = statistics.median(samples)
                        if not row["parity_pass"]:
                            row["error"] = "value/gradient parity failed; timing is diagnostic only"
                    except Exception as exc:
                        row["error"] = repr(exc)
                    records.append(row)
                    print(json.dumps(row), flush=True)
                    report = folder / "state-benchmark.json"
                    report.write_text(json.dumps(records, indent=2) + "\n")
                    fs.put_file(str(report), f"{prefix}/{report.name}")
    if any(r["error"] for r in records):
        raise RuntimeError("State-pass candidate failed; see retained diagnostics")


if __name__ == "__main__":
    main()
