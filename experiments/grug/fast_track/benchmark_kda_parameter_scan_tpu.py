# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check scan reconstruction through the full KDA parameterization on TPU v4."""

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

from experiments.grug.moe.kda import chunk_kda


def evaluate(inputs, cotangent, segment_ids, *, mode, scan_impl):
    def outputs(xs):
        return chunk_kda(*xs, chunk_size=16, segment_ids=segment_ids, scan_impl=scan_impl)

    def objective(xs):
        out, _ = outputs(xs)
        return jnp.sum(out * cotangent)

    return outputs(inputs) if mode == "forward" else jax.value_and_grad(objective)(inputs)


def compare(actual, reference):
    errors = []
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True):
        delta = jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))
        errors.append(
            {
                "shape": a.shape,
                "dtype": str(a.dtype),
                "max_abs_error": float(jnp.max(delta)),
                "mean_abs_error": float(jnp.mean(delta)),
                "bad_count": int(jnp.sum(~jnp.isclose(a, b, atol=1e-5, rtol=1e-5))),
                "finite": bool(jnp.all(jnp.isfinite(a))),
            }
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-vjp", action="store_true", help="Also test direct interleaving of split cotangents.")
    args = parser.parse_args()
    device = jax.local_devices()[0]
    if device.device_kind != "TPU v4":
        raise RuntimeError("This benchmark requires TPU v4")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        parser.error("Artifacts must stay in us-central2")
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    records = []
    with jax.default_device(device):
        for shape in ((1, 2, 112, 8), (2, 4, 1024, 64), (8, 4, 4096, 128)):
            keys = jax.random.split(jax.random.key(args.seed), 5)
            inputs = (
                *(jax.random.normal(key, shape, jnp.bfloat16) for key in keys[:3]),
                -0.05 * jax.nn.sigmoid(jax.random.normal(keys[3], shape)),
                jax.nn.sigmoid(jax.random.normal(keys[4], shape[:-1])),
            )
            cotangent = jax.random.normal(jax.random.key(args.seed + 1), shape)
            for document_length in (None, 1024, 83):
                segments = None if document_length is None else jnp.arange(shape[-2]) // document_length
                for mode in ("forward", "value_and_grad"):
                    reference = None
                    implementations = {"jax_reference": "parallel", "stacked": "parallel_stacked"}
                    if args.split_vjp:
                        implementations["stacked_split"] = "parallel_stacked_split"
                    for name, scan_impl in implementations.items():
                        record = {
                            "kernel": "chunk_kda",
                            "implementation": name,
                            "shape": shape,
                            "pass_mode": mode,
                            "leaf_order": (
                                ["output", "final_state"]
                                if mode == "forward"
                                else ["objective", "dq", "dk", "dv", "dg", "dbeta"]
                            ),
                            "document_length": document_length,
                            "dtype": "qkv=bf16,g_beta=fp32,internal_default_bf16",
                            "seed": args.seed,
                            "cotangent_seed": args.seed + 1,
                            "backend": "tpu",
                            "device_type": device.device_kind,
                            "device_count": 1,
                            "block_sizes": {"chunk": 16},
                            "git_sha": args.revision,
                            "jax_version": jax.__version__,
                            "jaxlib_version": jaxlib.__version__,
                            "xla_flags": os.environ.get("XLA_FLAGS", ""),
                            "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                            "atol": 1e-5,
                            "rtol": 1e-5,
                            "error": None,
                        }
                        try:
                            fn = partial(
                                evaluate,
                                mode=mode,
                                scan_impl=scan_impl,
                            )
                            start = time.monotonic()
                            compiled = jax.jit(fn).lower(inputs, cotangent, segments).compile()
                            record["compile_time"] = time.monotonic() - start
                            if shape[-1] == 128:
                                hlo = output / f"{name}-{mode}-doc{document_length}-hlo.txt"
                                hlo.write_text(compiled.as_text())
                                fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
                            actual = jax.block_until_ready(compiled(inputs, cotangent, segments))
                            if name == "jax_reference":
                                reference = actual
                            if reference is None:
                                raise RuntimeError("Reference computation failed")
                            errors = compare(actual, reference)
                            record["leaf_errors"] = errors
                            if any(e["bad_count"] or not e["finite"] for e in errors):
                                raise AssertionError("KDA output or parameter gradient differs from reference")
                            record["temporary_bytes"] = compiled.memory_analysis().temp_size_in_bytes
                            for _ in range(2):
                                jax.block_until_ready(compiled(inputs, cotangent, segments))
                            samples = []
                            for _ in range(5):
                                start = time.monotonic()
                                jax.block_until_ready(compiled(inputs, cotangent, segments))
                                samples.append(time.monotonic() - start)
                            record.update(steady_state_time=statistics.median(samples), steady_state_samples=samples)
                        except Exception as exc:
                            record["error"] = repr(exc)
                        records.append(record)
                        print(json.dumps(record), flush=True)
                        report = output / "parameter-scan-benchmark.json"
                        report.write_text(json.dumps(records, indent=2) + "\n")
                        fs.put_file(str(report), f"{prefix}/{report.name}")
    if any(r["error"] for r in records):
        raise RuntimeError("A parameter-scan candidate failed; see the retained report")


if __name__ == "__main__":
    main()
