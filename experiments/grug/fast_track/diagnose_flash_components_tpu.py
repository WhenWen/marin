# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate bias layout costs from the installed TPU FlashAttention kernel."""

import argparse
import hashlib
import json
import math
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
from jax.experimental.pallas.ops.tpu import flash_attention as flash_module
from levanter.callbacks.profiler import ProfileOptionsConfig

from experiments.grug.fast_track import benchmark_inkling_flash_tpu as probe
from experiments.grug.fast_track.belay_tpu import require_tpu_rounding

PROTOCOL = "https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6176653159"
COMPONENTS = ("bias-expansion", "preexpanded-attention", "bias-projection", "original-wrapper")


def preexpanded_attention(q, k, v, dense_bias, segments):
    scaled_q = (q * (1 / math.sqrt(q.shape[-1]))).astype(q.dtype)
    return probe.flash_core(scaled_q, k, v, dense_bias, segments)


def bias_projection(gradient):
    return probe.expanded_bias_bwd(1152, None, gradient)[0]


@jax.jit
def finite(values):
    return jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(values)]).all()


@jax.jit
def differences(actual, expected):
    return tuple(
        jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))
        for a, b in zip(actual, expected, strict=True)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    require_tpu_rounding()
    device = jax.local_devices(backend="tpu")[0]
    if device.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Use TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp/belay-flash-components"))
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "revision": args.revision,
        "protocol_url": PROTOCOL,
        "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/96",
        "status": "running",
        "device": str(device),
        "device_count_used": 1,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(probe.__file__), Path(flash_module.__file__))
        },
        "compilation": [],
        "warmups": [],
        "timings": [],
    }

    def upload(path):
        fs.put_file(str(path), f"{prefix}/{path.relative_to(folder)}")

    def save():
        path = folder / "components.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        upload(path)

    def compile_and_measure(name, function, inputs):
        report["phase"] = {"component": name, "stage": "compiling"}
        save()
        start = time.monotonic()
        compiled = jax.jit(function).lower(*inputs).compile()
        elapsed = time.monotonic() - start
        path = folder / f"{name}-hlo.txt"
        path.write_text(compiled.as_text())
        report["compilation"].append(
            {
                "component": name,
                "seconds": elapsed,
                "temporary_bytes": compiled.memory_analysis().temp_size_in_bytes,
                "hlo_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        upload(path)
        for stage, repeats in (("warmups", 2), ("timings", 3)):
            for repeat in range(repeats):
                report["phase"] = {"component": name, "stage": stage, "repeat": repeat}
                save()
                start = time.monotonic()
                values = jax.block_until_ready(compiled(*inputs))
                elapsed = time.monotonic() - start
                is_finite = bool(finite(values))
                row = {"component": name, "repeat": repeat, "seconds": elapsed, "finite": is_finite}
                report[stage].append(row)
                save()
                print(json.dumps({"stage": stage, **row}), flush=True)
                if not is_finite:
                    raise ValueError(f"Nonfinite component: {name}")
        return compiled, values

    save()
    try:
        with jax.default_device(device):
            q, k, v, band, cotangent = (
                jax.random.normal(jax.random.key(seed), shape, jnp.bfloat16)
                for seed, shape in (
                    (10, (8, 4096, 6, 128)),
                    (11, (8, 4096, 6, 128)),
                    (12, (8, 4096, 6, 128)),
                    (13, (8, 6, 4096, 1152)),
                    (14, (8, 4096, 6, 128)),
                )
            )
            segments = jnp.broadcast_to(jnp.arange(4096, dtype=jnp.int32) // 1024, (8, 4096))
            report["input_sha256"] = {
                name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                for name, value in zip(
                    ("q", "k", "v", "band", "cotangent", "segments"),
                    (q, k, v, band, cotangent, segments),
                    strict=True,
                )
            }
            _, dense = compile_and_measure("bias-expansion", partial(probe.expanded_bias, width=1152), (band,))
            _, kernel_values = compile_and_measure(
                "preexpanded-attention",
                partial(probe.values_and_gradients, attention=preexpanded_attention),
                (q, k, v, dense, segments, cotangent),
            )
            _, band_gradient = compile_and_measure("bias-projection", bias_projection, (kernel_values[-1],))
            inputs = q, k, v, band, segments, cotangent
            compiled, original = compile_and_measure(
                "original-wrapper", partial(probe.values_and_gradients, attention=probe.flash_gather), inputs
            )
            composed = (*kernel_values[:-1], band_gradient)
            report["composition_max_absolute_differences"] = dict(
                zip(("output", "dq", "dk", "dv", "dband"), map(float, differences(composed, original)), strict=True)
            )
            report["phase"] = {"component": "original-wrapper", "stage": "profiling"}
            save()
            options = ProfileOptionsConfig(
                host_tracer_level=1, python_tracer_level=0, enable_hlo_proto=True
            ).build_jax_profile_options()
            with jax.profiler.trace(str(folder / "profile"), profiler_options=options):
                with jax.profiler.TraceAnnotation("original_flash_wrapper"):
                    jax.block_until_ready(compiled(*inputs))
            for path in (folder / "profile").rglob("*"):
                if path.is_file():
                    upload(path)
        report["medians"] = {
            name: statistics.median(row["seconds"] for row in report["timings"] if row["component"] == name)
            for name in COMPONENTS
        }
        report["status"] = "completed"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
