# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure fully local Inkling attention through the installed TPU FlashAttention.

Inputs belong to one chip; this probe does not provide distributed dispatch.
Dense bias expansion and its gradient are included in every timing.
"""

import argparse
import functools
import json
import math
import os
import statistics
import time
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds, flash_attention
from levanter.grug.attention._core import AttentionMask, _blocked_inkling_math, align_kv_heads
from levanter.grug.attention._inkling_relpos import dense_rel_bias

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.diagnose_kda_state_precision import errors


def blocked(q, k, v, band, segments):
    mask = AttentionMask.causal().with_segment_ids(segments)
    return _blocked_inkling_math(q, k, v, mask, band, block_size=128)


def flash_core(q, k, v, bias, segments):
    k = align_kv_heads(k, num_q_heads=q.shape[2])
    v = align_kv_heads(v, num_q_heads=q.shape[2])
    output = flash_attention(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        bias,
        SegmentIds(segments, segments),
        causal=True,
        sm_scale=1.0,
    )
    return output.transpose(0, 2, 1, 3)


def flash(q, k, v, band, segments):
    """Fully local attention with dense Inkling bias and packed causal masking."""
    scaled_q = (q * (1 / math.sqrt(q.shape[-1]))).astype(q.dtype)
    return flash_core(scaled_q, k, v, dense_rel_bias(band, q.shape[1], k.shape[1]), segments)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def expanded_bias(band, width):
    """Expand full self-attention bias, with a one-to-one gather for its adjoint."""
    if band.shape[-1] != width:
        raise ValueError("Static band width does not match the input")
    return dense_rel_bias(band, band.shape[-2], band.shape[-2])


def expanded_bias_fwd(band, width):
    return expanded_bias(band, width), None


def expanded_bias_bwd(width, _, gradient):
    length = gradient.shape[-2]
    keys = (jnp.arange(length)[:, None] // 128) * 128 + jnp.arange(width)[None, :] - (width - 128)
    valid = (keys >= 0) & (keys < gradient.shape[-1])
    indices = jnp.broadcast_to(jnp.clip(keys, 0, gradient.shape[-1] - 1), (*gradient.shape[:-1], width))
    values = jnp.take_along_axis(gradient, indices, axis=-1)
    return (jnp.where(valid, values, 0),)


expanded_bias.defvjp(expanded_bias_fwd, expanded_bias_bwd)


def flash_gather(q, k, v, band, segments):
    scaled_q = (q * (1 / math.sqrt(q.shape[-1]))).astype(q.dtype)
    return flash_core(scaled_q, k, v, expanded_bias(band, band.shape[-1]), segments)


def flash_gather_fp32(q, k, v, band, segments):
    scaled_q = (q * (1 / math.sqrt(q.shape[-1]))).astype(q.dtype).astype(jnp.float32)
    with jax.default_matmul_precision("highest"):
        output = flash_core(
            scaled_q,
            k.astype(jnp.float32),
            v.astype(jnp.float32),
            expanded_bias(band.astype(jnp.float32), band.shape[-1]),
            segments,
        )
    return output.astype(q.dtype)


def bias_values_and_gradients(band, ct):
    output, pullback = jax.vjp(lambda x: expanded_bias(x, band.shape[-1]), band)
    return output, pullback(ct)[0]


def bias_adjoint_check(cpu, device):
    """Check the linear layout map against independent NumPy indexing."""
    results = []
    for seed in (0, 1, 2):
        for length, width in ((128, 256), (256, 256), (1024, 1152)):
            rng = np.random.default_rng(seed)
            band = rng.normal(size=(1, 2, length, width)).astype(ml_dtypes.bfloat16)
            ct = rng.normal(size=(1, 2, length, length)).astype(ml_dtypes.bfloat16)
            expected = np.zeros(ct.shape, dtype=ml_dtypes.bfloat16)
            expected_grad = np.zeros(band.shape, dtype=ml_dtypes.bfloat16)
            for row in range(length):
                for column in range(width):
                    key = (row // 128) * 128 + column - (width - 128)
                    if 0 <= key < length:
                        expected[:, :, row, key] = band[:, :, row, column]
                        expected_grad[:, :, row, column] = ct[:, :, row, key]

            for backend, target, dtype in (("cpu", cpu, np.float64), ("tpu", device, ml_dtypes.bfloat16)):
                with jax.enable_x64(backend == "cpu"), jax.default_device(target):
                    actual = jax.jit(bias_values_and_gradients)(
                        jax.device_put(band.astype(dtype), target), jax.device_put(ct.astype(dtype), target)
                    )
                    for value, oracle in zip(jax.device_get(actual), (expected, expected_grad), strict=True):
                        np.testing.assert_array_equal(np.asarray(value), oracle.astype(dtype))
                results.append({"seed": seed, "length": length, "width": width, "backend": backend, "exact": True})
    return results


def reference64(q, k, v, band, segments, *, sliding_window=None):
    """Independent dense attention, including the block-relative bias indexing."""
    length = q.shape[1]
    repeat = q.shape[2] // k.shape[2]
    k = jnp.repeat(k, repeat, axis=2)
    v = jnp.repeat(v, repeat, axis=2)
    positions = np.arange(length)
    columns = positions[None, :] - (positions[:, None] // 128) * 128 + band.shape[-1] - 128
    valid = (columns >= 0) & (columns < band.shape[-1])
    indices = np.clip(columns, 0, band.shape[-1] - 1)
    bias = band[:, :, np.arange(length)[:, None], indices]
    bias = jnp.where(valid[None, None], bias, 0)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q / math.sqrt(q.shape[-1]), k) + bias
    mask = (positions[:, None] >= positions[None, :])[None] & (segments[:, :, None] == segments[:, None, :])
    if sliding_window is not None:
        mask = mask & (positions[None, :] >= positions[:, None] - sliding_window + 1)[None]
    scores = jnp.where(mask[:, None], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def values_and_gradients(q, k, v, band, segments, cotangent, *, attention):
    output, pullback = jax.vjp(lambda a, b, c, d: attention(a, b, c, d, segments), q, k, v, band)
    return output, *pullback(cotangent)


def fixture(seed, length, scenario):
    rng = np.random.default_rng(seed)
    scale = 4 if scenario == "strong-logits" else 1
    q = (scale * rng.normal(size=(1, length, 2, 128))).astype(ml_dtypes.bfloat16)
    k = (scale * rng.normal(size=(1, length, 1, 128))).astype(ml_dtypes.bfloat16)
    v = rng.normal(size=k.shape).astype(ml_dtypes.bfloat16)
    band = (0.1 * rng.normal(size=(1, 2, length, 256))).astype(ml_dtypes.bfloat16)
    segments = np.zeros((1, length), dtype=np.int32)
    if scenario == "packed":
        segments[:, 37:] += 1
        segments[:, 113:] += 1
    ct = rng.normal(size=q.shape).astype(ml_dtypes.bfloat16)
    return q, k, v, band, segments, ct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=("dense-bias", "inverse-gather", "inverse-gather-fp32"),
        default=["dense-bias"],
    )
    args = parser.parse_args()
    require_tpu_rounding()
    device = jax.local_devices(backend="tpu")[0]
    cpu = jax.local_devices(backend="cpu")[0]
    if device.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("This probe requires TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "inkling-flash"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "kernel": "inkling-attention",
        "revision": args.revision,
        "git_sha": args.revision,
        "dtype": "bfloat16 external inputs and outputs; inverse-gather-fp32 uses float32 internally",
        "backend": "tpu",
        "device_count": 1,
        "block_sizes": "128 query/key forward and backward defaults",
        "jax_version": jax.__version__,
        "device_type": device.device_kind,
        "device_count_used": 1,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "numerics": [],
        "compilation": [],
        "warmups": [],
        "timings": [],
        "status": "running",
    }

    def save():
        path = folder / "inkling-flash.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    candidates = {"dense-bias": flash, "inverse-gather": flash_gather, "inverse-gather-fp32": flash_gather_fp32}
    implementations = {"blocked": blocked, **{name: candidates[name] for name in args.candidates}}
    if any(name.startswith("inverse-gather") for name in args.candidates):
        report["bias_adjoint"] = bias_adjoint_check(cpu, device)
        save()
    functions = {
        name: jax.jit(functools.partial(values_and_gradients, attention=fn)) for name, fn in implementations.items()
    }
    for seed in (0, 1, 2):
        for length in (128, 256):
            for scenario in ("one-document", "packed", "strong-logits"):
                host_inputs = fixture(seed, length, scenario)
                with jax.enable_x64(True), jax.default_device(cpu):
                    oracle_inputs = tuple(
                        jax.device_put(x.astype(np.int32 if i == 4 else np.float64), cpu)
                        for i, x in enumerate(host_inputs)
                    )
                    oracle = jax.jit(functools.partial(values_and_gradients, attention=reference64))(*oracle_inputs)
                    oracle = tuple(np.asarray(x) for x in jax.device_get(oracle))
                with jax.default_device(device):
                    inputs = tuple(jax.device_put(x, device) for x in host_inputs)
                    row = {"seed": seed, "length": length, "scenario": scenario, "implementations": {}}
                    for name, fn in functions.items():
                        values = jax.block_until_ready(fn(*inputs))
                        row["implementations"][name] = {
                            key: errors(actual, expected)
                            for key, actual, expected in zip(
                                ("output", "dq", "dk", "dv", "dband"), values, oracle, strict=True
                            )
                        }
                    row["passed"] = {}
                    for name in args.candidates:
                        passed = True
                        for key in ("output", "dq", "dk", "dv", "dband"):
                            baseline = row["implementations"]["blocked"][key]
                            candidate = row["implementations"][name][key]
                            passed &= baseline["finite"] and candidate["finite"]
                            passed &= candidate["max_abs_error"] <= 1.25 * baseline["max_abs_error"] + 1e-3
                            passed &= candidate["relative_l2"] <= 1.25 * baseline["relative_l2"] + 1e-3
                        row["passed"][name] = passed
                    report["numerics"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
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
                compile_time = time.monotonic() - start
                compiled[name] = executable
                report["compilation"].append(
                    {
                        "shape": shape,
                        "implementation": name,
                        "compile_time": compile_time,
                        "temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
                    }
                )
                path = folder / f"b{batch}-s{length}-{name}-hlo.txt"
                path.write_text(executable.as_text())
                fs.put_file(str(path), f"{prefix}/{path.name}")
                save()
                for repeat in range(2):
                    report["phase"] = {"stage": "warmup", "shape": shape, "implementation": name, "repeat": repeat}
                    save()
                    start = time.monotonic()
                    values = jax.block_until_ready(executable(*inputs))
                    elapsed = time.monotonic() - start
                    finite = bool(jax.jit(lambda xs: jnp.stack([jnp.isfinite(x).all() for x in xs]).all())(values))
                    del values
                    report["warmups"].append(
                        {"shape": shape, "implementation": name, "repeat": repeat, "seconds": elapsed, "finite": finite}
                    )
                    save()
                    if not finite:
                        raise ValueError(f"Nonfinite target outputs/gradients: {name}, {shape}")
            for repeat in range(8):
                names = list(implementations)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    report["phase"] = {"stage": "timing", "shape": shape, "implementation": name, "repeat": repeat}
                    save()
                    start = time.monotonic()
                    jax.block_until_ready(compiled[name](*inputs))
                    report["timings"].append(
                        {"shape": shape, "implementation": name, "repeat": repeat, "seconds": time.monotonic() - start}
                    )
                    save()
                    print(json.dumps(report["timings"][-1]), flush=True)
            save()
            medians = {
                name: statistics.median(
                    r["seconds"] for r in report["timings"] if r["shape"] == shape and r["implementation"] == name
                )
                for name in implementations
            }
            print(json.dumps({"shape": shape, "medians": medians}), flush=True)
        report["numerical_gate_passed"] = {
            name: all(r["passed"][name] for r in report["numerics"]) for name in args.candidates
        }
        report["status"] = "completed"
        save()


if __name__ == "__main__":
    main()
