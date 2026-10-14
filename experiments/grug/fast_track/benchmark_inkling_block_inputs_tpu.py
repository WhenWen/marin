# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure attention with query and band blocks passed as scan inputs.

Protocol: https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6171089614
This operator probe has no optimizer or training loop.
"""

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
from levanter.grug.attention._core import AttentionMask, _blocked_inkling_math, _reference_attention_math
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band

from experiments.grug.fast_track import attention_mask_crop, flash_block_bias
from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.benchmark_inkling_flash_tpu import fixture, reference64, values_and_gradients
from experiments.grug.fast_track.diagnose_kda_state_precision import errors

TENSORS = ("output", "dq", "dk", "dv", "dband")
INPUTS = ("q", "k", "v", "band", "segments", "cotangent")


def block_inputs_attention(q, k, v, mask, rel_bias, *, block_size=REL_BIAS_BLOCK):
    """Map disjoint query/band blocks while keeping shared K/V reductions."""
    batch, length, heads, dim = q.shape
    if not isinstance(mask, AttentionMask):
        raise ValueError("The block-input probe requires a structured AttentionMask")
    if block_size != REL_BIAS_BLOCK or length % block_size or length != k.shape[1]:
        raise ValueError("The block-input probe requires 128-row blocks and self-attention")
    blocks = length // block_size
    queries = q.reshape(batch, blocks, block_size, heads, dim).transpose(1, 0, 2, 3, 4)
    bands = rel_bias.reshape(batch, heads, blocks, block_size, rel_bias.shape[-1]).transpose(2, 0, 1, 3, 4)

    @jax.checkpoint
    def block(inputs):
        query, band, index = inputs
        start = index * block_size
        keys = k.shape[1]
        padded = jnp.pad(band, ((0, 0), (0, 0), (0, 0), (keys, keys)))
        bias = jax.lax.dynamic_slice_in_dim(padded, keys + rel_extent_of_band(band) - start, keys, axis=-1)
        tile_mask = mask.materialize_mask(block_size, keys, query_offset=start)
        return _reference_attention_math(query, k, v, tile_mask, bias, logits_dtype=jnp.float32)

    output = jax.lax.map(block, (queries, bands, jnp.arange(blocks)))
    return output.transpose(1, 0, 2, 3, 4).reshape(batch, length, heads, v.shape[-1])


def attention(q, k, v, band, segments, *, implementation, sliding_window):
    mask = AttentionMask.causal(sliding_window=sliding_window).with_segment_ids(segments)
    return implementation(q, k, v, mask, band, block_size=REL_BIAS_BLOCK)


def probe_functions(window, candidate="block-inputs"):
    if candidate in ("flash-high", "flash-highest"):
        return {
            "reference": jax.jit(
                partial(
                    values_and_gradients,
                    attention=partial(attention, implementation=_blocked_inkling_math, sliding_window=window),
                )
            ),
            candidate: jax.jit(
                partial(flash_block_bias.values_and_gradients, precision=candidate.removeprefix("flash-"), window=window)
            ),
        }
    implementation = block_inputs_attention if candidate == "block-inputs" else attention_mask_crop.mask_crop_attention
    return {
        name: jax.jit(
            partial(
                values_and_gradients,
                attention=partial(attention, implementation=implementation, sliding_window=window),
            )
        )
        for name, implementation in (("reference", _blocked_inkling_math), (candidate, implementation))
    }


def finite_errors(actual, expected):
    result = errors(actual, expected)
    return {key: None if isinstance(value, float) and not np.isfinite(value) else value for key, value in result.items()}


def numerical_gate(row, candidate_name="block-inputs"):
    return all(
        baseline["finite"] and candidate["finite"] and candidate[metric] <= 1.25 * baseline[metric] + 1e-3
        for key, baseline in row["errors"]["reference"].items()
        for candidate in (row["errors"][candidate_name][key],)
        for metric in ("max_abs_error", "relative_l2")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument(
        "--candidate", choices=("block-inputs", "mask-crop", "flash-high", "flash-highest"), default="block-inputs"
    )
    args = parser.parse_args()
    candidate = args.candidate
    crop = candidate == "mask-crop"
    flash = candidate in ("flash-high", "flash-highest")
    expanded_fixtures = crop or flash
    scenarios = attention_mask_crop.SCENARIOS if expanded_fixtures else ("one-document", "packed", "strong-logits")
    make_fixture = attention_mask_crop.crop_fixture if expanded_fixtures else fixture
    cpu = jax.local_devices(backend="cpu")[0]
    if args.cpu_preflight:
        device = cpu
    else:
        require_tpu_rounding()
        device = jax.local_devices(backend="tpu")[0]
        if device.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
            raise ValueError("Use TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp/belay-attention-block-inputs")) / (
        "cpu" if args.cpu_preflight else "tpu"
    )
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    fs.makedirs(prefix, exist_ok=True)
    source_paths = (
        Path("experiments/grug/fast_track/benchmark_inkling_block_inputs_tpu.py"),
        Path("experiments/grug/fast_track/benchmark_inkling_flash_tpu.py"),
        Path("lib/levanter/src/levanter/grug/attention/_core.py"),
        Path("experiments/grug/fast_track/attention_mask_crop.py"),
        Path("experiments/grug/fast_track/flash_block_bias.py"),
    )
    report = {
        "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/96",
        "protocol_url": (
            flash_block_bias.PROTOCOL
            if flash
            else (
                attention_mask_crop.PROTOCOL
                if crop
                else "https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6171089614"
            )
        ),
        "candidate": candidate,
        "revision": args.revision,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "status": "running",
        "cpu_preflight": args.cpu_preflight,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "device_type": device.device_kind,
        "device_count_used": 1,
        "local_device_count_available": len(jax.local_devices(backend=device.platform)),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "numerics": [],
        "compilation": [],
        "fixtures": [],
        "warmups": [],
        "timings": [],
        "performance": [],
    }

    def upload(path):
        fs.put_file(str(path), f"{prefix}/{path.name}")

    def save():
        path = folder / f"attention-{candidate}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        upload(path)

    save()
    try:
        if flash:
            report["layout_checks"] = []
            report["phase"] = {"stage": "layout-checks"}
            save()
            for row, arrays in flash_block_bias.layout_checks(device):
                report["layout_checks"].append(row)
                if row["window"] is None:
                    name = f"layout-{row['backend']}-{row['length']}-{row['width']}-{row['seed']}-{row['dtype']}.npz"
                    path = folder / name
                    np.savez(path, **arrays)
                    upload(path)
                save()
            if args.cpu_preflight:
                report["status"] = "completed"
                report["gate_pass"] = True
                save()
                return
        if crop:
            with jax.default_device(cpu):
                report["support_checks"] = attention_mask_crop.support_check()
            save()
            if args.cpu_preflight:
                report["status"] = "completed"
                report["gate_pass"] = True
                save()
                return
        for length in (256, 1024, 2048):
            for window in (None, 129):
                functions = probe_functions(window, candidate)
                for seed in (0, 1, 2):
                    for scenario in scenarios:
                        report["phase"] = {
                            "stage": "numerics",
                            "length": length,
                            "window": window,
                            "seed": seed,
                            "scenario": scenario,
                        }
                        save()
                        host = make_fixture(seed, length, scenario)
                        path = folder / f"fixture-{seed}-{length}-{scenario}.npz"
                        if window is None:
                            np.savez(path, **dict(zip(INPUTS, host, strict=True)))
                            upload(path)
                        with jax.enable_x64(True), jax.default_device(cpu):
                            inputs64 = tuple(
                                jax.device_put(x.astype(np.int32 if i == 4 else np.float64), cpu)
                                for i, x in enumerate(host)
                            )
                            oracle_fn = jax.jit(
                                partial(values_and_gradients, attention=partial(reference64, sliding_window=window))
                            )
                            oracle = tuple(np.asarray(x) for x in jax.device_get(oracle_fn(*inputs64)))
                        row = {"seed": seed, "length": length, "window": window, "scenario": scenario, "errors": {}}
                        actuals = {}
                        with jax.default_device(device):
                            inputs = tuple(jax.device_put(x, device) for x in host)
                            if crop:
                                row["width_distribution"] = attention_mask_crop.width_distribution(inputs[4], window)
                            for name, fn in functions.items():
                                values = jax.block_until_ready(fn(*inputs))
                                actuals[name] = jax.device_get(values)
                                row["errors"][name] = {
                                    key: finite_errors(actual, expected)
                                    for key, actual, expected in zip(TENSORS, values, oracle, strict=True)
                                }
                                if seed == 0 and scenario == "one-document" and length == 256:
                                    path = folder / f"{name}-window-{window}-jaxpr.txt"
                                    path.write_text(str(jax.make_jaxpr(fn)(*inputs)))
                                    upload(path)
                        row["direct_errors"] = {
                            key: finite_errors(candidate, reference)
                            for key, candidate, reference in zip(
                                TENSORS, actuals[candidate], actuals["reference"], strict=True
                            )
                        }
                        row["passed"] = numerical_gate(row, candidate)
                        report["numerics"].append(row)
                        save()
                        print(
                            json.dumps({k: row[k] for k in ("seed", "length", "window", "scenario", "passed")}),
                            flush=True,
                        )
        if not all(row["passed"] for row in report["numerics"]):
            raise ValueError("Independent FP64 numerical gate failed")
        if not args.cpu_preflight:
            with jax.default_device(device):
                for batch, length, heads, kv_heads in ((1, 1024, 2, 1), (8, 4096, 6, 6)):
                    shape = [batch, length, heads, kv_heads, 128]
                    for window in (None, 129 if length == 1024 else 2048):
                        functions = probe_functions(window, candidate)
                        compiled = {}
                        for seed in (10, 20, 30):
                            q, k, v, band, ct = (
                                jax.random.normal(jax.random.key(seed + offset), dims, jnp.bfloat16)
                                for offset, dims in enumerate(
                                    (
                                        (batch, length, heads, 128),
                                        (batch, length, kv_heads, 128),
                                        (batch, length, kv_heads, 128),
                                        (batch, heads, length, 1152),
                                        (batch, length, heads, 128),
                                    )
                                )
                            )
                            hashes = [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in (q, k, v, band, ct)]
                            report["fixtures"].append(
                                {"shape": shape, "window": window, "seed": seed, "sha256_q_k_v_band_ct": hashes}
                            )
                            for scenario in (
                                ("one-document", "packed1024", "mixed")
                                if expanded_fixtures
                                else ("one-document", "packed1024")
                            ):
                                segments = jnp.zeros((batch, length), jnp.int32)
                                if scenario == "packed1024":
                                    segments = jnp.broadcast_to(
                                        jnp.arange(length, dtype=jnp.int32) // 1024, (batch, length)
                                    )
                                if scenario == "mixed":
                                    segments = jnp.arange(length, dtype=jnp.int32)[None] // (
                                        257 + 137 * jnp.arange(batch, dtype=jnp.int32)[:, None]
                                    )
                                inputs = q, k, v, band, segments, ct
                                case = {"shape": shape, "window": window, "seed": seed, "scenario": scenario}
                                if crop:
                                    case["width_distribution"] = attention_mask_crop.width_distribution(segments, window)
                                for name, fn in functions.items():
                                    if name not in compiled:
                                        report["phase"] = {**case, "stage": "compiling", "implementation": name}
                                        save()
                                        start = time.monotonic()
                                        executable = fn.lower(*inputs).compile()
                                        compiled[name] = executable
                                        report["compilation"].append(
                                            {
                                                "shape": shape,
                                                "window": window,
                                                "implementation": name,
                                                "seconds": time.monotonic() - start,
                                                "temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
                                            }
                                        )
                                        path = folder / f"b{batch}-s{length}-window-{window}-{name}-hlo.txt"
                                        path.write_text(executable.as_text())
                                        upload(path)
                                    for repeat in range(2):
                                        values = jax.block_until_ready(compiled[name](*inputs))
                                        finite = bool(
                                            jax.jit(lambda xs: jnp.stack([jnp.isfinite(x).all() for x in xs]).all())(
                                                values
                                            )
                                        )
                                        report["warmups"].append(
                                            {**case, "implementation": name, "repeat": repeat, "finite": finite}
                                        )
                                        save()
                                        if not finite:
                                            raise ValueError(f"Nonfinite outputs/gradients: {name}, {case}")
                                samples = {name: [] for name in functions}
                                for repeat in range(8):
                                    names = list(functions)
                                    if repeat % 2:
                                        names.reverse()
                                    for name in names:
                                        start = time.monotonic()
                                        jax.block_until_ready(compiled[name](*inputs))
                                        elapsed = time.monotonic() - start
                                        samples[name].append(elapsed)
                                        report["timings"].append(
                                            {**case, "implementation": name, "repeat": repeat, "seconds": elapsed}
                                        )
                                    save()
                                medians = {name: statistics.median(values) for name, values in samples.items()}
                                reduction = 1 - medians[candidate] / medians["reference"]
                                result = {
                                    **case,
                                    "medians": medians,
                                    "reduction": reduction,
                                    "passed": reduction >= (0.05 if length == 4096 else -0.05),
                                }
                                report["performance"].append(result)
                                print(json.dumps(result), flush=True)
                                save()
        report["gate_pass"] = all(row["passed"] for row in report["numerics"] + report["performance"])
        report["status"] = "completed"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
