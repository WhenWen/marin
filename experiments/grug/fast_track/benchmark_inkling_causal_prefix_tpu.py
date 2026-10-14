# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Causal-prefix Inkling attention candidate for the frozen TPU operator probe."""

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
from levanter.grug.attention._core import _reference_attention_math
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.benchmark_inkling_flash_tpu import blocked, fixture, reference64, values_and_gradients
from experiments.grug.fast_track.diagnose_kda_state_precision import errors


def prefix_block(q, k, v, band, segments, start, *, prefix, logits_dtype):
    queries = jax.lax.dynamic_slice_in_dim(q, start, REL_BIAS_BLOCK, axis=1)
    local_band = jax.lax.dynamic_slice_in_dim(band, start, REL_BIAS_BLOCK, axis=2)
    padded = jnp.pad(local_band, ((0, 0), (0, 0), (0, 0), (prefix, prefix)))
    bias = jax.lax.dynamic_slice_in_dim(padded, prefix + rel_extent_of_band(band) - start, prefix, axis=-1)
    query_segments = jax.lax.dynamic_slice_in_dim(segments, start, REL_BIAS_BLOCK, axis=1)
    causal = jnp.arange(prefix)[None, :] <= (start + jnp.arange(REL_BIAS_BLOCK))[:, None]
    mask = causal[None] & (query_segments[:, :, None] == segments[:, None, :prefix])
    return _reference_attention_math(queries, k[:, :prefix], v[:, :prefix], mask, bias, logits_dtype=logits_dtype)


def causal_prefix_attention(q, k, v, band, segments, *, quantum=1024, logits_dtype=jnp.float32):
    batch, length, heads, _ = q.shape
    if quantum % REL_BIAS_BLOCK or length % REL_BIAS_BLOCK:
        raise ValueError("Query length and prefix quantum must be multiples of 128")
    if k.shape[1] != length or v.shape[1] != length or segments.shape != (batch, length):
        raise ValueError("This probe requires causal self-attention and per-batch segment IDs")
    prefixes = tuple(min(i, length) for i in range(quantum, length + quantum, quantum))
    branches = tuple(partial(prefix_block, prefix=prefix, logits_dtype=logits_dtype) for prefix in prefixes)

    @jax.checkpoint
    def block(index):
        start = index * REL_BIAS_BLOCK
        return jax.lax.switch(start // quantum, branches, q, k, v, band, segments, start)

    blocks = jax.lax.map(block, jnp.arange(length // REL_BIAS_BLOCK))
    return blocks.transpose(1, 0, 2, 3, 4).reshape(batch, length, heads, v.shape[-1])


def probe_functions(quantum):
    return {
        name: jax.jit(partial(values_and_gradients, attention=attention))
        for name, attention in (
            ("reference", blocked),
            ("causal-prefix", partial(causal_prefix_attention, quantum=quantum)),
        )
    }


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
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "causal-prefix-attention"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "revision": args.revision,
        "status": "running",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "device_type": device.device_kind,
        "device_count_used": 1,
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
        path = folder / "causal-prefix-attention.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        upload(path)

    save()
    for length, quantum in ((256, 128), (1024, 128), (2048, 1024)):
        functions = probe_functions(quantum)
        for seed in (0, 1, 2):
            for scenario in ("one-document", "packed", "strong-logits"):
                host = fixture(seed, length, scenario)
                path = folder / f"fixture-{seed}-{length}-{scenario}.npz"
                np.savez(path, **dict(zip(("q", "k", "v", "band", "segments", "cotangent"), host, strict=True)))
                upload(path)
                with jax.enable_x64(True), jax.default_device(cpu):
                    inputs64 = tuple(
                        jax.device_put(x.astype(np.int32 if i == 4 else np.float64), cpu) for i, x in enumerate(host)
                    )
                    oracle = jax.jit(partial(values_and_gradients, attention=reference64))(*inputs64)
                    oracle = tuple(np.asarray(x) for x in jax.device_get(oracle))
                row = {"seed": seed, "length": length, "quantum": quantum, "scenario": scenario, "errors": {}}
                with jax.default_device(device):
                    inputs = tuple(jax.device_put(x, device) for x in host)
                    for name, fn in functions.items():
                        values = jax.block_until_ready(fn(*inputs))
                        row["errors"][name] = {
                            key: errors(actual, expected)
                            for key, actual, expected in zip(
                                ("output", "dq", "dk", "dv", "dband"), values, oracle, strict=True
                            )
                        }
                row["passed"] = all(
                    baseline["finite"]
                    and candidate["finite"]
                    and candidate[metric] <= 1.25 * baseline[metric] + 1e-3
                    for key, baseline in row["errors"]["reference"].items()
                    for candidate in (row["errors"]["causal-prefix"][key],)
                    for metric in ("max_abs_error", "relative_l2")
                )
                report["numerics"].append(row)
                save()
                print(json.dumps({k: row[k] for k in ("seed", "length", "scenario", "passed")}), flush=True)
    functions = probe_functions(1024)
    with jax.default_device(device):
        for batch, length, heads, kv_heads in ((1, 1024, 2, 1), (8, 4096, 6, 6)):
            shape = [batch, length, heads, kv_heads, 128]
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
                    {"shape": shape, "seed": seed, "sha256_q_k_v_band_ct": hashes, "generator": "jax.random.normal; key(seed+offset), offset=0..4; bfloat16"}
                )
                for scenario in ("one-document", "packed1024"):
                    segments = jnp.zeros((batch, length), jnp.int32)
                    if scenario == "packed1024":
                        segments = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32) // 1024, (batch, length))
                    inputs = q, k, v, band, segments, ct
                    case = {"shape": shape, "seed": seed, "scenario": scenario}
                    for name, fn in functions.items():
                        if name not in compiled:
                            report["phase"] = {**case, "stage": "compiling", "implementation": name}
                            save()
                            start = time.monotonic()
                            executable = fn.lower(*inputs).compile()
                            compiled[name] = executable
                            report["compilation"].append(
                                {"shape": shape, "implementation": name, "seconds": time.monotonic() - start,
                                 "temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
                                 "prefixes": list(range(1024, length + 1, 1024))}
                            )
                            path = folder / f"b{batch}-s{length}-{name}-hlo.txt"
                            path.write_text(executable.as_text())
                            upload(path)
                        for repeat in range(2):
                            values = jax.block_until_ready(compiled[name](*inputs))
                            finite = bool(jax.jit(lambda xs: jnp.stack([jnp.isfinite(x).all() for x in xs]).all())(values))
                            report["warmups"].append({**case, "implementation": name, "repeat": repeat, "finite": finite})
                            save()
                            if not finite:
                                raise ValueError(f"Nonfinite target outputs/gradients: {name}, {case}")
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
                            report["timings"].append({**case, "implementation": name, "repeat": repeat, "seconds": elapsed})
                        save()
                    medians = {name: statistics.median(values) for name, values in samples.items()}
                    reduction = 1 - medians["causal-prefix"] / medians["reference"]
                    result = {**case, "medians": medians, "reduction": reduction, "passed": reduction >= (0.05 if length == 4096 else -0.05)}
                    report["performance"].append(result)
                    print(json.dumps(result), flush=True)
                    save()
    report["gate_pass"] = all(row["passed"] for row in report["numerics"] + report["performance"])
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
