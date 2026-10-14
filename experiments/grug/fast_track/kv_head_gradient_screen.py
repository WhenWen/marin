# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolate shared-KV gradient rounding before the full attention screen."""

import hashlib
import io
import json
from functools import partial

import fsspec
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from experiments.grug.fast_track.attention_mask_crop import SCENARIOS
from experiments.grug.fast_track.kv_head_gradient_tpu import sum_head_gradients

PROTOCOL = "https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6180942176"
FP32_FAILURES = (
    (0, 256, None, "reused"),
    (1, 1024, None, "packed"),
    (2, 1024, None, "packed"),
    (0, 2048, 129, "one-document"),
    (0, 2048, 129, "packed"),
)
BF16_FAILURES = (*FP32_FAILURES, (0, 256, None, "one-document"), (0, 1024, 129, "packed"))


def fixture_identity(row):
    return tuple(row[key] for key in ("seed", "length", "window", "scenario"))


def cast_sum(gradient):
    return jnp.sum(gradient, axis=2, keepdims=True, dtype=jnp.float32).astype(jnp.bfloat16)


def isolation_checks(source, device, folder):
    """Yield results and artifact paths, including failed numerical comparisons."""
    fs, root = fsspec.core.url_to_fs(source)
    manifest_bytes = fs.cat(f"{root}/analysis.json")
    manifest = json.loads(manifest_bytes)
    if {fixture_identity(row) for row in manifest["rows"]} != set(FP32_FAILURES) or len(manifest["rows"]) != 5:
        raise ValueError("Expected the five archived FP32-scaling failures")
    cpu = device.platform == "cpu"
    for row in manifest["rows"]:
        seed, length, window, scenario = fixture_identity(row)
        stem = f"belay-flash-dv-{seed}-{length}-{window}-{scenario}"
        raw = fs.cat(f"{root}/{stem}.npz")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != row["derived_array_sha256"]:
            raise ValueError(f"Archived input hash mismatch: {stem}")
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            heads = arrays["per_head_fp64"]
            oracle = arrays["oracle_fp64"]
        if heads.shape[2:] != (2, 128):
            raise ValueError("The isolated fixtures must contain exactly two query heads")
        represented = heads.astype(np.float32)
        expected = (represented[:, :, 0:1] + represented[:, :, 1:2]).astype(ml_dtypes.bfloat16).astype(np.float32)
        early = represented.astype(ml_dtypes.bfloat16).astype(np.float64).sum(axis=2, keepdims=True)
        early = early.astype(ml_dtypes.bfloat16).astype(np.float32)
        inputs = jax.device_put(represented, device)
        functions = {
            "pallas": partial(sum_head_gradients, num_kv_heads=1, interpret=cpu),
            "jax-cast": cast_sum,
        }
        for name, fn in functions.items():
            compiled = jax.jit(fn).lower(inputs).compile()
            result = jax.block_until_ready(compiled(inputs))
            actual = np.asarray(result).astype(np.float32)
            finite = bool(np.isfinite(actual).all() and np.isfinite(heads).all() and np.isfinite(oracle).all())
            matches_final = bool(np.array_equal(actual, expected))
            matches_early = bool(np.array_equal(actual, early))
            passed = finite and (matches_final if name == "pallas" or cpu else matches_early)
            key = f"isolation-{stem}-{name}"
            array_path = folder / f"{key}.npz"
            np.savez(array_path, actual=actual, expected=expected, early=early, oracle_fp64=oracle, inputs=represented)
            hlo_path = folder / f"{key}.hlo.txt"
            hlo_path.write_text(compiled.as_text())
            jaxpr_path = folder / f"{key}.jaxpr.txt"
            jaxpr_path.write_text(str(jax.make_jaxpr(fn)(inputs)))
            paths = (array_path, hlo_path, jaxpr_path)
            yield {
                "fixture": stem,
                "implementation": name,
                "backend": device.platform,
                "dtype": str(result.dtype),
                "finite": finite,
                "matches_fp32_sum_then_bf16": matches_final,
                "matches_bf16_heads_then_sum": matches_early,
                "passed": passed,
                "input_sha256": digest,
                "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "temporary_bytes": compiled.memory_analysis().temp_size_in_bytes,
                "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            }, paths


def control_reproduction(rows):
    """Check exact failed fixture/tensor/metric identities for both old paths."""
    expected_fixtures = {
        (seed, length, window, scenario)
        for seed in (0, 1, 2)
        for length in (256, 1024, 2048)
        for window in (None, 129)
        for scenario in SCENARIOS
    }
    complete = len(rows) == len(expected_fixtures) and {fixture_identity(row) for row in rows} == expected_fixtures
    results = {}
    for name, expected in (("flash-fp32-scale", FP32_FAILURES), ("flash-highest", BF16_FAILURES)):
        failures = set()
        for row in rows:
            for tensor, reference in row["errors"]["reference"].items():
                actual = row["errors"][name][tensor]
                for metric in ("max_abs_error", "relative_l2"):
                    if not (
                        reference["finite"] and actual["finite"] and actual[metric] <= 1.25 * reference[metric] + 0.001
                    ):
                        failures.add((*fixture_identity(row), tensor, metric))
        expected_failures = {(*identity, "dv", "max_abs_error") for identity in expected}
        results[name] = {
            "passed": complete and failures == expected_failures,
            "complete_fixture_grid": complete,
            "failures": sorted(failures, key=str),
            "expected": sorted(expected_failures, key=str),
        }
    return results
