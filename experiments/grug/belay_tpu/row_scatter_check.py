# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare expert-row scatter reductions with the selected scan on TPU."""

import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.operator_check import initialize_operator_check
from experiments.grug.belay_tpu.row_scatter import combine_scattered_routes, sum_scattered_routes
from experiments.grug.belay_tpu.streaming_combine import combine_streaming, sum_routes


def evaluate(implementation, mode, rows, order, inverse, weights, gradient):
    if mode == "dispatch_backward":
        if implementation == "scan":
            return (sum_routes(rows, inverse.reshape(weights.shape), implementation="scan"),)
        return (sum_scattered_routes(rows, order, inverse, weights.shape[1], reduction=implementation),)
    combine = (
        functools.partial(combine_streaming, implementation="scan", scalar_permutation="sort")
        if implementation == "scan"
        else functools.partial(combine_scattered_routes, reduction=implementation)
    )

    def function(x, w):
        return combine(x, order, inverse, w)

    if mode == "combine_forward":
        return (function(rows, weights),)
    output, pullback = jax.vjp(function, rows, weights)
    return output, *pullback(gradient)


def oracle(rows, order, inverse, weights, gradient):
    """Compute all route derivatives in FP64 with bounded host working arrays."""
    tokens, topk = weights.shape
    width = rows.shape[1]
    output = np.empty((tokens, width), np.float64)
    unweighted = np.empty_like(output)
    d_weights = np.empty(weights.shape, np.float64)
    d_rows = np.empty(rows.shape, np.float64)
    for start in range(0, tokens, 1024):
        end = min(start + 1024, tokens)
        slots = np.asarray(rows[inverse[start * topk : end * topk]], np.float64).reshape(-1, topk, width)
        upstream = np.asarray(gradient[start:end], np.float64)
        output[start:end] = np.sum(slots * np.asarray(weights[start:end], np.float64)[..., None], axis=1)
        unweighted[start:end] = slots.sum(axis=1)
        d_weights[start:end] = np.sum(slots * upstream[:, None, :], axis=-1)
    for start in range(0, rows.shape[0], 4096):
        end = min(start + 4096, rows.shape[0])
        token = order[start:end] // topk
        coefficient = weights.reshape(-1)[order[start:end]]
        d_rows[start:end] = np.asarray(gradient[token], np.float64) * np.asarray(coefficient, np.float64)[:, None]
    return {
        "combine_forward": (output,),
        "combine_vjp": (output, d_rows, d_weights),
        "dispatch_backward": (unweighted,),
    }


def main():
    output, operator_topology, reference_topology = initialize_operator_check()
    tracker.log_summary({"kernel_check/input_seed": 444, "kernel_check/reference": "host_fp64_token_slot_arithmetic"})
    failures = []
    records = []
    with jax.default_device(jax.local_devices()[0]):
        tokens, topk, width = 32768, 10, 256
        for case in ("balanced", "skewed"):
            rng = np.random.default_rng(444)
            labels = rng.integers(384, size=tokens * topk)
            if case == "skewed":
                labels[rng.random(labels.size) < 0.9] = 0
            order = jnp.asarray(np.argsort(labels, kind="stable"), jnp.int32)
            inverse = jnp.argsort(order)
            rows = jnp.asarray(rng.standard_normal((tokens * topk, width), dtype=np.float32), jnp.bfloat16)
            rows = rows.at[9 * rows.shape[0] // 10 :].set(0)
            weights = jnp.asarray(rng.random((tokens, topk), dtype=np.float32), jnp.bfloat16).at[:, 0].set(0)
            gradient = jnp.asarray(rng.standard_normal((tokens, width), dtype=np.float32), jnp.bfloat16)
            inputs = (rows, order, inverse, weights, gradient)
            expected = oracle(*(np.asarray(value) for value in inputs))
            for mode in ("combine_forward", "combine_vjp", "dispatch_backward"):
                compiled = {}
                mode_records = {}
                for implementation in ("scan", "restore", "segment"):
                    function = jax.jit(functools.partial(evaluate, implementation, mode))
                    start = time.perf_counter()
                    executable = function.lower(*inputs).compile()
                    compile_time = time.perf_counter() - start
                    start = time.perf_counter()
                    actual = jax.block_until_ready(executable(*inputs))
                    first_execution_time = time.perf_counter() - start
                    errors = measure_errors(actual, expected[mode], 2e-2)
                    passed = all(
                        np.isfinite(error["normalized_max"]) and error["normalized_max"] <= 1 for error in errors
                    )
                    if not passed:
                        failures.append((case, mode, implementation))
                    compiled[implementation] = executable
                    mode_records[implementation] = {
                        "stage": "row_scatter_comparison",
                        "kernel": mode,
                        "implementation": implementation,
                        "case": case,
                        "shape": [tokens, topk, width],
                        "dtype": str(rows.dtype),
                        "weight_dtype": str(weights.dtype),
                        "backend": jax.default_backend(),
                        "device_type": jax.local_devices()[0].device_kind,
                        "device_count": 1,
                        "rank": jax.process_index(),
                        "operator_topology": operator_topology,
                        "reference_topology": reference_topology,
                        "block_sizes": {},
                        "oracle": "host_fp64_token_slot_arithmetic",
                        "passed": passed,
                        "error": errors,
                        "compile_time": compile_time,
                        "first_execution_time": first_execution_time,
                        "samples": [],
                        "memory": str(executable.memory_analysis()),
                        "temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
                        "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                    }
                    completed = {**mode_records[implementation], "stage": "row_scatter_compiled"}
                    print(json.dumps(completed), flush=True)
                    with (output / "row-scatter-compiled.jsonl").open("a") as file:
                        file.write(json.dumps(completed) + "\n")
                for iteration in range(18):
                    sequence = ("scan", "restore", "segment") if iteration % 2 == 0 else ("segment", "restore", "scan")
                    for implementation in sequence:
                        start = time.perf_counter()
                        jax.block_until_ready(compiled[implementation](*inputs))
                        mode_records[implementation]["samples"].append(time.perf_counter() - start)
                for record in mode_records.values():
                    record["steady_state_time"] = float(np.median(record["samples"][2:]))
                    print(json.dumps(record), flush=True)
                    with (output / "row-scatter-check.jsonl").open("a") as file:
                        file.write(json.dumps(record) + "\n")
                    records.append(record)
    tracker.log_summary({"kernel_check/row_scatter_pass": not failures, "kernel_check/records": len(records)})
    tracker.current_tracker().finish()
    if failures:
        raise AssertionError(f"Route-scatter oracle failures: {failures}")


if __name__ == "__main__":
    main()
