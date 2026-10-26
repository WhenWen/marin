# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare all-gradient fusion with selected row GEMMs and the prior fusion."""

import functools
import json
import os
import time
from itertools import pairwise

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.expert_gradient_adapter import partitioned_grouped_dot
from experiments.grug.belay_tpu.fused_expert_backward import fused_expert_relu2
from experiments.grug.belay_tpu.fused_row_relu2 import fused_row_relu2
from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.operator_check import initialize_operator_check


def selected(x, up, down, groups):
    pre = partitioned_grouped_dot(x, up, groups, row_partitions=2)
    post = jnp.square(jax.nn.relu(pre))
    return partitioned_grouped_dot(post, down, groups, row_partitions=2)


def evaluate(function, mode, x, up, down, groups, gradient):
    if mode == "forward":
        return (function(x, up, down, groups),)

    def operation(x, up, down):
        return function(x, up, down, groups)

    output, backward = jax.vjp(operation, x, up, down)
    return output, *backward(gradient)


def independent_slices(counts, x, up, down, groups):
    del groups
    values = []
    for expert, (lo, hi) in enumerate(pairwise(np.cumsum([0, *counts]))):
        pre = (x[lo:hi].astype(jnp.float32) @ up[expert].astype(jnp.float32)).astype(x.dtype)
        hidden = jnp.square(jax.nn.relu(pre))
        values.append((hidden.astype(jnp.float32) @ down[expert].astype(jnp.float32)).astype(x.dtype))
    return jnp.concatenate(values)


def main():
    output, operator_topology, reference_topology = initialize_operator_check()
    tracker.log_summary({"kernel_check/input_seed": 445})
    cases = (
        ("boundary", 1024, 8, 128, 256),
        ("medium", 32768, 384, 256, 384),
        ("balanced", 327680, 384, 256, 384),
        ("skewed", 327680, 384, 256, 384),
    )
    failures, records = [], []

    def record(row):
        print(json.dumps(row), flush=True)
        with (output / "fused-expert-backward-check.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")

    with jax.default_device(jax.local_devices()[0]):
        for case, rows, experts, width, hidden in cases:
            counts = np.full(experts, rows // experts, np.int32)
            counts[: rows % experts] += 1
            if case == "boundary":
                counts = np.asarray([37, 0, 487, 0, 101, 221, 0, 178], np.int32)
            elif case == "skewed":
                counts[0] = rows * 9 // 10
                remainder = rows - counts[0]
                counts[1:] = remainder // (experts - 1)
                counts[1 : 1 + remainder % (experts - 1)] += 1
            assert counts.sum() == rows
            rng = np.random.default_rng(445)
            x = jnp.asarray(rng.standard_normal((rows, width), dtype=np.float32) * 0.1, jnp.bfloat16)
            up = jnp.asarray(rng.standard_normal((experts, width, hidden), dtype=np.float32) * 0.05, jnp.bfloat16)
            down = jnp.asarray(rng.standard_normal((experts, hidden, width), dtype=np.float32) * 0.05, jnp.bfloat16)
            gradient = jnp.asarray(rng.standard_normal(x.shape, dtype=np.float32) * 0.1, jnp.bfloat16)
            rejected = min(1024, rows // 4)
            x, gradient = x.at[-rejected:].set(0), gradient.at[-rejected:].set(0)
            inputs = (x, up, down, jnp.asarray(counts), gradient)
            for mode in ("forward", "forward_vjp"):
                expected = None
                oracle = "selected_row_partitioned_gmm"
                if case == "boundary":
                    reference = functools.partial(independent_slices, tuple(counts))
                    with jax.default_matmul_precision("highest"):
                        expected = jax.jit(functools.partial(evaluate, reference, mode))(*inputs)
                    oracle = "independent_expert_slices"
                executables, mode_records = {}, {}
                implementations = (
                    ("selected", selected),
                    ("fused_v1", fused_row_relu2),
                    ("fused_all", fused_expert_relu2),
                )
                for name, function in implementations:
                    start = time.perf_counter()
                    compiled = jax.jit(functools.partial(evaluate, function, mode)).lower(*inputs).compile()
                    compile_time = time.perf_counter() - start
                    start = time.perf_counter()
                    actual = jax.block_until_ready(compiled(*inputs))
                    first_execution = time.perf_counter() - start
                    if expected is None:
                        expected = actual
                    error = measure_errors(actual, expected, 2e-2)
                    passed = all(np.isfinite(v["normalized_max"]) and v["normalized_max"] <= 1 for v in error)
                    if not passed:
                        failures.append((case, mode, name))
                    row = {
                        "stage": "fused_expert_backward_compiled",
                        "kernel": mode,
                        "implementation": name,
                        "case": case,
                        "shape": [rows, experts, width, hidden],
                        "dtype": str(x.dtype),
                        "rank": jax.process_index(),
                        "backend": jax.default_backend(),
                        "device_type": jax.local_devices()[0].device_kind,
                        "device_count": 1,
                        "operator_topology": operator_topology,
                        "reference_topology": reference_topology,
                        "block_sizes": {"tile_rows": 512, "row_partitions": 2},
                        "oracle": oracle,
                        "error": error,
                        "passed": passed,
                        "compile_time": compile_time,
                        "first_execution_time": first_execution,
                        "temporary_bytes": compiled.memory_analysis().temp_size_in_bytes,
                        "memory": str(compiled.memory_analysis()),
                        "samples": [],
                        "git_sha": os.environ["BELAY_SOURCE_COMMIT"],
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                    }
                    record(row)
                    executables[name], mode_records[name] = compiled, row
                for iteration in range(12):
                    names = tuple(name for name, _ in implementations)
                    rotation = iteration % len(names)
                    sequence = names[rotation:] + names[:rotation]
                    if iteration % 2:
                        sequence = sequence[::-1]
                    for name in sequence:
                        start = time.perf_counter()
                        jax.block_until_ready(executables[name](*inputs))
                        mode_records[name]["samples"].append(time.perf_counter() - start)
                for row in mode_records.values():
                    row["stage"] = "fused_expert_backward_comparison"
                    row["steady_state_time"] = float(np.median(row["samples"][2:]))
                    record(row)
                    records.append(row)
    tracker.log_summary({"kernel_check/fused_expert_backward_pass": not failures, "kernel_check/records": len(records)})
    tracker.current_tracker().finish()
    if failures:
        raise AssertionError(f"Fused expert oracle failures: {failures}")


if __name__ == "__main__":
    main()
