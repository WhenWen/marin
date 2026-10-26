# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check target-shaped weighted routes before the first TPU optimizer update."""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np

from experiments.grug.belay_tpu.capacity_layout import weighted_interval_route_layout
from experiments.grug.belay_tpu.kernel_check import measure_errors
from experiments.grug.belay_tpu.streaming_combine import combine_streaming
from experiments.grug.belay_tpu.streaming_combine_check import routing_weight_gradient_reference
from experiments.grug.belay_tpu.weighted_routes import _combine_preordered


def evaluate_weighted_routes(flat, counts, accepted, rows, weights, gradient):
    order, inverse, keep, cached = weighted_interval_route_layout(flat, counts, accepted, weights)
    output, backward = jax.vjp(lambda x, w: _combine_preordered(x, order, inverse, w, cached), rows, weights)
    return (order, inverse, keep, cached), (output, *backward(gradient))


def check_weighted_routes():
    """Validate capacity order and combine derivatives on each host's first TPU."""
    if jax.default_backend() != "tpu":
        raise RuntimeError("Weighted route validation requires TPU")
    failures = []
    with jax.default_device(jax.local_devices()[0]):
        tokens, topk, width, experts = 32768, 10, 256, 384
        for case in ("balanced", "skewed"):
            rng = np.random.default_rng(450)
            flat = rng.integers(experts, size=tokens * topk, dtype=np.int32)
            if case == "skewed":
                flat[rng.random(flat.size) < 0.9] = 0
            counts = np.bincount(flat, minlength=experts).astype(np.int32)
            accepted = (counts * 0.93).astype(np.int32)
            accepted[::11] = 0
            keep = np.zeros(flat.shape, bool)
            for expert, count in enumerate(accepted):
                keep[np.flatnonzero(flat == expert)[:count]] = True
            order = np.argsort(np.where(keep, flat, experts), kind="stable").astype(np.int32)
            inverse = np.argsort(order).astype(np.int32)
            rows = jnp.asarray(rng.standard_normal((tokens * topk, width), dtype=np.float32), jnp.bfloat16)
            rows = rows.at[int(accepted.sum()) :].set(0)
            weights = jnp.asarray(rng.random((tokens, topk), dtype=np.float32), jnp.bfloat16).at[:, 0].set(0)
            gradient = jnp.asarray(rng.standard_normal((tokens, width), dtype=np.float32), jnp.bfloat16)
            layout, actual = jax.jit(evaluate_weighted_routes)(
                jnp.asarray(flat), jnp.asarray(counts), jnp.asarray(accepted), rows, weights, gradient
            )
            layout_pass = all(
                np.array_equal(got, expected) for got, expected in zip(layout[:3], (order, inverse, keep), strict=True)
            )
            payload_pass = np.array_equal(
                np.asarray(layout[3]).view(np.uint16), np.asarray(weights).reshape(-1)[order].view(np.uint16)
            )

            def reference(values, coefficients, *, order=order, inverse=inverse):
                return combine_streaming(
                    values, jnp.asarray(order), jnp.asarray(inverse), coefficients, implementation="scan"
                )

            output, backward = jax.vjp(reference, rows, weights)
            expected = (output, *backward(gradient))
            source_errors = measure_errors(actual, expected, 2e-2)
            oracle = routing_weight_gradient_reference(np.asarray(rows), order, inverse, np.asarray(gradient), topk=topk)
            oracle_errors = measure_errors((actual[-1],), (oracle,), 2e-2)
            # Weight-gradient cross-kernel differences are diagnostic. Its
            # accuracy gate uses the independent FP64 derivative at the same tolerance.
            passed = (
                layout_pass
                and payload_pass
                and all(
                    np.isfinite(item["normalized_max"]) and item["normalized_max"] <= 1
                    for item in (*source_errors[:2], *oracle_errors)
                )
            )
            record = dict(
                stage="weighted_routes_oracle",
                case=case,
                seed=450,
                rank=jax.process_index(),
                shape=[tokens, topk, width, experts],
                layout_exact=layout_pass,
                payload_exact=payload_pass,
                source_errors=source_errors,
                fp64_weight_gradient_errors=oracle_errors,
                atol=2e-2,
                rtol=2e-2,
                passed=passed,
                optimizer_updates=0,
            )
            print(json.dumps(record), flush=True)
            with (Path(os.environ["IRIS_OUTPUT_DIR"]) / "weighted-routes-check.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            if not passed:
                failures.append(case)
    tracker.log_summary({"preflight/weighted_routes_pass": not failures})
    if failures:
        raise AssertionError(f"Weighted route numerical failures: {failures}")
