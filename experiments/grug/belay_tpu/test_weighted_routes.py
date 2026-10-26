# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check payload bits, accepted prefixes and complete expert/route pullbacks."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu import local_experts
from experiments.grug.belay_tpu.capacity_layout import interval_route_layout, weighted_interval_route_layout
from experiments.grug.belay_tpu.streaming_combine import combine_streaming, dispatch_streaming
from experiments.grug.belay_tpu.test_capacity_layout import oracle
from experiments.grug.belay_tpu.weighted_routes import _combine_preordered, weighted_local_expert_mlp


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("acceptance", ["all", "none", "partial"])
def test_payload_and_capacity_order(dtype, acceptance):
    rng = np.random.default_rng(446)
    flat = rng.integers(7, size=129, dtype=np.int32)
    flat[:80] = 0
    counts = np.bincount(flat, minlength=7).astype(np.int32)
    accepted = counts if acceptance == "all" else np.zeros_like(counts)
    if acceptance == "partial":
        accepted = rng.integers(counts + 1, dtype=np.int32)
    weights = jnp.asarray(rng.normal(size=flat.size), dtype).at[:4].set(jnp.asarray([0.0, -0.0, np.inf, np.nan], dtype))
    expected = oracle(flat, accepted)
    actual = jax.jit(weighted_interval_route_layout)(
        jnp.asarray(flat), jnp.asarray(counts), jnp.asarray(accepted), weights
    )
    for value, reference in zip(actual[:3], expected, strict=True):
        np.testing.assert_array_equal(value, reference)
    bits = np.uint16 if dtype == jnp.bfloat16 else np.uint32
    np.testing.assert_array_equal(np.asarray(actual[3]).view(bits), np.asarray(weights)[expected[0]].view(bits))


def test_target_reserve_payload_order():
    rng = np.random.default_rng(447)
    flat = rng.integers(384, size=327680, dtype=np.int32)
    counts = np.bincount(flat, minlength=384).astype(np.int32)
    accepted = rng.integers(counts + 1, dtype=np.int32)
    weights = jnp.asarray(rng.normal(size=flat.size), jnp.bfloat16)
    expected = oracle(flat, accepted)
    actual = jax.jit(weighted_interval_route_layout)(
        jnp.asarray(flat), jnp.asarray(counts), jnp.asarray(accepted), weights
    )
    for value, reference in zip(actual[:3], expected, strict=True):
        np.testing.assert_array_equal(value, reference)
    np.testing.assert_array_equal(
        np.asarray(actual[3]).view(np.uint16), np.asarray(weights)[expected[0]].view(np.uint16)
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_combine_pullback_against_fp64_derivative(dtype):
    rng = np.random.default_rng(448)
    tokens, topk, width = 129, 10, 256
    order = rng.permutation(tokens * topk).astype(np.int32)
    inverse = np.argsort(order).astype(np.int32)
    rows = jnp.asarray(rng.normal(size=(tokens * topk, width)), dtype).at[-129:].set(0)
    weights = jnp.asarray(rng.uniform(size=(tokens, topk)), dtype).at[:, 0].set(0)
    dy = jnp.asarray(rng.normal(size=(tokens, width)), dtype)

    def candidate(values, coefficients):
        cached = jax.lax.stop_gradient(coefficients.reshape(-1)[order])
        return _combine_preordered(values, jnp.asarray(order), jnp.asarray(inverse), coefficients, cached)

    value, backward = jax.vjp(candidate, rows, weights)
    drows, dweights = backward(dy)
    x, w, g = (np.asarray(v, np.float64) for v in (rows, weights, dy))
    expected = (
        (x[inverse].reshape(tokens, topk, width) * w[..., None]).sum(axis=1),
        g[order // topk] * w.reshape(-1)[order, None],
        (g[order // topk] * x).sum(axis=1)[inverse].reshape(tokens, topk),
    )
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for actual, reference in zip((value, drows, dweights), expected, strict=True):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(np.asarray(actual, np.float32), reference, atol=tolerance, rtol=tolerance)
    assert np.max(np.abs(np.asarray(dweights)[:, 0])) > 0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("acceptance", ["all", "none", "partial"])
def test_full_expert_pullback(monkeypatch, dtype, acceptance):
    rng = np.random.default_rng(449)
    tokens, topk, experts, width, hidden = 17, 3, 7, 16, 32
    ids = jnp.asarray(rng.integers(experts, size=(tokens, topk)), jnp.int32)
    counts = np.bincount(np.asarray(ids).ravel(), minlength=experts)
    accepted = counts if acceptance == "all" else np.zeros_like(counts)
    if acceptance == "partial":
        accepted = rng.integers(counts + 1)
    accepted = jnp.asarray(accepted, jnp.int32)
    shapes = ((tokens, width), (tokens, topk), (experts, width, hidden), (experts, hidden, width))
    values = tuple(jnp.asarray(rng.normal(size=shape) * 0.2, dtype) for shape in shapes)
    dy = jnp.asarray(rng.normal(size=(tokens, width)), dtype)
    monkeypatch.setattr(local_experts, "route_layout", interval_route_layout)
    monkeypatch.setattr(local_experts, "dispatch_local", dispatch_streaming)
    monkeypatch.setattr(local_experts, "combine_local", functools.partial(combine_streaming, implementation="scan"))

    def evaluate(implementation):
        def forward(x, weights, up, down):
            return implementation(x, ids, weights, up, down, accepted_counts=accepted)[0]

        output, backward = jax.vjp(forward, *values)
        return output, backward(dy)

    expected = evaluate(local_experts.local_expert_mlp)
    actual = evaluate(weighted_local_expert_mlp)
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.isfinite(value).all()
        np.testing.assert_allclose(
            np.asarray(value, np.float32), np.asarray(reference, np.float32), atol=tolerance, rtol=tolerance
        )
