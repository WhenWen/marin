# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check route-scatter values and pullbacks against FP64 token-slot arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.row_scatter import combine_scattered_routes, sum_scattered_routes


@pytest.mark.parametrize("reduction", ["restore", "segment"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("case", ["balanced", "skewed", "empty"])
def test_scatter_reductions_match_token_slot_arithmetic(reduction, dtype, case):
    rng = np.random.default_rng(444)
    tokens, topk, width = 33, 3, 16
    labels = rng.integers(7, size=tokens * topk)
    if case == "skewed":
        labels[:80] = 0
    order = np.argsort(labels, kind="stable").astype(np.int32)
    inverse = np.argsort(order).astype(np.int32)
    rows = jnp.asarray(rng.normal(size=(tokens * topk, width)), dtype)
    rows = rows.at[0 if case == "empty" else 83 :].set(0)
    weights = jnp.asarray(rng.random((tokens, topk)), dtype).at[:, 0].set(0)
    gradient = jnp.asarray(rng.normal(size=(tokens, width)), dtype)
    values = np.asarray(rows, np.float64)
    coefficients = np.asarray(weights, np.float64)
    upstream = np.asarray(gradient, np.float64)
    slots = values[inverse].reshape(tokens, topk, width)
    expected = (
        np.sum(slots * coefficients[..., None], axis=1),
        upstream[order // topk] * coefficients.reshape(-1)[order, None],
        np.sum(slots * upstream[:, None, :], axis=-1),
    )
    function = jax.jit(lambda x, w: combine_scattered_routes(x, order, inverse, w, reduction))
    actual, pullback = jax.vjp(function, rows, weights)
    observed = (actual, *pullback(gradient))
    tolerance = 1e-5 if dtype == jnp.float32 else 2e-2
    for got, want in zip(observed, expected, strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(np.asarray(got, np.float64), want, atol=tolerance, rtol=tolerance)
    unweighted = jax.jit(lambda x: sum_scattered_routes(x, order, inverse, topk, reduction=reduction))(rows)
    np.testing.assert_allclose(np.asarray(unweighted, np.float64), slots.sum(axis=1), atol=tolerance, rtol=tolerance)
