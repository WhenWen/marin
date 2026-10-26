# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent route-wise algebra for the joint six-product backward."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.joint_expert_combine import make_joint_expert_combine


def reference(rows, up, down, weights, groups, active, order, inverse, gradient):
    x, u, d, w, g = (np.asarray(value, np.float64) for value in (rows, up, down, weights, gradient))
    outputs, dx = np.zeros_like(x), np.zeros_like(x)
    dup, ddown, dweights = np.zeros_like(u), np.zeros_like(d), np.zeros(x.shape[0])
    bounds = np.cumsum([0, *groups])
    for expert in range(len(groups)):
        lo, hi = bounds[expert], min(bounds[expert + 1], active)
        if hi <= lo:
            continue
        index = order[lo:hi]
        pre = x[lo:hi] @ u[expert]
        positive = np.maximum(pre, 0)
        hidden = positive**2
        output = hidden @ d[expert]
        outputs[lo:hi] = output
        upstream = g[index // w.shape[1]]
        weighted = upstream * w.ravel()[index, None]
        dpre = (weighted @ d[expert].T) * (2 * positive)
        dx[lo:hi] = dpre @ u[expert].T
        dup[expert] = x[lo:hi].T @ dpre
        ddown[expert] = hidden.T @ weighted
        # The oracle uses the forward expert output, independently of the proposed transpose identity.
        dweights[lo:hi] = (output * upstream).sum(axis=1)
    value = (outputs[inverse].reshape(*w.shape, x.shape[1]) * w[..., None]).sum(axis=1)
    return value, dx, dup, ddown, dweights[inverse].reshape(w.shape)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("output_dtype", [None, jnp.float32])
@pytest.mark.parametrize(
    "groups,active",
    [((61, 0, 273, 178), 512), ((31, 0, 143, 81), 255), ((31, 0, 143, 338), 255), ((0, 0, 0, 0), 0)],
)
def test_joint_backward_matches_route_output_oracle(dtype, output_dtype, groups, active):
    rng = np.random.default_rng(441)
    count, width, hidden, topk = 512, 128, 256, 4
    rows = jnp.asarray(rng.normal(scale=0.2, size=(count, width)), dtype).at[active:].set(jnp.nan)
    up = jnp.asarray(rng.normal(scale=0.1, size=(4, width, hidden)), dtype)
    down = jnp.asarray(rng.normal(scale=0.1, size=(4, hidden, width)), dtype)
    weights = jnp.asarray(rng.uniform(0.1, 0.9, size=(count // topk, topk)), dtype).at[:, 0].set(0)
    order = rng.permutation(count).astype(np.int32)
    inverse = np.argsort(order).astype(np.int32)
    gradient = jnp.asarray(rng.normal(scale=0.2, size=(count // topk, width)), output_dtype or dtype)
    combined = make_joint_expert_combine(interpret=True, output_dtype=output_dtype)

    def candidate(x, u, d, w):
        return combined(x, u, d, jnp.asarray(groups, jnp.int32), active, jnp.asarray(order), jnp.asarray(inverse), w)

    with jax.default_matmul_precision("highest"):
        output, backward = jax.vjp(candidate, rows, up, down, weights)
        actual = (output, *backward(gradient))
    expected = reference(rows, up, down, weights, groups, active, order, inverse, gradient)
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for value, target in zip(actual, expected, strict=True):
        assert np.isfinite(value).all()
        np.testing.assert_allclose(np.asarray(value, np.float32), target, atol=tolerance, rtol=tolerance)
    if active:
        # A zero route coefficient must still receive the derivative of its expert output.
        assert np.max(np.abs(np.asarray(actual[-1])[:, 0])) > 0
    else:
        assert all(np.count_nonzero(np.asarray(value)) == 0 for value in actual)
