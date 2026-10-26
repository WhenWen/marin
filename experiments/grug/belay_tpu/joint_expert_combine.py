# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the down-projection pullback for route gradients without saved outputs."""

from functools import partial

import jax
import jax.numpy as jnp

from experiments.grug.belay_tpu.expert_gradient_adapter import _grouped_dot, whole_width_tiling
from experiments.grug.belay_tpu.expert_gradient_tpu import expert_weight_gradient
from experiments.grug.belay_tpu.streaming_combine import sum_routes


def make_joint_expert_combine(*, interpret=False, output_dtype=None):
    """Build a six-GEMM ReLU² MLP/combine VJP retaining only input and preactivation.

    Route gradients use h dot (g W_down.T), where g is the unweighted token
    cotangent. This remains defined at zero route weights. Moving route weights
    across the product changes floating-point rounding, but not the real-valued
    derivative. The unused route suffix is masked before and after each product.
    """
    grouped_dot = partial(_grouped_dot, interpret=interpret, tiling=whole_width_tiling, row_partitions=2)
    weight_gradient = partial(expert_weight_gradient, partitions=2, tile_rows=512, interpret=interpret)

    def forward(rows, up, down, groups, active_rows, order, inverse, weights):
        active = jnp.arange(rows.shape[0])[:, None] < active_rows
        rows = jnp.where(active, rows, 0)
        pre = jnp.where(active, grouped_dot(rows, up.astype(rows.dtype), groups), 0)
        hidden = jnp.square(jax.nn.relu(pre))
        outputs = jnp.where(active, grouped_dot(hidden, down.astype(rows.dtype), groups), 0)
        result = sum_routes(
            outputs, inverse.reshape(weights.shape), weights, implementation="scan", output_dtype=output_dtype
        )
        return result, (rows, up, down, groups, active_rows, order, inverse, weights, pre)

    @jax.custom_vjp
    def combined(rows, up, down, groups, active_rows, order, inverse, weights):
        return forward(rows, up, down, groups, active_rows, order, inverse, weights)[0]

    def backward(residuals, gradient):
        rows, up, down, groups, active_rows, order, inverse, weights, pre = residuals
        active = jnp.arange(rows.shape[0])[:, None] < active_rows
        unweighted = jnp.where(active, gradient.astype(rows.dtype)[order // weights.shape[1]], 0)
        # Reconstruct hidden values after the cotangent arrives, without reusing
        # the forward hidden tensor through common-subexpression elimination.
        pre, unweighted = jax.lax.optimization_barrier((pre, unweighted))
        projected = jnp.where(active, grouped_dot(unweighted, down.astype(rows.dtype), groups, transpose=True), 0)
        row_weights = weights.reshape(-1)[order].astype(jnp.float32)
        positive = jax.nn.relu(pre)
        hidden = jnp.square(positive)
        positive_float, projected_float = positive.astype(jnp.float32), projected.astype(jnp.float32)
        dpre = (2 * positive_float * projected_float * row_weights[:, None]).astype(rows.dtype)
        weighted = (unweighted.astype(jnp.float32) * row_weights[:, None]).astype(rows.dtype)
        ddown = weight_gradient(hidden, weighted, groups, output_dtype=down.dtype)
        dup = weight_gradient(rows, dpre, groups, output_dtype=up.dtype)
        dx = jnp.where(active, grouped_dot(dpre, up.astype(rows.dtype), groups, transpose=True), 0)
        dweights = (hidden.astype(jnp.float32) * projected_float).sum(axis=-1)
        dweights = dweights[inverse].reshape(weights.shape).astype(weights.dtype)
        return dx, dup, ddown, None, None, None, None, dweights

    combined.defvjp(forward, backward)
    return combined
