# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Device-local MoE route reductions that write token order from expert order."""

from functools import partial

import jax
import jax.numpy as jnp

from experiments.grug.belay_tpu.routing_tpu import restore_assignment_scalars_sort


def sum_scattered_routes(rows, order, inverse, topk, weights=None, *, reduction="restore"):
    """Reduce local expert rows using a permutation or a repeated-token scatter.

    ``order`` is a permutation of token-slot indices, including rejected rows.
    The caller zeros rejected rows before this function. FP32 accumulation and
    the output dtype match the local-token MoE contract.
    """
    tokens, width = rows.shape[0] // topk, rows.shape[1]
    if reduction == "restore":
        restored = jnp.zeros_like(rows).at[order].set(rows, unique_indices=True)
        values = restored.reshape(tokens, topk, width).astype(jnp.float32)
        if weights is not None:
            values = values * weights.astype(jnp.float32)[..., None]
        result = values.sum(axis=1)
    elif reduction == "segment":
        values = rows.astype(jnp.float32)
        if weights is not None:
            row_weights = restore_assignment_scalars_sort(weights.reshape(-1), inverse)
            values = values * row_weights.astype(jnp.float32)[:, None]
        result = jax.ops.segment_sum(values, order // topk, num_segments=tokens)
    else:
        raise ValueError(f"Unknown route scatter reduction: {reduction}")
    return result.astype(rows.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def combine_scattered_routes(rows, order, inverse, weights, reduction="restore"):
    """Combine local rows while retaining the existing expert-order pullback."""
    return sum_scattered_routes(rows, order, inverse, weights.shape[1], weights, reduction=reduction)


def _combine_forward(rows, order, inverse, weights, reduction):
    output = combine_scattered_routes(rows, order, inverse, weights, reduction)
    return output, (rows, order, inverse, weights)


def _combine_backward(reduction, residuals, gradient):
    del reduction
    rows, order, inverse, weights = residuals
    row_weights = restore_assignment_scalars_sort(weights.reshape(-1), inverse)
    upstream = gradient[order // weights.shape[1]].astype(jnp.float32)
    d_rows = (upstream * row_weights[:, None]).astype(rows.dtype)
    d_weights = (upstream * rows.astype(jnp.float32)).sum(axis=-1)
    d_weights = restore_assignment_scalars_sort(d_weights, order)
    return d_rows, None, None, d_weights.reshape(weights.shape).astype(weights.dtype)


combine_scattered_routes.defvjp(_combine_forward, _combine_backward)
