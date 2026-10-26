# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry combine weights with route metadata instead of gathering them in backward."""

import jax
import jax.numpy as jnp

from experiments.grug.belay_tpu import local_experts
from experiments.grug.belay_tpu.capacity_layout import weighted_interval_route_layout
from experiments.grug.belay_tpu.streaming_combine import sum_routes


@jax.custom_vjp
def _combine_preordered(rows, order, inverse, weights, row_weights):
    # row_weights is a cache produced by weighted_interval_route_layout. Like
    # order/inverse, its consistency is guaranteed by the caller below.
    del order, row_weights
    return sum_routes(rows, inverse.reshape(weights.shape), weights, implementation="scan")


def _combine_forward(rows, order, inverse, weights, row_weights):
    output = _combine_preordered(rows, order, inverse, weights, row_weights)
    return output, (rows, order, weights, row_weights)


def _combine_backward(residuals, gradient):
    rows, order, weights, row_weights = residuals
    upstream = gradient[order // weights.shape[1]].astype(jnp.float32)
    d_rows = (upstream * row_weights[:, None].astype(jnp.float32)).astype(rows.dtype)
    d_weights = (upstream * rows.astype(jnp.float32)).sum(axis=-1).astype(weights.dtype)
    # Rounding commutes with permutation. Convert before this integer-key sort
    # so its payload matches the source's BF16 scalar-gather output.
    d_weights = jax.lax.sort_key_val(order, d_weights)[1].reshape(weights.shape)
    return d_rows, None, None, d_weights, None


_combine_preordered.defvjp(_combine_forward, _combine_backward)


def weighted_local_expert_mlp(x, selected, combine, w_up, w_down, *, accepted_counts, expert_combine=None):
    """Use the same expert MLP and capacity order with weights carried by its sorts."""
    if expert_combine is not None:
        raise ValueError("Weighted route layout has not been composed with the joint expert VJP")
    tokens, topk = selected.shape
    flat = selected.reshape(-1)
    counts = local_experts.route_counts(flat, w_up.shape[0])
    order, inverse, keep, row_weights = weighted_interval_route_layout(flat, counts, accepted_counts, combine)
    active_rows = accepted_counts.sum()
    active = jnp.arange(flat.size) < active_rows
    physical_counts = accepted_counts.at[-1].add(flat.size - active_rows)
    rows = jnp.where(active[:, None], local_experts.dispatch_local(x, order, inverse, topk), 0)
    outputs = local_experts.expert_mlp(rows, w_up, w_down, physical_counts, active_rows)
    outputs = jnp.where(active[:, None], outputs, 0)
    result = _combine_preordered(outputs, order, inverse, combine, row_weights)
    return result, keep.reshape(tokens, topk)
