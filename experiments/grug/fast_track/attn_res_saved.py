# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Order saved AttnRes pullbacks without repeating the forward computation."""

import equinox as eqx
import jax


@eqx.filter_custom_vjp
def saved_layer(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key, *, layer_forward, statistic_keys):
    """Save a layer's pullback and eagerly accumulate its history cotangents."""
    partial, stats = layer_forward(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key)
    return partial, diff_args[1], diff_args[2], stats


@saved_layer.def_fwd
def _forward(perturbed, diff_args, mask, token_ids, use_long, layer_index, eps, noise_key, *, layer_forward, statistic_keys):
    del perturbed

    def forward(args):
        partial, stats = layer_forward(args, mask, token_ids, use_long, layer_index, eps, noise_key)
        differentiable = {k: stats[k] for k in statistic_keys if k in stats}
        return (partial, differentiable), stats

    (partial, differentiable), pullback, stats = eqx.filter_vjp(forward, diff_args, has_aux=True)
    return (partial, diff_args[1], diff_args[2], stats), (pullback, tuple(differentiable))


@saved_layer.def_bwd
def _backward(residuals, grad_out, perturbed, diff_args, mask, token_ids, use_long, layer_index, eps, noise_key,
              *, layer_forward, statistic_keys):
    del perturbed, diff_args, mask, token_ids, use_long, eps, noise_key, layer_forward, statistic_keys
    pullback, keys = residuals
    with jax.named_scope(f"attn_res_saved_bwd{layer_index}"):
        d_partial, d_blocks, d_logits, d_stats = jax.lax.optimization_barrier(grad_out)
        ((d_layer, own_blocks, own_logits, d_partial_in, d_queries, d_extras),) = pullback(
            (d_partial, {k: d_stats[k] for k in keys})
        )
        add = lambda a, b: b if a is None else a if b is None else a + b  # noqa: E731
        is_none = lambda x: x is None  # noqa: E731
        d_blocks = jax.tree.map(add, d_blocks, own_blocks, is_leaf=is_none)
        d_logits = jax.tree.map(add, d_logits, own_logits, is_leaf=is_none)
        return jax.lax.optimization_barrier((d_layer, d_blocks, d_logits, d_partial_in, d_queries, d_extras))
