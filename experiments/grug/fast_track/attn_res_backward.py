# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Saved layer pullbacks with explicit history-gradient accumulation."""

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp


class AttnResLayerBackward(StrEnum):
    """How a Block AttnRes layer's backward obtains its forward intermediates."""

    RECOMPUTE = "recompute"
    """Custom VJP that saves only the layer inputs and re-runs the layer forward in backward (lowest
    memory; costs a second forward of every layer)."""
    SAVE = "save"
    """Plain autodiff: the forward keeps the layer's residuals, so backward runs no forward again.
    Same math as RECOMPUTE; needs the memory to hold every layer's residuals."""
    SAVE_THREADED = "save_threaded"
    """Diagnostic saved pullback with RECOMPUTE's explicit history-cotangent accumulation."""


@eqx.filter_custom_vjp
def saved_attn_res_layer(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key, *, layer_fn, differentiable_stats):
    """Save a layer-local pullback while threading history cotangents through each layer."""
    _, blocks, block_logits, _, _, _ = diff_args
    partial, stats = layer_fn(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key)
    return partial, blocks, block_logits, stats


@saved_attn_res_layer.def_fwd
def saved_attn_res_layer_fwd(
    perturbed, diff_args, mask, token_ids, use_long, layer_index, eps, noise_key, *, layer_fn, differentiable_stats
):
    del perturbed
    _, blocks, block_logits, _, _, _ = diff_args
    partial, vjp_fn, stats = jax.vjp(
        lambda args: layer_fn(args, mask, token_ids, use_long, layer_index, eps, noise_key),
        diff_args,
        has_aux=True,
    )
    if any(key in stats for key in differentiable_stats):
        raise ValueError("SAVE_THREADED is a diagnostic for layers without differentiable statistics")
    return (partial, blocks, block_logits, stats), vjp_fn


@saved_attn_res_layer.def_bwd
def saved_attn_res_layer_bwd(
    vjp_fn,
    grad_out,
    perturbed,
    diff_args,
    mask,
    token_ids,
    use_long,
    layer_index,
    eps,
    noise_key,
    *,
    layer_fn,
    differentiable_stats,
):
    del perturbed, mask, token_ids, use_long, eps, noise_key, layer_fn, differentiable_stats
    d_partial, d_blocks, d_block_logits, _ = grad_out
    _, blocks, block_logits, _, _, _ = diff_args
    d_blocks = jax.tree.map(lambda d, x: jnp.zeros_like(x) if d is None else d, d_blocks, blocks, is_leaf=_is_none)
    d_block_logits = jax.tree.map(
        lambda d, x: jnp.zeros_like(x) if d is None else d, d_block_logits, block_logits, is_leaf=_is_none
    )
    with jax.named_scope(f"attn_res_bwd{layer_index}"):
        vjp_fn, d_partial, d_blocks, d_block_logits = jax.lax.optimization_barrier(
            (vjp_fn, d_partial, d_blocks, d_block_logits)
        )
        ((d_layer, d_blocks_own, d_block_logits_own, d_partial_in, d_queries, d_logit_bias),) = vjp_fn(d_partial)
        d_blocks = tuple(a + b for a, b in zip(d_blocks, d_blocks_own, strict=True))
        d_block_logits = jax.tree.map(jnp.add, d_block_logits, d_block_logits_own)
        return jax.lax.optimization_barrier((d_layer, d_blocks, d_block_logits, d_partial_in, d_queries, d_logit_bias))


def _is_none(x) -> bool:
    return x is None
