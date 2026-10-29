# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution-only ports from Belay's qualified TPU implementation.

Source: belay-orca at fcad3cbe6, experiments/grug/{moe,fast_track}.
The model, chunk preparation, routing rule and optimizer remain the historical KMA.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import jax
import jax.numpy as jnp
from levanter.grug.attention import _core as attention_core

from experiments.grug.fast_track import model
from experiments.grug.moe import kda
from experiments.grug.muon_mup_tpu.inkling_tpu import sliced_dense_rel_bias
from experiments.grug.muon_mup_tpu.kda_state_tpu import state_pass

REFERENCE_DENSE_BIAS = attention_core.dense_rel_bias
REFERENCE_QUANTILE = model._bincount_upper_quantile
REFERENCE_RECURRENCE = kda._parallel_chunk_recurrence


def factorized_expert_histogram(indices: jax.Array, n_bins: int, group_size: int, radix: int) -> jax.Array:
    """Count clipped ``[tokens, experts]`` indices with binary matrix products.

    Each expert's binary one-hot matrices contract over tokens independently.
    Accumulation is int32 and exact while the token count fits; groups limit
    live memory. TPU XLA can fold the source int8 one-hots into predicates.
    """
    tokens, experts = indices.shape
    groups = indices.T.reshape(experts // group_size, group_size, tokens)
    high_bins = (n_bins + radix - 1) // radix

    def count_group(group):
        high = jax.nn.one_hot(group // radix, high_bins, dtype=jnp.int8)
        low = jax.nn.one_hot(group % radix, radix, dtype=jnp.int8)
        counts = jax.lax.dot_general(high, low, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.int32)
        return counts.reshape(group_size, high_bins * radix)[:, :n_bins]

    return jax.lax.map(count_group, groups).reshape(experts, n_bins)


def histogram_crossing(counts, rank, last_valid):
    """Return the crossing bin, suffix count and bin height for each expert."""
    cumulative = jnp.cumsum(counts[:, ::-1], axis=-1)[:, ::-1]
    crossing = jnp.clip(jnp.sum(cumulative >= rank[:, None], axis=-1) - 1, 0, last_valid)
    ct = jnp.take_along_axis(cumulative, crossing[:, None], axis=-1)[:, 0]
    height = jnp.take_along_axis(counts, crossing[:, None], axis=-1)[:, 0]
    return crossing, ct, height


def two_stage_histogram_crossing(indices, rank, *, batch_axes, n_bins, group_size):
    """Count coarse bins globally, then count only each expert's crossing block.

    Integer collectives preserve exact counts. A second histogram excludes
    unselected blocks with negative indices, whose high-digit one-hot is zero.
    """
    block_bins = 64
    coarse_bins = (n_bins + block_bins - 1) // block_bins
    coarse_indices = indices // block_bins
    coarse_counts = jax.lax.psum(
        factorized_expert_histogram(coarse_indices, coarse_bins, group_size, 16), batch_axes
    ).astype(jnp.float32)
    block, coarse_ct, coarse_height = histogram_crossing(coarse_counts, rank, coarse_bins - 1)
    above = coarse_ct - coarse_height
    selected = jnp.where(coarse_indices == block[None, :], indices % block_bins, -1)
    fine_counts = jax.lax.psum(factorized_expert_histogram(selected, block_bins, group_size, 8), batch_axes).astype(
        jnp.float32
    )
    cumulative = jnp.cumsum(fine_counts[:, ::-1], axis=-1)[:, ::-1] + above[:, None]
    last_valid = jnp.minimum(block_bins - 1, n_bins - 1 - block * block_bins)
    local_bin = jnp.clip(jnp.sum(cumulative >= rank[:, None], axis=-1) - 1, 0, last_valid)
    ct = jnp.take_along_axis(cumulative, local_bin[:, None], axis=-1)[:, 0]
    height = jnp.take_along_axis(fine_counts, local_bin[:, None], axis=-1)[:, 0]
    return block * block_bins + local_bin, ct, height


def two_stage_quantile(s_local, *, num_experts, n_bins, lo, hi, target_rank):
    # Explicitly qualify KMA's 384-expert shape; no silent 512-expert-only fallback.
    if num_experts != 384 or n_bins != 10000:
        raise ValueError("This execution adapter is qualified only for KMA's 384 experts / 10000 bins")
    bin_width = (hi - lo) / n_bins
    indices = jnp.clip(((s_local - lo) / bin_width).astype(jnp.int32), 0, n_bins - 1)
    rank = jnp.broadcast_to(jnp.asarray(target_rank, jnp.float32), (num_experts,))
    crossing, ct, height = two_stage_histogram_crossing(
        indices, rank, batch_axes=model._BATCH_AXES, n_bins=n_bins, group_size=64
    )
    lower = lo + crossing.astype(jnp.float32) * bin_width
    return lower + bin_width * (ct - rank) / jnp.maximum(height, 1.0)


def tpu_recurrence(prep, state, mm, lead, dk, dv):
    chunks, chunk_size = prep.kw.shape[-3:-1]

    def flatten(x):
        return x.reshape(-1, chunks, *x.shape[-2:])

    output, final = state_pass(
        flatten(prep.q_inflate),
        flatten(prep.attn),
        flatten(prep.kw),
        flatten(prep.k_cumdecay),
        flatten(prep.v_pseudo),
        prep.decay.reshape(-1, chunks, 1, dk),
        state.reshape(-1, dk, dv),
        precision=jax.lax.Precision.HIGH,
    )
    return output.reshape(*lead, chunks, chunk_size, dv), final.reshape(*lead, dk, dv)


@contextmanager
def execution_mode(mode):
    """Select explicit execution patches before tracing, without changing configuration."""
    if mode not in ("reference", "histogram", "tpu", "inkling"):
        raise ValueError(f"Unknown execution mode: {mode}")
    with ExitStack() as stack:
        quantile = two_stage_quantile if mode in ("histogram", "tpu") else REFERENCE_QUANTILE
        recurrence = tpu_recurrence if mode == "tpu" else REFERENCE_RECURRENCE
        dense_bias = sliced_dense_rel_bias if mode == "inkling" else REFERENCE_DENSE_BIAS
        stack.enter_context(patch.object(attention_core, "dense_rel_bias", dense_bias))
        stack.enter_context(patch.object(model, "_bincount_upper_quantile", quantile))
        stack.enter_context(patch.object(kda, "_parallel_chunk_recurrence", recurrence))
        yield
