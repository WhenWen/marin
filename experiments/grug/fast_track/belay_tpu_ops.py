# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared adapters for the experimentally qualified Belay TPU execution path."""

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.kernels.pallas.fused_cross_entropy_loss.config import BlockSizes

from experiments.grug.fast_track import model
from experiments.grug.fast_track.column_sharded_embedding import column_sharded_embedding
from experiments.grug.fast_track.model import _proj as reference_projection
from experiments.grug.moe.kda_state_tpu import state_pass


def gathered_projection(cfg, x, w, out_sharding=None):
    """Gather small projection weights while retaining their stored sharding."""
    return reference_projection(cfg, x, jax.sharding.reshard(w, P(None, None)), out_sharding)


def tpu_recurrence(prep, state, mm, lead, dk, dv, *, scan_impl, precision=jax.lax.Precision.HIGHEST):
    """Apply the state kernel to local chunks inside the model's KDA shard_map."""
    n, c = prep.kw.shape[-3:-1]

    def flatten(x):
        return x.reshape(-1, n, *x.shape[-2:])

    out, final = state_pass(
        flatten(prep.q_inflate),
        flatten(prep.attn),
        flatten(prep.kw),
        flatten(prep.k_cumdecay),
        flatten(prep.v_pseudo),
        prep.decay.reshape(-1, n, 1, dk),
        state.reshape(-1, dk, dv),
        precision=precision,
    )
    return out.reshape(*lead, n, c, dv), final.reshape(*lead, dk, dv)


def pallas_cross_entropy(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    *,
    implementation: str,
    block_sizes: BlockSizes,
    **kwargs: Any,
) -> jax.Array:
    if implementation != "xla_fast_bwd" or block_sizes != model._CE_BLOCK_SIZES or lm_head.shape != (512, 16384):
        raise ValueError("The Belay TPU profile requires the frozen base16 head")
    return fused_linear_softmax_cross_entropy_loss(
        hidden,
        lm_head,
        labels,
        implementation="pallas_tpu",
        block_sizes=BlockSizes(1024, 512, 512),
        **kwargs,
    )


def column_embedding_gather(table: jax.Array, ids: jax.Array) -> jax.Array:
    if table.shape != (524288, 512) or ids.ndim != 2 or ids.shape[1] != 4096 or table.dtype != jnp.bfloat16:
        raise ValueError("The Belay TPU profile requires the frozen base16 bigram table and sequence length")
    return column_sharded_embedding(table, ids)
