# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Causal-prefix Inkling attention candidate for the frozen TPU operator probe."""

from functools import partial

import jax
import jax.numpy as jnp
from levanter.grug.attention._core import _reference_attention_math
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band


def prefix_block(q, k, v, band, segments, start, *, prefix, logits_dtype):
    queries = jax.lax.dynamic_slice_in_dim(q, start, REL_BIAS_BLOCK, axis=1)
    local_band = jax.lax.dynamic_slice_in_dim(band, start, REL_BIAS_BLOCK, axis=2)
    padded = jnp.pad(local_band, ((0, 0), (0, 0), (0, 0), (prefix, prefix)))
    bias = jax.lax.dynamic_slice_in_dim(padded, prefix + rel_extent_of_band(band) - start, prefix, axis=-1)
    query_segments = jax.lax.dynamic_slice_in_dim(segments, start, REL_BIAS_BLOCK, axis=1)
    causal = jnp.arange(prefix)[None, :] <= (start + jnp.arange(REL_BIAS_BLOCK))[:, None]
    mask = causal[None] & (query_segments[:, :, None] == segments[:, None, :prefix])
    return _reference_attention_math(queries, k[:, :prefix], v[:, :prefix], mask, bias, logits_dtype=logits_dtype)


def causal_prefix_attention(q, k, v, band, segments, *, quantum=1024, logits_dtype=jnp.float32):
    batch, length, heads, _ = q.shape
    if quantum % REL_BIAS_BLOCK or length % REL_BIAS_BLOCK:
        raise ValueError("Query length and prefix quantum must be multiples of 128")
    if k.shape[1] != length or v.shape[1] != length or segments.shape != (batch, length):
        raise ValueError("This probe requires causal self-attention and per-batch segment IDs")
    prefixes = tuple(min(i, length) for i in range(quantum, length + quantum, quantum))
    branches = tuple(partial(prefix_block, prefix=prefix, logits_dtype=logits_dtype) for prefix in prefixes)

    @jax.checkpoint
    def block(index):
        start = index * REL_BIAS_BLOCK
        return jax.lax.switch(start // quantum, branches, q, k, v, band, segments, start)

    blocks = jax.lax.map(block, jnp.arange(length // REL_BIAS_BLOCK))
    return blocks.transpose(1, 0, 2, 3, 4).reshape(batch, length, heads, v.shape[-1])
