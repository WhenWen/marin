# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental mask-supported key cropping for aligned causal attention."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention._core import AttentionMask, _reference_attention_math
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band

from experiments.grug.fast_track.benchmark_inkling_flash_tpu import fixture

PROTOCOL = "https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6176382867"
SCENARIOS = ("one-document", "packed", "strong-logits", "reused", "mixed-batch")


def key_widths(length):
    return tuple(sorted({length, *(width for width in (128, 256, 512, 1024, 2048, 4096) if width <= length)}))


def crop_selection(tile_mask, query_start):
    """Select the smallest static width containing every allowed key."""
    length = tile_mask.shape[-1]
    widths = jnp.asarray(key_widths(length), jnp.int32)
    first = jnp.min(jnp.where(tile_mask, jnp.arange(length), length))
    empty_row = jnp.any(~jnp.any(tile_mask, axis=-1))
    required = jnp.where(empty_row, length, query_start + REL_BIAS_BLOCK - first)
    branch = jnp.sum(widths < required, dtype=jnp.int32)
    width = widths[branch]
    start = jnp.clip(query_start + REL_BIAS_BLOCK - width, 0, length - width)
    return branch, start, width


def mask_crop_attention(q, k, v, mask, rel_bias, *, block_size=REL_BIAS_BLOCK):
    """Crop masked keys without changing causal or segment visibility."""
    batch, length, heads, _ = q.shape
    if not isinstance(mask, AttentionMask) or not mask.is_causal:
        raise ValueError("Key cropping requires structured causal attention")
    if block_size != REL_BIAS_BLOCK or length % block_size or length != k.shape[1]:
        raise ValueError("Key cropping requires aligned 128-query self-attention blocks")

    def cropped(width, operands):
        query, band, tile_mask, query_start, key_start = operands
        keys = jax.lax.dynamic_slice_in_dim(k, key_start, width, axis=1)
        values = jax.lax.dynamic_slice_in_dim(v, key_start, width, axis=1)
        allowed = jax.lax.dynamic_slice_in_dim(tile_mask, key_start, width, axis=-1)
        padded = jnp.pad(band, ((0, 0), (0, 0), (0, 0), (width, width)))
        bias = jax.lax.dynamic_slice_in_dim(
            padded, width + rel_extent_of_band(band) - query_start + key_start, width, axis=-1
        )
        return _reference_attention_math(query, keys, values, allowed, bias, logits_dtype=jnp.float32)

    branches = tuple(partial(cropped, width) for width in key_widths(length))

    @jax.checkpoint
    def block(index):
        query_start = index * block_size
        query = jax.lax.dynamic_slice_in_dim(q, query_start, block_size, axis=1)
        band = jax.lax.dynamic_slice_in_dim(rel_bias, query_start, block_size, axis=2)
        tile_mask = mask.materialize_mask(block_size, length, query_offset=query_start)
        branch, key_start, _ = crop_selection(tile_mask, query_start)
        return jax.lax.switch(branch, branches, (query, band, tile_mask, query_start, key_start))

    output = jax.lax.map(block, jnp.arange(length // block_size))
    return output.transpose(1, 0, 2, 3, 4).reshape(batch, length, heads, v.shape[-1])


def crop_fixture(seed, length, scenario):
    """Extend the existing numerical fixtures with reused IDs and mixed rows."""
    if scenario not in ("reused", "mixed-batch"):
        return fixture(seed, length, scenario)
    first = list(fixture(seed, length, "one-document"))
    if scenario == "reused":
        first[4] = (np.arange(length, dtype=np.int32)[None] // 37) % 3
        return tuple(first)
    second = list(fixture(seed + 1000, length, "one-document"))
    for row, boundaries in ((first, (37, 113)), (second, (193, 773))):
        for boundary in boundaries:
            row[4][:, boundary:] += 1
    return tuple(np.concatenate((a, b), axis=0) for a, b in zip(first, second, strict=True))


def width_distribution(segments, window):
    """Record the actual selector's widths outside the timing interval."""
    length = segments.shape[-1]
    mask = AttentionMask.causal(sliding_window=window).with_segment_ids(segments)

    def selection(index):
        start = index * REL_BIAS_BLOCK
        return crop_selection(mask.materialize_mask(REL_BIAS_BLOCK, length, query_offset=start), start)[2]

    values = np.asarray(jax.jit(lambda: jax.lax.map(selection, jnp.arange(length // REL_BIAS_BLOCK)))())
    return {str(width): int(np.sum(values == width)) for width in key_widths(length)}


def support_check():
    """Check retained support against independently enumerated NumPy masks."""
    results = []
    selector = jax.jit(crop_selection)
    for length in (256, 1024, 2048, 4096):
        positions = np.arange(length)
        for window in (None, 129, 2048):
            for scenario in ("contiguous", "reused", "mixed", "empty"):
                segments = np.stack((positions // 257, positions // 394)).astype(np.int32)
                if scenario == "contiguous":
                    segments = np.broadcast_to(positions // 1024, (2, length))
                if scenario == "reused":
                    segments = np.broadcast_to((positions // 37) % 3, (2, length))
                for start in range(0, length, REL_BIAS_BLOCK):
                    allowed = np.zeros((2, REL_BIAS_BLOCK, length), dtype=bool)
                    for row in range(2):
                        for offset in range(REL_BIAS_BLOCK):
                            query = start + offset
                            lower = 0 if window is None else max(0, query - window + 1)
                            keys = np.arange(lower, query + 1)
                            allowed[row, offset, keys] = segments[row, keys] == segments[row, query]
                    if scenario == "empty":
                        allowed[0, 0] = False
                    _, key_start, width = (int(x) for x in selector(allowed, np.int32(start)))
                    assert 0 <= key_start <= length - width
                    assert not allowed[..., :key_start].any()
                    assert not allowed[..., key_start + width :].any()
                    if scenario == "empty":
                        assert width == length
                    results.append(
                        dict(length=length, window=window, scenario=scenario, query_start=start,
                             key_start=key_start, width=width)
                    )
    return results
