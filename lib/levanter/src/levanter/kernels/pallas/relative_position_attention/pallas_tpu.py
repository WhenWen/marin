# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU Pallas attention with compact input-dependent relative-position bias.

This implementation is derived from JAX's TPU FlashAttention kernel. Instead
of accepting a materialized ``[batch, heads, query, key]`` bias, it accepts a
compact ``[batch, heads, query, relative_extent]`` band. Entry
``[..., q, relative_extent - 1 - d]`` is added to the score for key ``q - d``;
the reversed storage order lets TPU-native slice and reshape operations skew a
band tile into score order without an unsupported gather or reverse.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8


def _output_shape(source: jax.Array, shape: tuple[int, ...], dtype) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        shape,
        dtype,
        manual_axis_type=jax.typeof(source).manual_axis_type,
    )


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are used to generate segment mask, which prevents attention between
    different segments in the input sequence. Each array is a list of ids
    (integers).
    Only the token with the same id can attend to each other.

    Attributes:
      q: segment ids along the Q sequence.
      kv: segment ids along the KV sequence.
    """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tile sizes parameterizing FlashAttention kernels.

    Those parameters have negligible effect on numerics, but affect performance
    greatly.
    """

    block_q: int
    block_k_major: int
    block_k: int
    block_b: int

    block_q_major_dkv: int | None = None
    block_k_major_dkv: int | None = None
    block_k_dkv: int | None = None
    block_q_dkv: int | None = None

    block_k_major_dq: int | None = None
    block_k_dq: int | None = None
    block_q_dq: int | None = None

    def __post_init__(self):
        def verify_major_minor(prefix, suffix, major, minor):
            if minor > major:
                raise ValueError(f"{prefix}{suffix}={minor} should be smaller than" f" {prefix}_major{suffix}={major}")
            if major % minor != 0:
                raise ValueError(f"{prefix}{suffix}={minor} should divide" f" {prefix}_major{suffix}={major}")

        verify_major_minor("block_k", "", self.block_k_major, self.block_k)
        if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
            verify_major_minor("block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv)
        if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
            verify_major_minor("block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv)
        if self.block_k_major_dq is not None and self.block_k_dq is not None:
            verify_major_minor("block_k", "_dq", self.block_k_major_dq, self.block_k_dq)

    @property
    def has_backward_blocks(self) -> bool:
        backward_blocks = (
            self.block_q_major_dkv,
            self.block_k_major_dkv,
            self.block_q_dkv,
            self.block_k_dkv,
            self.block_k_major_dq,
            self.block_k_dq,
            self.block_q_dq,
        )
        return all(b is not None for b in backward_blocks)

    @classmethod
    def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model, relative_extent=None):
        # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
        del batch_size, num_heads, d_model  # Unused.
        max_block = min(512, relative_extent) if relative_extent is not None else 512
        block_q = next(
            candidate for candidate in (512, 256, 128) if candidate <= max_block and q_seq_len % candidate == 0
        )
        block_k = next(
            candidate for candidate in (512, 256, 128) if candidate <= max_block and kv_len % candidate == 0
        )
        return BlockSizes(
            block_q=block_q,
            block_k_major=block_k,
            block_k=block_k,
            block_b=1,
            block_q_major_dkv=block_q,
            block_k_major_dkv=block_k,
            block_k_dkv=block_k,
            block_q_dkv=block_q,
            block_k_major_dq=block_k,
            block_k_dq=block_k,
            block_q_dq=block_q,
        )


@functools.partial(
    jax.jit,
    static_argnames=["causal", "sliding_window", "sm_scale", "block_sizes", "debug", "interpret"],
)
def relative_position_attention_pallas(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    relative_logits=None,  # [batch_size, num_heads, q_seq_len, relative_extent]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    sliding_window: int | None = None,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,
    interpret: bool = False,
):
    batch_size, num_heads, q_seq_len, d_model = q.shape
    batch_size_k, num_heads_k, kv_seq_len, d_model_k = k.shape
    batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = v.shape
    if batch_size != batch_size_k or batch_size != batch_size_v:
        raise ValueError(
            f"Batch size mismatch: got {batch_size}, {batch_size_k} and" f" {batch_size_v} (for q, k, v respectively)"
        )
    if num_heads != num_heads_k or num_heads != num_heads_v:
        raise ValueError(
            f"Head count mismatch: got {num_heads}, {num_heads_k}," f" {num_heads_v} (for q, k, v respectively)"
        )
    if d_model != d_model_k:
        raise ValueError(f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k" " respectively)")
    if d_model != d_model_v:
        raise NotImplementedError("V model dimension unequal to KV model dimension unsupported")
    if kv_seq_len != kv_seq_len_v:
        raise ValueError(f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}")
    if relative_logits is None:
        raise ValueError("relative_logits must have shape [batch, heads, query, relative_extent]")
    if relative_logits.shape[:3] != (batch_size, num_heads, q_seq_len) or relative_logits.ndim != 4:
        raise ValueError(
            "Relative logits shape mismatch: expected "
            f"({batch_size}, {num_heads}, {q_seq_len}, relative_extent), got {relative_logits.shape}"
        )
    if relative_logits.shape[-1] <= 0:
        raise ValueError("relative_extent must be positive")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    if segment_ids is not None:
        if segment_ids.q.shape != (batch_size, q_seq_len):
            raise ValueError(
                f"Q segment ids shape mismatch: expected ({batch_size=}," f" {q_seq_len=},), got {segment_ids.q.shape}"
            )
        if segment_ids.kv.shape != (batch_size, kv_seq_len):
            raise ValueError(
                f"KV segment ids shape mismatch: expected ({batch_size=},"
                f" {kv_seq_len=},), got {segment_ids.kv.shape}"
            )
    if block_sizes is None:
        block_sizes = BlockSizes.get_default(
            batch_size,
            num_heads,
            q_seq_len,
            kv_seq_len,
            d_model,
            relative_logits.shape[-1],
        )
    return _flash_attention(
        q,
        k,
        v,
        relative_logits,
        segment_ids,
        False,
        causal,
        sliding_window,
        sm_scale,
        block_sizes,
        debug,
        interpret,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 12))
def _flash_attention(
    q,
    k,
    v,
    relative_logits,
    segment_ids,
    save_residuals,
    causal,
    sliding_window,
    sm_scale,
    block_sizes,
    debug,
    interpret,
):
    return _flash_attention_impl(
        q,
        k,
        v,
        relative_logits,
        segment_ids,
        save_residuals,
        causal,
        sliding_window,
        sm_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        debug,
        interpret,
    )


def _flash_attention_fwd(
    q,
    k,
    v,
    relative_logits,
    segment_ids,
    save_residuals,
    causal,
    sliding_window,
    sm_scale,
    block_sizes,
    debug,
    interpret,
):
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    o, l, m = _flash_attention(
        q,
        k,
        v,
        relative_logits,
        segment_ids,
        True,
        causal,
        sliding_window,
        sm_scale,
        block_sizes,
        debug,
        interpret,
    )
    return o, (q, k, v, relative_logits, segment_ids, o, l, m)


def _flash_attention_bwd(
    save_residuals: bool,
    causal: bool,
    sliding_window: int | None,
    sm_scale: float,
    block_sizes: BlockSizes,
    debug: bool,
    interpret: bool,
    residuals,
    do,
):
    """VJP rule for FlashAttention."""
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    q, k, v, relative_logits, segment_ids, o, l, m = residuals
    if not block_sizes.has_backward_blocks:
        raise ValueError("Program is being differentiated, but not all backward blocks are" " specified")

    di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)  # [batch_size, num_heads, q_seq_len]

    dk, dv = _flash_attention_bwd_dkv(
        q,
        k,
        v,
        relative_logits,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_major_dkv,  # pyrefly: ignore[bad-argument-type]
        block_k_major=block_sizes.block_k_major_dkv,  # pyrefly: ignore[bad-argument-type]
        block_k=block_sizes.block_k_dkv,  # pyrefly: ignore[bad-argument-type]
        block_q=block_sizes.block_q_dkv,  # pyrefly: ignore[bad-argument-type]
        sm_scale=sm_scale,
        causal=causal,
        sliding_window=sliding_window,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
        interpret=interpret,
    )

    dq, ds = _flash_attention_bwd_dq(
        q,
        k,
        v,
        relative_logits,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_dq,  # pyrefly: ignore[bad-argument-type]
        block_k_major=block_sizes.block_k_major_dq,  # pyrefly: ignore[bad-argument-type]
        block_k=block_sizes.block_k_dq,  # pyrefly: ignore[bad-argument-type]
        sm_scale=sm_scale,
        causal=causal,
        sliding_window=sliding_window,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
        interpret=interpret,
    )
    return dq, dk, dv, ds, None


_flash_attention.defvjp(fwd=_flash_attention_fwd, bwd=_flash_attention_bwd)


MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
    # A block is considered below or on diagonal as long as the bottom left
    # corner of the block is below or on diagonal.
    return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


def _relative_distances(
    *,
    block_q: int,
    block_k: int,
    query_start: int | jax.Array,
    key_start: int | jax.Array,
) -> jax.Array:
    row_ids = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_k), 0)
    row_ids += query_start
    col_ids = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1)
    col_ids += key_start
    return row_ids - col_ids


def _stored_relative_start(
    *,
    relative_extent: int,
    block_q: int,
    block_k: int,
    q_seq_index: int | jax.Array,
    kv_seq_index: int | jax.Array,
    padding_before: int = 0,
) -> int | jax.Array:
    for name, value in (
        ("relative_extent", relative_extent),
        ("block_q", block_q),
        ("block_k", block_k),
        ("padding_before", padding_before),
    ):
        if value % MIN_BLOCK_SIZE:
            raise ValueError(f"{name}={value} must be a multiple of {MIN_BLOCK_SIZE}")
    block_start = (
        padding_before // MIN_BLOCK_SIZE
        + relative_extent // MIN_BLOCK_SIZE
        - block_q // MIN_BLOCK_SIZE
        - q_seq_index * (block_q // MIN_BLOCK_SIZE)
        + kv_seq_index * (block_k // MIN_BLOCK_SIZE)
    )
    return block_start * MIN_BLOCK_SIZE


def _relative_bias_block(
    relative_logits_tile_ref,
    *,
    block_q: int,
    block_k: int,
    relative_extent: int,
    query_start: int | jax.Array,
    key_start: int | jax.Array,
) -> jax.Array:
    relative_block_size = relative_logits_tile_ref.shape[-1]
    relative_span = block_q + block_k - 1
    if relative_span >= relative_block_size:
        raise ValueError(f"relative block size {relative_block_size} must exceed the score-tile span {relative_span}")
    relative_logits = jnp.reshape(relative_logits_tile_ref[...], (block_q, relative_block_size)).astype(jnp.float32)
    stored_start = relative_extent - block_q - query_start + key_start
    load_start = jnp.maximum(stored_start, 0)
    shift = relative_block_size - (block_q - 1) - stored_start + load_start
    skewed = pltpu.roll(
        relative_logits,
        shift,
        axis=1,
        stride=1,
        stride_axis=0,
    )
    bias = skewed[:, :block_k]
    distances = _relative_distances(
        block_q=block_q,
        block_k=block_k,
        query_start=query_start,
        key_start=key_start,
    )
    valid = (distances >= 0) & (distances < relative_extent)
    return jnp.where(valid, bias, 0.0)


def _pack_softmax_residuals(l: jax.Array, m: jax.Array) -> jax.Array:
    block_q = l.shape[0]
    packed_rows = block_q // MIN_BLOCK_SIZE
    if block_q % MIN_BLOCK_SIZE or 2 * packed_rows > NUM_SUBLANES:
        raise ValueError(f"cannot pack softmax residuals for query block size {block_q}")
    packed = [
        jnp.reshape(l[:, 0], (packed_rows, MIN_BLOCK_SIZE)),
        jnp.reshape(m[:, 0], (packed_rows, MIN_BLOCK_SIZE)),
    ]
    if 2 * packed_rows < NUM_SUBLANES:
        packed.append(jnp.zeros((NUM_SUBLANES - 2 * packed_rows, MIN_BLOCK_SIZE), dtype=l.dtype))
    return jnp.concatenate(packed, axis=0)


def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
    block_b = q_tile_ref.shape[0]
    if block_b != 1:
        raise ValueError(f"TPU relative-position attention requires block_b=1, got {block_b}")
    # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
    if kwargs["block_k"] == kwargs["kv_seq_len"]:
        kernel = _flash_attention_kernel_single_batch_single_step
    else:
        kernel = _flash_attention_kernel_single_batch
    kernel(q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_single_batch(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    relative_logits_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    lm_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    sliding_window,
    sm_scale,
    block_k,
    kv_seq_len,
    relative_extent,
    mask_value,
    kv_program_count,
    sparse_window,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]
    head_dim = q_tile_ref.shape[-1]

    kv_program_idx = pl.program_id(3)
    q_seq_idx = pl.program_id(2)
    latest_kv_seq_idx = ((q_seq_idx + 1) * block_q - 1) // block_k_major
    kv_seq_idx = latest_kv_seq_idx - (kv_program_count - 1) + kv_program_idx if sparse_window else kv_program_idx

    @pl.when(kv_program_idx == 0)
    def start_new_sequence():
        m_scratch_ref[...] = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
        l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
        acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

    valid_kv_block = (kv_seq_idx >= 0) & (kv_seq_idx < kv_seq_len // block_k_major)
    if causal:
        should_run = valid_kv_block & below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
    else:
        should_run = valid_kv_block
    if sliding_window is not None:
        q_block_start = q_seq_idx * block_q
        kv_block_end = (kv_seq_idx + 1) * block_k_major - 1
        should_run = should_run & (kv_block_end >= q_block_start - (sliding_window - 1))

    @pl.when(should_run)
    def run():
        m_prev = jnp.reshape(m_scratch_ref[...], (block_q, MIN_BLOCK_SIZE))
        l_prev = jnp.reshape(l_scratch_ref[...], (block_q, MIN_BLOCK_SIZE))
        q = jnp.reshape(q_tile_ref[...], (block_q, head_dim))
        k = jnp.reshape(k_tile_ref[...], (block_k, head_dim))

        s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)  # [block_q, block_k]

        if sm_scale != 1.0:
            s *= sm_scale

        if relative_logits_tile_ref is not None:
            s += _relative_bias_block(
                relative_logits_tile_ref,
                block_q=block_q,
                block_k=block_k,
                relative_extent=relative_extent,
                query_start=q_seq_idx * block_q,
                key_start=kv_seq_idx * block_k_major,
            ).astype(jnp.float32)

        mask = None
        if q_segment_ids_tile_ref is not None:
            repeats, rem = divmod(block_k, NUM_LANES)
            if rem:
                raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
            q_segment_ids = jnp.reshape(q_segment_ids_tile_ref[...], (block_q, NUM_LANES))
            q_segment_ids = jnp.tile(q_segment_ids, (1, repeats))
            kv_segment_ids = jnp.reshape(kv_segment_ids_tile_ref[...], (NUM_SUBLANES, block_k))[:1]
            mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

        distances = _relative_distances(
            block_q=block_q,
            block_k=block_k,
            query_start=q_seq_idx * block_q,
            key_start=kv_seq_idx * block_k_major,
        )
        if causal:
            causal_mask = distances >= 0
            mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        if sliding_window is not None:
            window_mask = distances < sliding_window
            mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)

        s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

        m_curr = jnp.max(s, axis=1)[:, None]
        m_next = jnp.maximum(m_prev, m_curr)

        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
            raise NotImplementedError(f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}")
        p = jnp.exp(s - jnp.tile(m_next, (1, block_k_repeats)))

        alpha = jnp.exp(m_prev - m_next)
        l_corr = alpha * l_prev
        l_next = jnp.sum(p, axis=1)[:, None] + l_corr

        head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
        l_broadcast = lambda l: jnp.tile(l, (1, head_dim_repeats))
        if rem:
            if head_dim_repeats == 0:
                l_broadcast = lambda l: l[:, :head_dim]
            else:
                raise NotImplementedError(f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger")
        l_scratch_ref[...] = jnp.reshape(l_next, l_scratch_ref.shape)
        m_scratch_ref[...] = jnp.reshape(m_next, m_scratch_ref.shape)

        accumulator = jnp.reshape(acc_scratch_ref[...], (block_q, head_dim))
        accumulator *= l_broadcast(alpha)
        v = jnp.reshape(v_tile_ref[...], (block_k, head_dim))
        o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
        accumulator += o_curr
        acc_scratch_ref[...] = jnp.reshape(accumulator, acc_scratch_ref.shape)

    @pl.when(kv_program_idx == kv_program_count - 1)
    def store_output():
        l = jnp.reshape(l_scratch_ref[...], (block_q, MIN_BLOCK_SIZE))
        l_inv_safe = jnp.where(l == 0.0, 1.0, 1.0 / l)
        head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
        if rem:
            if head_dim_repeats == 0:
                l_inv_safe = l_inv_safe[:, :head_dim]
            else:
                raise NotImplementedError(f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger")
        else:
            l_inv_safe = jnp.tile(l_inv_safe, (1, head_dim_repeats))
        output = jnp.reshape(acc_scratch_ref[...], (block_q, head_dim)) * l_inv_safe
        o_tile_ref[...] = jnp.reshape(output, o_tile_ref.shape).astype(o_tile_ref.dtype)
        if lm_ref is not None:
            m = jnp.reshape(m_scratch_ref[...], (block_q, MIN_BLOCK_SIZE))
            lm_ref[...] = _pack_softmax_residuals(l, m).astype(lm_ref.dtype)


def _flash_attention_kernel_single_batch_single_step(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    relative_logits_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    lm_ref: Any | None = None,
    *,
    causal,
    sliding_window,
    sm_scale,
    block_k,
    kv_seq_len,
    relative_extent,
    mask_value,
    kv_program_count,
    sparse_window,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]
    head_dim = q_tile_ref.shape[3]

    assert kv_seq_len == block_k_major == block_k
    del kv_program_count, sparse_window

    q = jnp.reshape(q_tile_ref[...], (block_q, head_dim))
    k = jnp.reshape(k_tile_ref[...], (block_k, head_dim))
    s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)  # [block_q, block_k]

    if sm_scale != 1.0:
        s *= sm_scale

    if relative_logits_tile_ref is not None:
        q_seq_idx = pl.program_id(2)
        s += _relative_bias_block(
            relative_logits_tile_ref,
            block_q=block_q,
            block_k=block_k,
            relative_extent=relative_extent,
            query_start=q_seq_idx * block_q,
            key_start=0,
        ).astype(jnp.float32)

    mask = None
    if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
            raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
        q_segment_ids = jnp.reshape(q_segment_ids_tile_ref[...], (block_q, NUM_LANES))
        q_segment_ids = jnp.tile(q_segment_ids, (1, repeats))  # [block_q, block_k].
        kv_segment_ids = jnp.reshape(kv_segment_ids_tile_ref[...], (NUM_SUBLANES, block_k))[:1]
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
        q_seq_idx = pl.program_id(2)
        distances = _relative_distances(
            block_q=block_q,
            block_k=block_k,
            query_start=q_seq_idx * block_q,
            key_start=0,
        )
        causal_mask = distances >= 0
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    if sliding_window is not None:
        q_seq_idx = pl.program_id(2)
        distances = _relative_distances(
            block_q=block_q,
            block_k=block_k,
            query_start=q_seq_idx * block_q,
            key_start=0,
        )
        window_mask = distances < sliding_window
        mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)
    s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

    m = jnp.max(s, axis=1)[:, None]
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=1)[:, None]
    p /= l

    if lm_ref is not None:
        lm_ref[...] = _pack_softmax_residuals(l, m).astype(lm_ref.dtype)

    v = jnp.reshape(v_tile_ref[...], (block_k, head_dim))
    output = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
    o_tile_ref[...] = jnp.reshape(output.astype(o_tile_ref.dtype), o_tile_ref.shape)


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    relative_logits: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    sliding_window: int | None,
    sm_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
    body_cost = pl.estimate_cost(
        mha_reference,
        q,
        k,
        v,
        relative_logits,
        segment_ids,
        causal=causal,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
    )
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=kernel_inputs_specs,
        kernel_outputs_specs=kernel_outputs_specs,
    )


def _bwd_cost_estimate(
    q,
    k,
    v,
    relative_logits,
    segment_ids,
    do,
    *,
    causal: bool,
    sliding_window: int | None,
    sm_scale: float,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate:
    del do
    forward_cost = pl.estimate_cost(
        mha_reference,
        q,
        k,
        v,
        relative_logits,
        segment_ids,
        causal=causal,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
    )
    # Backward evaluates the score path and three matrix products. The explicit
    # multiplier avoids JAX cost-estimator failures on the differentiated
    # softmax while retaining shape-dependent scheduling information.
    body_cost = pl.CostEstimate(
        flops=3 * forward_cost.flops,
        transcendentals=2 * forward_cost.transcendentals,
        bytes_accessed=forward_cost.bytes_accessed,
        remote_bytes_transferred=forward_cost.remote_bytes_transferred,
    )
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=kernel_inputs_specs,
        kernel_outputs_specs=kernel_outputs_specs,
    )


def _flash_attention_impl(
    q,
    k,
    v,
    relative_logits,
    segment_ids,
    save_residuals,
    causal,
    sliding_window,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
    interpret,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    relative_extent = relative_logits.shape[-1]
    relative_block_size = math.ceil((block_q + block_k_major - 1) / MIN_BLOCK_SIZE) * MIN_BLOCK_SIZE
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)
    if block_k_major != block_k:
        raise ValueError("relative-position attention requires equal major and compute key block sizes")

    kv_block_count = kv_seq_len // block_k_major
    sparse_window = causal and sliding_window is not None
    kv_program_count = (
        min(kv_block_count, math.ceil((sliding_window + block_q - 1) / block_k_major))
        if sparse_window
        else kv_block_count
    )

    def logical_kv_index(q_seq_index, kv_program_index):
        if sparse_window:
            latest_kv_seq_index = ((q_seq_index + 1) * block_q - 1) // block_k_major
            return latest_kv_seq_index - (kv_program_count - 1) + kv_program_index
        return kv_program_index

    def safe_kv_index(q_seq_index, kv_program_index):
        kv_seq_index = logical_kv_index(q_seq_index, kv_program_index)
        return lax.clamp(0, kv_seq_index, kv_block_count - 1)

    # TODO(apaszke): Tile over heads as well.
    grid = (
        pl.cdiv(batch_size, block_b),
        num_heads,
        pl.cdiv(q_seq_len, block_q),
        kv_program_count,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        next_kv_index = safe_kv_index(q_seq_index, kv_seq_index)
        return (batch_index, head_index, next_kv_index, 0)

    def relative_logits_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        kv_seq_index = safe_kv_index(q_seq_index, kv_seq_index)
        stored_start = _stored_relative_start(
            relative_extent=relative_extent,
            block_q=block_q,
            block_k=block_k_major,
            q_seq_index=q_seq_index,
            kv_seq_index=kv_seq_index,
        )
        load_start = lax.select(stored_start >= 0, stored_start, 0)
        return (
            batch_index,
            head_index,
            q_seq_index * block_q,
            load_start,
        )

    def o_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        sliding_window=sliding_window,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        relative_extent=relative_extent,
        kv_program_count=kv_program_count,
        sparse_window=sparse_window,
    )
    out_shape = _output_shape(q, q.shape, q.dtype)
    out_shape = [out_shape]
    out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

    if block_k != kv_seq_len:
        m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
        scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
        scratch_shapes = []

    if save_residuals:
        out_specs = [
            *out_specs,
            pl.BlockSpec((pl.squeezed, pl.squeezed, NUM_SUBLANES, MIN_BLOCK_SIZE), lm_index_map),
        ]
        lm = _output_shape(
            q,
            (batch_size, num_heads, pl.cdiv(q_seq_len, block_q) * NUM_SUBLANES, MIN_BLOCK_SIZE),
            jnp.float32,
        )
        out_shape = (*out_shape, lm)
    else:
        out_specs = [*out_specs, None]
        out_shape = (*out_shape, None)

    relative_logits_block_spec = (
        pl.BlockSpec(
            (
                pl.squeezed,
                pl.squeezed,
                pl.Element(block_q),
                pl.Element(
                    relative_block_size,
                    padding=(0, q_seq_len + relative_block_size),
                ),
            ),
            relative_logits_index_map,
        )
        if relative_logits is not None
        else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
            del head_index
            next_kv_index = safe_kv_index(q_seq_index, kv_seq_index)
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((block_b, block_q, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        relative_logits_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    o, *aux = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            )
        ),
        cost_estimate=_fwd_cost_estimate(
            q,
            k,
            v,
            relative_logits,
            segment_ids,
            causal=causal,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            kernel_inputs_specs=(q, k, v, relative_logits, q_segment_ids, kv_segment_ids),
            kernel_outputs_specs=out_shape,
        ),
    )(q, k, v, relative_logits, q_segment_ids, kv_segment_ids)
    if save_residuals:
        lm = jnp.reshape(
            aux[-1],
            (batch_size, num_heads, pl.cdiv(q_seq_len, block_q), NUM_SUBLANES, MIN_BLOCK_SIZE),
        )
        packed_rows = block_q // MIN_BLOCK_SIZE
        l = jnp.reshape(lm[..., :packed_rows, :], (batch_size, num_heads, q_seq_len))
        m = jnp.reshape(lm[..., packed_rows : 2 * packed_rows, :], (batch_size, num_heads, q_seq_len))
        return (o, l, m)
    else:
        return o


def _flash_attention_dkv_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    relative_logits_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    sliding_window: int | None,
    mask_value: float,
    q_seq_len: int,
    relative_extent: int,
    block_q: int,
    block_k: int,
    q_program_count: int,
    sparse_window: bool,
):
    _, _, block_q_major, _ = q_tile_ref.shape
    _, _, block_k_major, _ = k_tile_ref.shape
    head_dim = q_tile_ref.shape[3]

    q_program_index = pl.program_id(axis=3)
    kv_seq_index = pl.program_id(axis=2)
    first_q_seq_index = (kv_seq_index * block_k_major) // block_q_major
    q_seq_index = first_q_seq_index + q_program_index if sparse_window else q_program_index

    @pl.when(q_program_index == 0)
    def start_new_sequence():
        dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
        dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

    def accumulate():
        k = jnp.reshape(k_tile_ref[...], (block_k, head_dim))
        v = jnp.reshape(v_tile_ref[...], (block_k, head_dim))
        q = jnp.reshape(q_tile_ref[...], (block_q, head_dim))
        l = jnp.reshape(l_tile_ref[...], (block_q, MIN_BLOCK_SIZE))
        m = jnp.reshape(m_tile_ref[...], (block_q, MIN_BLOCK_SIZE))
        do = jnp.reshape(do_tile_ref[...], (block_q, head_dim))
        di = jnp.reshape(di_tile_ref[...], (block_q, MIN_BLOCK_SIZE)).astype(jnp.float32)

        capped_logits = lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        if sm_scale != 1.0:
            capped_logits *= sm_scale

        if relative_logits_tile_ref is not None:
            capped_logits += _relative_bias_block(
                relative_logits_tile_ref,
                block_q=block_q,
                block_k=block_k,
                relative_extent=relative_extent,
                query_start=q_seq_index * block_q_major,
                key_start=kv_seq_index * block_k_major,
            ).astype(jnp.float32)

        mask = None
        if q_segment_ids_tile_ref is not None:
            repeats, rem = divmod(block_k, NUM_LANES)
            if rem:
                raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
            q_segment_ids = jnp.reshape(q_segment_ids_tile_ref[...], (block_q, NUM_LANES))
            q_segment_ids = jnp.tile(q_segment_ids, (1, repeats))
            kv_segment_ids = jnp.reshape(kv_segment_ids_tile_ref[...], (NUM_SUBLANES, block_k))[:1]
            mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

        distances = _relative_distances(
            block_q=block_q,
            block_k=block_k,
            query_start=q_seq_index * block_q_major,
            key_start=kv_seq_index * block_k_major,
        )
        if causal:
            causal_mask = distances >= 0
            mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        if sliding_window is not None:
            window_mask = distances < sliding_window
            mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)

        capped_logits = capped_logits if mask is None else capped_logits + jnp.where(mask, 0.0, mask_value)

        p = jnp.exp(capped_logits - jnp.tile(m, (1, block_k // MIN_BLOCK_SIZE)))
        p = p * jnp.tile(1 / l, (1, block_k // MIN_BLOCK_SIZE))
        dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
        dv_scratch_ref[...] += dv.astype(dv_scratch_ref.dtype)

        dp = lax.dot_general(do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)
        ds = (dp - jnp.tile(di, (1, block_k // MIN_BLOCK_SIZE))) * p
        ds_qk = ds * sm_scale if sm_scale != 1.0 else ds
        dk = lax.dot(ds_qk.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
        dk_scratch_ref[...] += dk.astype(dk_scratch_ref.dtype)

    valid_q_block = (q_seq_index >= 0) & (q_seq_index < q_seq_len // block_q_major)
    if causal:
        should_run = valid_q_block & below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major)
    else:
        should_run = valid_q_block
    if sliding_window is not None:
        q_block_start = q_seq_index * block_q_major
        kv_block_end = (kv_seq_index + 1) * block_k_major - 1
        should_run = should_run & (q_block_start <= kv_block_end + (sliding_window - 1))

    @pl.when(should_run)
    def run():
        accumulate()

    @pl.when(q_program_index == q_program_count - 1)
    def end_of_q_sequence():
        dv_tile_ref[...] = jnp.reshape(dv_scratch_ref[...], dv_tile_ref.shape).astype(dv_tile_ref.dtype)
        dk_tile_ref[...] = jnp.reshape(dk_scratch_ref[...], dk_tile_ref.shape).astype(dk_tile_ref.dtype)


def _flash_attention_bwd_dkv(
    q,
    k,
    v,
    relative_logits,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int,
    block_q: int,
    block_k_major: int,
    block_k: int,
    sm_scale: float,
    causal: bool = False,
    sliding_window: int | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
    interpret: bool = False,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    relative_extent = relative_logits.shape[-1]
    relative_block_size = math.ceil((block_q_major + block_k_major - 1) / MIN_BLOCK_SIZE) * MIN_BLOCK_SIZE
    _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
    _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)
    if block_q_major != block_q or block_k_major != block_k:
        raise ValueError("relative-position dKV currently requires equal major and compute block sizes")

    q_block_count = q_seq_len // block_q_major
    sparse_window = causal and sliding_window is not None
    q_program_count = (
        min(q_block_count, math.ceil((sliding_window + block_k_major - 1) / block_q_major))
        if sparse_window
        else q_block_count
    )

    def logical_q_index(kv_seq_index, q_program_index):
        if sparse_window:
            first_q_seq_index = (kv_seq_index * block_k_major) // block_q_major
            return first_q_seq_index + q_program_index
        return q_program_index

    def safe_q_index(kv_seq_index, q_program_index):
        q_seq_index = logical_q_index(kv_seq_index, q_program_index)
        return lax.clamp(0, q_seq_index, q_block_count - 1)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

    # kv index needs to be before q index since q index is the contractng
    # dimension.
    grid = (
        batch_size,
        num_heads,
        kv_seq_len // block_k_major,
        q_program_count,
    )

    def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
        next_q_index = safe_q_index(kv_seq_index, q_seq_index)
        return (batch_index, head_index, next_q_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)
    do_spec = qo_spec
    assert do.ndim == len(qo_spec.block_shape)

    def kv_index_map(batch_index, head_index, kv_seq_index, _):
        return (batch_index, head_index, kv_seq_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
        q_seq_index = safe_q_index(kv_seq_index, q_seq_index)
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def relative_logits_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
        q_seq_index = safe_q_index(kv_seq_index, q_seq_index)
        stored_start = _stored_relative_start(
            relative_extent=relative_extent,
            block_q=block_q_major,
            block_k=block_k_major,
            q_seq_index=q_seq_index,
            kv_seq_index=kv_seq_index,
        )
        load_start = lax.select(stored_start >= 0, stored_start, 0)
        return (
            batch_index,
            head_index,
            q_seq_index * block_q_major,
            load_start,
        )

    drelative_logits_spec = (
        pl.BlockSpec(
            (
                pl.squeezed,
                pl.squeezed,
                pl.Element(block_q_major),
                pl.Element(
                    relative_block_size,
                    padding=(0, q_seq_len + relative_block_size),
                ),
            ),
            relative_logits_index_map,
        )
        if relative_logits is not None
        else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
            del head_index
            next_q_index = safe_q_index(kv_seq_index, q_seq_index)
            return (batch_index, next_q_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _):
            del head_index
            return (batch_index, 0, kv_seq_index)

        q_segment_ids_spec = pl.BlockSpec((1, block_q_major, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        drelative_logits_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        _output_shape(k, (batch_size, num_heads, kv_seq_len, head_dim), k.dtype),
        _output_shape(v, (batch_size, num_heads, kv_seq_len, head_dim), v.dtype),
    ]

    def dkv_index_map(batch_index, head_index, kv_seq_index, _):
        return (batch_index, head_index, kv_seq_index, 0)

    dkv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), dkv_index_map)
    out_specs = [dkv_spec, dkv_spec]
    scratch_shapes = [
        pltpu.VMEM((block_k_major, head_dim), jnp.float32),
        pltpu.VMEM((block_k_major, head_dim), jnp.float32),
    ]

    kernel = functools.partial(
        _flash_attention_dkv_kernel,
        block_q=block_q,
        block_k=block_k,
        sm_scale=sm_scale,
        causal=causal,
        sliding_window=sliding_window,
        mask_value=mask_value,
        q_seq_len=q_seq_len,
        relative_extent=relative_extent,
        q_program_count=q_program_count,
        sparse_window=sparse_window,
    )
    name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dk, dv = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shapes,
            debug=debug,
            interpret=interpret,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            ),
            cost_estimate=_bwd_cost_estimate(
                q,
                k,
                v,
                relative_logits,
                segment_ids,
                do,
                causal=causal,
                sliding_window=sliding_window,
                sm_scale=sm_scale,
                kernel_inputs_specs=(
                    q,
                    k,
                    v,
                    relative_logits,
                    q_segment_ids,
                    kv_segment_ids,
                    l,
                    m,
                    do,
                    di,
                ),
                kernel_outputs_specs=out_shapes,
            ),
        )(q, k, v, relative_logits, q_segment_ids, kv_segment_ids, l, m, do, di)
        assert dk.shape == k.shape
        assert dv.shape == v.shape
    return dk, dv


def _flash_attention_dq_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    relative_logits_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    ds_tile_ref,
    dq_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    sliding_window: int | None,
    mask_value: float,
    kv_seq_len: int,
    relative_extent: int,
    relative_padding: int,
    block_k: int,
    kv_program_count: int,
    sparse_window: bool,
):
    _, _, block_k_major, _ = k_tile_ref.shape
    _, _, block_q_major, _ = q_tile_ref.shape
    head_dim = q_tile_ref.shape[3]
    relative_block_size = relative_logits_tile_ref.shape[-1] if relative_logits_tile_ref is not None else 0
    relative_span = block_q_major + block_k - 1

    kv_program_index = pl.program_id(axis=3)
    q_seq_index = pl.program_id(axis=2)
    latest_kv_seq_index = ((q_seq_index + 1) * block_q_major - 1) // block_k_major
    kv_seq_index = (
        latest_kv_seq_index - (kv_program_count - 1) + kv_program_index if sparse_window else kv_program_index
    )

    @pl.when(kv_program_index == 0)
    def start_new_sequence():
        dq_scratch_ref[...] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)
        if ds_tile_ref is not None:
            ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

    def accumulate():
        q = jnp.reshape(q_tile_ref[...], (block_q_major, head_dim))
        k = jnp.reshape(k_tile_ref[...], (block_k, head_dim))
        v = jnp.reshape(v_tile_ref[...], (block_k, head_dim))
        l = jnp.reshape(l_tile_ref[...], (block_q_major, MIN_BLOCK_SIZE))
        m = jnp.reshape(m_tile_ref[...], (block_q_major, MIN_BLOCK_SIZE))
        do = jnp.reshape(do_tile_ref[...], (block_q_major, head_dim))
        di = jnp.reshape(di_tile_ref[...], (block_q_major, MIN_BLOCK_SIZE)).astype(jnp.float32)

        capped_logits = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        if sm_scale != 1.0:
            capped_logits *= sm_scale

        if relative_logits_tile_ref is not None:
            capped_logits += _relative_bias_block(
                relative_logits_tile_ref,
                block_q=block_q_major,
                block_k=block_k,
                relative_extent=relative_extent,
                query_start=q_seq_index * block_q_major,
                key_start=kv_seq_index * block_k_major,
            ).astype(jnp.float32)

        mask = None
        if q_segment_ids_tile_ref is not None:
            repeats, rem = divmod(block_k, NUM_LANES)
            if rem:
                raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
            q_segment_ids = jnp.reshape(q_segment_ids_tile_ref[...], (block_q_major, NUM_LANES))
            q_segment_ids = jnp.tile(q_segment_ids, (1, repeats))
            kv_segment_ids = jnp.reshape(kv_segment_ids_tile_ref[...], (NUM_SUBLANES, block_k))[:1]
            mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

        distances = _relative_distances(
            block_q=block_q_major,
            block_k=block_k,
            query_start=q_seq_index * block_q_major,
            key_start=kv_seq_index * block_k_major,
        )
        if causal:
            causal_mask = distances >= 0
            mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        if sliding_window is not None:
            window_mask = distances < sliding_window
            mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)
        capped_logits = capped_logits if mask is None else capped_logits + jnp.where(mask, 0.0, mask_value)

        p = jnp.exp(capped_logits - jnp.tile(m, (1, block_k // MIN_BLOCK_SIZE)))
        p = p * jnp.tile(1 / l, (1, block_k // MIN_BLOCK_SIZE))  # [block_q_major, block_k]

        # di: [block_q_major, 128]
        # do: [block_q_major, head_dim]
        # v: [block_k_major, head_dim]
        dp = jax.lax.dot_general(
            do,
            v,
            TRANS_B_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        ds = (dp - jnp.tile(di, (1, block_k // MIN_BLOCK_SIZE))) * p
        if mask is not None:
            ds = jnp.where(mask, ds, 0.0)

        if ds_tile_ref is not None:
            if relative_span >= relative_block_size:
                raise ValueError(
                    f"relative block size {relative_block_size} must exceed the score-tile span {relative_span}"
                )
            score_gradient_tile = jnp.concatenate(
                [
                    ds,
                    jnp.zeros((block_q_major, relative_block_size - block_k), ds.dtype),
                ],
                axis=1,
            )
            stored_start = _stored_relative_start(
                relative_extent=relative_extent,
                block_q=block_q_major,
                block_k=block_k_major,
                q_seq_index=q_seq_index,
                kv_seq_index=kv_seq_index,
            )
            compact_tile = pltpu.roll(
                score_gradient_tile,
                (relative_block_size + block_q_major - 1 + stored_start - jnp.maximum(stored_start, 0)),
                axis=1,
                stride=relative_block_size - 1,
                stride_axis=0,
            )
            relative_columns = jax.lax.broadcasted_iota(jnp.int32, compact_tile.shape, 1) + stored_start
            compact_tile = jnp.where((relative_columns >= 0) & (relative_columns < relative_extent), compact_tile, 0.0)

            @pl.when((stored_start < relative_extent) & (stored_start + relative_block_size > 0))
            def store_relative_gradient():
                write_start = stored_start + relative_padding
                ds_tile_ref[:, :, :, pl.ds(write_start, relative_block_size)] += jnp.reshape(
                    compact_tile.astype(ds_tile_ref.dtype), (1, 1, block_q_major, relative_block_size)
                )

        ds_qk = ds * sm_scale if sm_scale != 1.0 else ds

        # dp: [block_q_major, block_k]
        # k: [block_k, head_dim]
        dq_scratch_ref[...] += lax.dot(
            ds_qk.astype(k.dtype),
            k,
            preferred_element_type=jnp.float32,
        ).astype(dq_scratch_ref.dtype)

    valid_kv_block = (kv_seq_index >= 0) & (kv_seq_index < kv_seq_len // block_k_major)
    if causal:
        should_run = valid_kv_block & below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major)
    else:
        should_run = valid_kv_block
    if sliding_window is not None:
        q_block_start = q_seq_index * block_q_major
        kv_block_end = (kv_seq_index + 1) * block_k_major - 1
        should_run = should_run & (kv_block_end >= q_block_start - (sliding_window - 1))

    @pl.when(should_run)
    def run():
        accumulate()

    @pl.when(kv_program_index == kv_program_count - 1)
    def end_of_kv_sequence():
        dq_tile_ref[...] = jnp.reshape(dq_scratch_ref[...], dq_tile_ref.shape).astype(dq_tile_ref.dtype)
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
    q,
    k,
    v,
    relative_logits,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int,
    block_k_major: int,
    block_k: int,
    sm_scale: float,
    causal: bool,
    sliding_window: int | None,
    mask_value: float,
    debug: bool,
    interpret: bool,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    relative_extent = relative_logits.shape[-1]
    relative_block_size = math.ceil((block_q_major + block_k_major - 1) / MIN_BLOCK_SIZE) * MIN_BLOCK_SIZE
    _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)
    if block_k_major != block_k:
        raise ValueError("relative-position dQ currently requires equal major and compute block sizes")

    kv_block_count = kv_seq_len // block_k_major
    sparse_window = causal and sliding_window is not None
    kv_program_count = (
        min(kv_block_count, math.ceil((sliding_window + block_q_major - 1) / block_k_major))
        if sparse_window
        else kv_block_count
    )

    def logical_kv_index(q_seq_index, kv_program_index):
        if sparse_window:
            latest_kv_seq_index = ((q_seq_index + 1) * block_q_major - 1) // block_k_major
            return latest_kv_seq_index - (kv_program_count - 1) + kv_program_index
        return kv_program_index

    def safe_kv_index(q_seq_index, kv_program_index):
        kv_seq_index = logical_kv_index(q_seq_index, kv_program_index)
        return lax.clamp(0, kv_seq_index, kv_block_count - 1)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

    grid = (
        batch_size,
        num_heads,
        q_seq_len // block_q_major,
        kv_program_count,
    )

    def qo_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    do_spec = qo_spec

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        next_kv_index = safe_kv_index(q_seq_index, kv_seq_index)
        return (batch_index, head_index, next_kv_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def relative_logits_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        kv_seq_index = safe_kv_index(q_seq_index, kv_seq_index)
        stored_start = _stored_relative_start(
            relative_extent=relative_extent,
            block_q=block_q_major,
            block_k=block_k_major,
            q_seq_index=q_seq_index,
            kv_seq_index=kv_seq_index,
        )
        load_start = lax.select(stored_start >= 0, stored_start, 0)
        return (
            batch_index,
            head_index,
            q_seq_index * block_q_major,
            load_start,
        )

    relative_logits_spec = (
        pl.BlockSpec(
            (
                pl.squeezed,
                pl.squeezed,
                pl.Element(block_q_major),
                pl.Element(
                    relative_block_size,
                    padding=(0, q_seq_len + relative_block_size),
                ),
            ),
            relative_logits_index_map,
        )
        if relative_logits is not None
        else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
            del head_index
            next_kv_index = safe_kv_index(q_seq_index, kv_seq_index)
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((1, block_q_major, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        relative_logits_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    relative_padding = relative_block_size
    drelative_width = (
        math.ceil((relative_padding + relative_extent + relative_block_size) / MIN_BLOCK_SIZE) * MIN_BLOCK_SIZE
    )
    out_shapes = [
        _output_shape(q, q.shape, q.dtype),
        (
            _output_shape(
                relative_logits,
                (*relative_logits.shape[:-1], drelative_width),
                relative_logits.dtype,
            )
            if relative_logits is not None
            else None
        ),
    ]
    dq_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    drelative_logits_spec = (
        pl.BlockSpec(
            (1, 1, block_q_major, drelative_width),
            qo_index_map,
        )
        if relative_logits is not None
        else None
    )
    out_specs = [
        dq_spec,
        drelative_logits_spec,
    ]
    scratch_shapes = [pltpu.VMEM((block_q_major, head_dim), jnp.float32)]

    kernel = functools.partial(
        _flash_attention_dq_kernel,
        sm_scale=sm_scale,
        causal=causal,
        sliding_window=sliding_window,
        mask_value=mask_value,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        relative_extent=relative_extent,
        relative_padding=relative_padding,
        kv_program_count=kv_program_count,
        sparse_window=sparse_window,
    )
    name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dq, ds = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shapes,
            debug=debug,
            interpret=interpret,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            ),
            cost_estimate=_bwd_cost_estimate(
                q,
                k,
                v,
                relative_logits,
                segment_ids,
                do,
                causal=causal,
                sliding_window=sliding_window,
                sm_scale=sm_scale,
                kernel_inputs_specs=(
                    q,
                    k,
                    v,
                    relative_logits,
                    q_segment_ids,
                    kv_segment_ids,
                    l,
                    m,
                    do,
                    di,
                ),
                kernel_outputs_specs=out_shapes,
            ),
        )(q, k, v, relative_logits, q_segment_ids, kv_segment_ids, l, m, do, di)

    return dq, ds[..., relative_padding : relative_padding + relative_extent]


@functools.partial(
    jax.jit,
    static_argnames=["causal", "sliding_window", "mask_value", "sm_scale"],
)
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q,
    k,
    v,
    relative_logits,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    sliding_window: int | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
):
    """Readable JAX reference for compact relative-position attention."""
    batch_size, num_heads, q_seq_len, _ = q.shape
    kv_seq_len = k.shape[2]
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32)
    if sm_scale != 1.0:
        logits *= sm_scale

    query_ids = jnp.arange(q_seq_len, dtype=jnp.int32)[:, None]
    key_ids = jnp.arange(kv_seq_len, dtype=jnp.int32)[None, :]
    distances = query_ids - key_ids
    relative_extent = relative_logits.shape[-1]
    safe_distances = jnp.clip(distances, 0, relative_extent - 1)
    batch_ids = jnp.arange(batch_size)[:, None, None, None]
    head_ids = jnp.arange(num_heads)[None, :, None, None]
    relative_query_ids = jnp.arange(q_seq_len)[None, None, :, None]
    relative_bias = relative_logits[batch_ids, head_ids, relative_query_ids, safe_distances[None, None, :, :]]
    relative_valid = (distances >= 0) & (distances < relative_extent)
    logits += jnp.where(relative_valid[None, None, :, :], relative_bias, 0.0)

    mask = jnp.ones((batch_size, 1, q_seq_len, kv_seq_len), dtype=jnp.bool_)
    if segment_ids is not None:
        mask &= segment_ids.q[:, None, :, None] == segment_ids.kv[:, None, None, :]
    if causal:
        mask &= distances[None, None, :, :] >= 0
    if sliding_window is not None:
        mask &= distances[None, None, :, :] < sliding_window

    logits += jnp.where(mask, 0.0, mask_value)
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v, preferred_element_type=jnp.float32).astype(v.dtype)


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
    if block > dim:
        raise ValueError(f"{block_name}={block} should be smaller or equal to {dim_name}={dim}")
    if should_divide and dim % block != 0:
        raise ValueError(f"{dim_name}={dim} should be divisible by {block_name}={block}")
