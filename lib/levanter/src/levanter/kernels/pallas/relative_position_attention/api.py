# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from typing import Literal

import jax
import jax.numpy as jnp
from haliax.partitioning import _get_mesh
from jax import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .pallas_tpu import (
    DEFAULT_MASK_VALUE,
    BlockSizes,
    SegmentIds,
    _flash_attention_bwd_dkv,
    _flash_attention_bwd_dq,
    _flash_attention_impl,
    mha_reference,
    relative_position_attention_pallas,
)

RelativePositionAttentionImplementation = Literal["reference", "pallas_tpu"]
_SHARD_MAP_CHECK_KWARG = "check_vma" if "check_vma" in inspect.signature(shard_map).parameters else "check_rep"
_SHARD_MAP_CHECK_KWARGS = {_SHARD_MAP_CHECK_KWARG: True}
_SHARD_MAP_NO_CHECK_KWARGS = {_SHARD_MAP_CHECK_KWARG: False}


def _named_partition_spec(x: jax.Array, *, label: str):
    sharding = jax.typeof(x).sharding
    if not isinstance(sharding, NamedSharding):
        raise TypeError(f"relative-position attention expects NamedSharding on {label}; got {sharding!r}")
    return sharding.spec


def _partition_axes(partition) -> tuple:
    if partition is None:
        return ()
    if isinstance(partition, tuple):
        return partition
    return (partition,)


def _pallas_tpu_sharded(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    relative_logits: jax.Array,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    sliding_window: int | None,
    sm_scale: float,
    block_sizes: BlockSizes | None,
    debug: bool,
) -> jax.Array:
    mesh = _get_mesh()
    if mesh is None or getattr(mesh, "empty", False):
        raise RuntimeError("TPU relative-position attention requires an explicit JAX mesh")

    q_spec = _named_partition_spec(q, label="q")
    k_spec = _named_partition_spec(k, label="k")
    v_spec = _named_partition_spec(v, label="v")
    relative_logits_spec = _named_partition_spec(relative_logits, label="relative_logits")
    for label, spec in (("q", q_spec), ("k", k_spec), ("v", v_spec)):
        if spec[2] is not None:
            raise NotImplementedError(
                f"TPU relative-position attention does not support sequence sharding; {label} has {spec}"
            )
    if relative_logits_spec[2] is not None or relative_logits_spec[3] is not None:
        raise NotImplementedError(
            "TPU relative-position attention requires unsharded query and relative-extent dimensions; "
            f"got {relative_logits_spec}"
        )
    batch_axes = _partition_axes(q_spec[0])
    for label, spec in (
        ("k", k_spec),
        ("v", v_spec),
        ("relative_logits", relative_logits_spec),
    ):
        if _partition_axes(spec[0]) != batch_axes:
            raise NotImplementedError(
                "TPU relative-position attention requires matching batch sharding; "
                f"q uses {q_spec[0]!r}, while {label} uses {spec[0]!r}"
            )

    kernel = functools.partial(
        relative_position_attention_pallas,
        causal=causal,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        debug=debug,
    )
    if segment_ids is None:

        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=(q_spec, k_spec, v_spec, relative_logits_spec),
            out_specs=q_spec,
            **_SHARD_MAP_CHECK_KWARGS,
        )
        def sharded_kernel(q, k, v, relative_logits):
            return kernel(q, k, v, relative_logits)

        # pyrefly: ignore[bad-specialization, bad-argument-count]
        return sharded_kernel(q, k, v, relative_logits)

    q_segment_spec = _named_partition_spec(segment_ids.q, label="segment_ids.q")
    kv_segment_spec = _named_partition_spec(segment_ids.kv, label="segment_ids.kv")
    for label, spec in (("segment_ids.q", q_segment_spec), ("segment_ids.kv", kv_segment_spec)):
        if _partition_axes(spec[0]) != batch_axes:
            raise NotImplementedError(
                "TPU relative-position attention requires segment ids to match batch sharding; "
                f"q uses {q_spec[0]!r}, while {label} uses {spec[0]!r}"
            )
    if block_sizes is None:
        block_sizes = BlockSizes.get_default(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[2],
            q.shape[3],
            relative_logits.shape[-1],
        )
    if not block_sizes.has_backward_blocks:
        raise ValueError("TPU relative-position attention requires backward block sizes")

    block_q_major_dkv = block_sizes.block_q_major_dkv
    block_k_major_dkv = block_sizes.block_k_major_dkv
    block_k_dkv = block_sizes.block_k_dkv
    block_q_dkv = block_sizes.block_q_dkv
    block_q_dq = block_sizes.block_q_dq
    block_k_major_dq = block_sizes.block_k_major_dq
    block_k_dq = block_sizes.block_k_dq
    assert block_q_major_dkv is not None
    assert block_k_major_dkv is not None
    assert block_k_dkv is not None
    assert block_q_dkv is not None
    assert block_q_dq is not None
    assert block_k_major_dq is not None
    assert block_k_dq is not None
    stats_spec = P(*q_spec[:3])

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(q_spec, k_spec, v_spec, relative_logits_spec, q_segment_spec, kv_segment_spec),
        out_specs=(q_spec, stats_spec, stats_spec),
        **_SHARD_MAP_NO_CHECK_KWARGS,
    )
    def sharded_forward(q, k, v, relative_logits, q_segment_ids, kv_segment_ids):
        return _flash_attention_impl(
            q,
            k,
            v,
            relative_logits,
            SegmentIds(q_segment_ids, kv_segment_ids),
            True,
            causal,
            sliding_window,
            sm_scale,
            block_sizes.block_b,
            block_sizes.block_q,
            block_sizes.block_k_major,
            block_sizes.block_k,
            debug,
            False,
        )

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            q_spec,
            k_spec,
            v_spec,
            relative_logits_spec,
            q_segment_spec,
            kv_segment_spec,
            stats_spec,
            stats_spec,
            q_spec,
            stats_spec,
        ),
        out_specs=(k_spec, v_spec),
        **_SHARD_MAP_NO_CHECK_KWARGS,
    )
    def sharded_dkv(q, k, v, relative_logits, q_segment_ids, kv_segment_ids, l, m, do, di):
        return _flash_attention_bwd_dkv(
            q,
            k,
            v,
            relative_logits,
            SegmentIds(q_segment_ids, kv_segment_ids),
            l,
            m,
            do,
            di,
            block_q_major=block_q_major_dkv,
            block_q=block_q_dkv,
            block_k_major=block_k_major_dkv,
            block_k=block_k_dkv,
            sm_scale=sm_scale,
            causal=causal,
            sliding_window=sliding_window,
            mask_value=DEFAULT_MASK_VALUE,
            debug=debug,
            interpret=False,
        )

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            q_spec,
            k_spec,
            v_spec,
            relative_logits_spec,
            q_segment_spec,
            kv_segment_spec,
            stats_spec,
            stats_spec,
            q_spec,
            stats_spec,
        ),
        out_specs=(q_spec, relative_logits_spec),
        **_SHARD_MAP_NO_CHECK_KWARGS,
    )
    def sharded_dq(q, k, v, relative_logits, q_segment_ids, kv_segment_ids, l, m, do, di):
        return _flash_attention_bwd_dq(
            q,
            k,
            v,
            relative_logits,
            SegmentIds(q_segment_ids, kv_segment_ids),
            l,
            m,
            do,
            di,
            block_q_major=block_q_dq,
            block_k_major=block_k_major_dq,
            block_k=block_k_dq,
            sm_scale=sm_scale,
            causal=causal,
            sliding_window=sliding_window,
            mask_value=DEFAULT_MASK_VALUE,
            debug=debug,
            interpret=False,
        )

    @jax.custom_vjp
    def sharded_segment_kernel(q, k, v, relative_logits, q_segment_ids, kv_segment_ids):
        # pyrefly: ignore[bad-argument-count]
        o, _, _ = sharded_forward(q, k, v, relative_logits, q_segment_ids, kv_segment_ids)
        return o

    def sharded_segment_kernel_fwd(q, k, v, relative_logits, q_segment_ids, kv_segment_ids):
        # pyrefly: ignore[bad-argument-count]
        o, l, m = sharded_forward(q, k, v, relative_logits, q_segment_ids, kv_segment_ids)
        return o, (q, k, v, relative_logits, q_segment_ids, kv_segment_ids, o, l, m)

    def sharded_segment_kernel_bwd(residuals, do):
        q, k, v, relative_logits, q_segment_ids, kv_segment_ids, o, l, m = residuals
        di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)
        # pyrefly: ignore[bad-argument-count]
        dk, dv = sharded_dkv(q, k, v, relative_logits, q_segment_ids, kv_segment_ids, l, m, do, di)
        # pyrefly: ignore[bad-argument-count]
        dq, drelative_logits = sharded_dq(q, k, v, relative_logits, q_segment_ids, kv_segment_ids, l, m, do, di)
        return dq, dk, dv, drelative_logits, None, None

    sharded_segment_kernel.defvjp(sharded_segment_kernel_fwd, sharded_segment_kernel_bwd)

    # pyrefly: ignore[bad-specialization, bad-argument-count]
    return sharded_segment_kernel(q, k, v, relative_logits, segment_ids.q, segment_ids.kv)


def relative_position_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    relative_logits: jax.Array,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool,
    sliding_window: int | None,
    sm_scale: float,
    implementation: RelativePositionAttentionImplementation | None = None,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Apply BHSD attention with a compact ``[B, H, Q, E]`` relative-bias band.

    Index ``d`` on the last axis supplies the bias for backward relative
    distance ``d = query - key``, matching Inkling's released implementation.
    """
    if implementation == "reference":
        return mha_reference(
            q,
            k,
            v,
            relative_logits,
            segment_ids,
            causal=causal,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
        )
    if implementation not in (None, "pallas_tpu"):
        raise ValueError(f"Unknown relative-position attention implementation: {implementation}")

    # The TPU kernel stores distances in reverse order so each score tile can
    # load a contiguous band. Keep that layout private to the kernel boundary.
    stored_relative_logits = jnp.flip(relative_logits, axis=-1)
    if interpret:
        return relative_position_attention_pallas(
            q,
            k,
            v,
            stored_relative_logits,
            segment_ids,
            causal=causal,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            debug=debug,
            interpret=True,
        )
    if jax.default_backend() == "tpu":
        return _pallas_tpu_sharded(
            q,
            k,
            v,
            stored_relative_logits,
            segment_ids,
            causal=causal,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            debug=debug,
        )
    if implementation == "pallas_tpu":
        raise RuntimeError("pallas_tpu relative-position attention requires TPU or interpret=True")
    return mha_reference(
        q,
        k,
        v,
        relative_logits,
        segment_ids,
        causal=causal,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
    )
