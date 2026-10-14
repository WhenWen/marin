# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental MoE transport after unchanged logical capacity clipping."""

from dataclasses import replace

import jax
import jax.numpy as jnp
from levanter.grug._moe import ep_ragged_all_to_all as ragged

_routed_experts = ragged._routed_experts


def pack_bf16_pairs(value):
    """Represent adjacent BF16 channels as one uint32 without arithmetic."""
    assert value.dtype == jnp.bfloat16 and value.shape[-1] % 2 == 0
    return jax.lax.bitcast_convert_type(value.reshape(*value.shape[:-1], value.shape[-1] // 2, 2), jnp.uint32)


def unpack_bf16_pairs(value):
    return jax.lax.bitcast_convert_type(value, jnp.bfloat16).reshape(*value.shape[:-1], value.shape[-1] * 2)


def uint32_transport(operand, output, *metadata, axis_name):
    """Keep accepted row intervals while transporting raw BF16 pairs as uint32."""
    if operand.dtype != jnp.bfloat16:
        return jax.lax.ragged_all_to_all(operand, output, *metadata, axis_name=axis_name)
    result = jax.lax.ragged_all_to_all(pack_bf16_pairs(operand), pack_bf16_pairs(output), *metadata, axis_name=axis_name)
    return unpack_bf16_pairs(result)


def pack_return_buffers(vector, scalar):
    """Append an FP32 scalar's bits to each BF16 row without numerical conversion."""
    assert vector.dtype == jnp.bfloat16 and scalar.dtype == jnp.float32
    assert scalar.shape == (vector.shape[0], 1)
    # Raw FP32 halves can be BF16 NaNs whose payloads get canonicalized. Encode
    # each byte in a normal BF16 carrier, then recover bytes without FP arithmetic.
    lanes = jax.lax.bitcast_convert_type(scalar[:, 0], jnp.uint8).astype(jnp.uint16) | jnp.uint16(0x3F00)
    vector_bits = jax.lax.bitcast_convert_type(vector, jnp.uint16)
    packed = jnp.concatenate((vector_bits, lanes), axis=-1)
    packed = jnp.pad(packed, ((0, 0), (0, -packed.shape[-1] % 128)))
    return jax.lax.bitcast_convert_type(packed, jnp.bfloat16)


def unpack_return_buffers(packed, width):
    bits = jax.lax.bitcast_convert_type(packed, jnp.uint16)
    scalar_bytes = (bits[:, width : width + 4] & jnp.uint16(0xFF)).astype(jnp.uint8)
    scalar = jax.lax.bitcast_convert_type(scalar_bytes, jnp.float32)
    vector = jax.lax.bitcast_convert_type(bits[:, :width], jnp.bfloat16)
    return vector, scalar[:, None]


def paired_return_transport(vector, vector_output, scalar, scalar_output, *metadata, axis_name):
    """Return two already-computed gradient buffers through one bit-preserving collective."""
    packed = pack_return_buffers(vector, scalar)
    output = pack_return_buffers(vector_output, scalar_output)
    returned = jax.lax.ragged_all_to_all(packed, output, *metadata, axis_name=axis_name)
    return unpack_return_buffers(returned, vector.shape[-1])


def preshaped_transport(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name):
    """Expose the TPU BF16 transport's two-part trailing shape before lowering."""
    shape = output.shape
    if operand.dtype == jnp.bfloat16 and operand.shape[-1] % 2 == 0:
        operand = operand.reshape(operand.shape[0], 2, operand.shape[-1] // 2)
        output = output.reshape(output.shape[0], 2, output.shape[-1] // 2)
    return jax.lax.ragged_all_to_all(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name=axis_name
    ).reshape(shape)


def coalesce_routing(routing: ragged._ExpertRouting, layout: ragged._ExpertLayout):
    """Combine already-clipped expert groups into one physical receiver buffer.

    Logical chunks own disjoint experts. Summing their clipped counts keeps
    every accepted sender/expert prefix while changing only receiver offsets
    and trailing padding. The combined buffer retains their summed capacity.
    """
    assert len(routing.chunk_clipped_group_sizes) == layout.chunks
    return routing._replace(chunk_clipped_group_sizes=(sum(routing.chunk_clipped_group_sizes),)), replace(
        layout, chunk_experts=layout.local_experts, chunk_capacity=layout.chunk_capacity * layout.chunks
    )


def coalesced_routed_experts(sorted_x, weights, moe_w13_local, moe_w2_local, routing, layout):
    routing, layout = coalesce_routing(routing, layout)
    return _routed_experts(sorted_x, weights, moe_w13_local, moe_w2_local, routing, layout)
