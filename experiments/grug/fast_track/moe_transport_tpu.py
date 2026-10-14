# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental MoE transport after unchanged logical capacity clipping."""

from dataclasses import replace

import jax
import jax.numpy as jnp
from levanter.grug._moe import ep_ragged_all_to_all as ragged

_routed_experts = ragged._routed_experts


def pack_return_buffers(vector, scalar):
    """Append an FP32 scalar's bits to each BF16 row without numerical conversion."""
    assert vector.dtype == jnp.bfloat16 and scalar.dtype == jnp.float32
    assert scalar.shape == (vector.shape[0], 1)
    lanes = jax.lax.bitcast_convert_type(scalar[:, 0], jnp.bfloat16)
    packed = jnp.concatenate((vector, lanes), axis=-1)
    return jnp.pad(packed, ((0, 0), (0, -packed.shape[-1] % 128)))


def unpack_return_buffers(packed, width):
    scalar = jax.lax.bitcast_convert_type(packed[:, width : width + 2], jnp.float32)
    return packed[:, :width], scalar[:, None]


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
