# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental fixed all-to-all transport for already accepted ragged rows."""

import math

import jax
import jax.numpy as jnp
from levanter.grug._moe.ep_common import _assignment_sources


def peer_capacity(input_rows, output_rows, peers):
    return 128 * math.ceil(2 * min(input_rows, output_rows) / (128 * peers))


def packed_row_indices(offsets, sizes, capacity, sentinel):
    """Map peer-major packed positions to original rows, with an invalid sentinel."""
    ends = jnp.cumsum(sizes, axis=-1)
    positions = jnp.arange(capacity, dtype=jnp.int32)
    groups = jax.vmap(lambda row: jnp.searchsorted(row, positions, side="right"))(ends)
    groups = jnp.minimum(groups, sizes.shape[-1] - 1)
    starts = ends - sizes
    original = jnp.take_along_axis(offsets - starts, groups, axis=-1) + positions[None, :]
    return jnp.where(positions[None, :] < ends[:, -1, None], original, sentinel)


def overflowed(send_sizes, recv_sizes, *, peers, capacity, axis_name):
    sent = send_sizes.reshape(peers, -1).sum(axis=-1)
    received = recv_sizes.reshape(peers, -1).sum(axis=-1)
    maximum = jax.lax.pmax(jnp.maximum(sent.max(), received.max()), axis_name)
    return maximum > capacity


def packed_transport(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name):
    """Transport accepted intervals without changing their destination positions.

    The caller must establish that all peer totals fit the padded capacity.
    Output offsets are sender-side remote offsets and must be exchanged before
    constructing each receiver's inverse map.
    """
    peers = jax.lax.axis_size(axis_name)
    capacity = peer_capacity(operand.shape[0], output.shape[0], peers)
    sources = packed_row_indices(
        input_offsets.reshape(peers, -1), send_sizes.reshape(peers, -1), capacity, operand.shape[0]
    )
    packed = jnp.where(
        (sources < operand.shape[0])[..., None], operand[jnp.minimum(sources, operand.shape[0] - 1)], 0
    )
    remote_offsets = jax.lax.all_to_all(
        output_offsets.reshape(peers, -1), axis_name, split_axis=0, concat_axis=0, tiled=True
    )
    received = jax.lax.all_to_all(packed, axis_name, split_axis=0, concat_axis=0, tiled=True)
    destinations = packed_row_indices(
        remote_offsets, recv_sizes.reshape(peers, -1), capacity, output.shape[0]
    ).reshape(-1)
    inverse = _assignment_sources(destinations, send_size=output.shape[0])
    received = received.reshape(-1, operand.shape[-1])
    return jnp.where(
        (inverse < received.shape[0])[:, None], received[jnp.minimum(inverse, received.shape[0] - 1)], output
    )


def padded_transport(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name):
    """Use packed all-to-all when every peer fits; otherwise preserve ragged transport."""
    peers = jax.lax.axis_size(axis_name)
    capacity = peer_capacity(operand.shape[0], output.shape[0], peers)
    overflow = overflowed(send_sizes, recv_sizes, peers=peers, capacity=capacity, axis_name=axis_name)
    operands = operand, output, input_offsets, send_sizes, output_offsets, recv_sizes

    def fallback(values):
        return jax.lax.ragged_all_to_all(*values, axis_name=axis_name)

    def packed(values):
        return packed_transport(*values, axis_name=axis_name)

    return jax.lax.cond(overflow, fallback, packed, operands), overflow
