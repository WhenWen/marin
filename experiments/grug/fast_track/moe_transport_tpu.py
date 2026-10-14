# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental physical MoE batching after unchanged logical capacity clipping."""

from dataclasses import replace

from levanter.grug._moe import ep_ragged_all_to_all as ragged

_routed_experts = ragged._routed_experts


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
