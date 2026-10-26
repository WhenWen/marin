# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned execution choices from the Ckpt 1 full-update comparisons."""

import functools

import jax
from levanter.grug._moe import ep_ragged_all_to_all

from experiments.grug.belay_tpu import local_experts, model, routing_tpu
from experiments.grug.belay_tpu.capacity_layout import factorized_route_counts, interval_route_layout
from experiments.grug.belay_tpu.expert_collectives import gather_expert_weights_with_owner_sum
from experiments.grug.belay_tpu.expert_gradient_adapter import partitioned_grouped_dot
from experiments.grug.belay_tpu.kda_tpu import chunk_kda_tpu
from experiments.grug.belay_tpu.local_experts import gather_expert_weights
from experiments.grug.belay_tpu.streaming_combine import combine_streaming, dispatch_streaming

SORTED_SCALAR_PROFILES = ("kda-owner-attention-sort-v1", "kda-owner-causal-sort-v1")
OWNER_ATTENTION_PROFILES = ("kda-owner-attention-v1", *SORTED_SCALAR_PROFILES)
KERNEL_PROFILES = ("source", "capacity-kda-v1", *OWNER_ATTENTION_PROFILES)


def kda_options(profile: str) -> dict:
    if profile not in KERNEL_PROFILES:
        raise ValueError(f"Unknown TPU kernel profile: {profile}")
    if profile == "source":
        return {}
    return {
        "state_precision": jax.lax.Precision.HIGH,
        "chunk_size": 64,
        "solve_groups": 4,
        "fused_prep": True,
        "fused_recurrence": True,
        "state_head_group": 4,
    }


def apply_tpu_kernel_profile(profile: str) -> None:
    """Install the selected process-local operators before model tracing."""
    options = kda_options(profile)
    if profile == "source":
        return

    model.chunk_kda_tpu = functools.partial(chunk_kda_tpu, **options)
    local_experts.route_counts = factorized_route_counts
    local_experts.route_layout = interval_route_layout
    local_experts.grouped_dot = functools.partial(partitioned_grouped_dot, row_partitions=2)
    local_experts.combine_local = functools.partial(
        combine_streaming,
        implementation="scan",
        scalar_permutation="sort" if profile in SORTED_SCALAR_PROFILES else "gather",
    )
    local_experts.dispatch_local = dispatch_streaming
    local_experts.gather_expert_weights = (
        gather_expert_weights_with_owner_sum if profile in OWNER_ATTENTION_PROFILES else gather_expert_weights
    )
    model._selected_router_logits = routing_tpu.unrolled_router_logits
    ep_ragged_all_to_all._accepted_assignments = routing_tpu.accepted_assignments_cutoff
    ep_ragged_all_to_all._restore_assignment_scalars = routing_tpu.restore_assignment_scalars_sort
    ep_ragged_all_to_all._permute_assignment_weights = routing_tpu.permute_assignment_weights_sort
