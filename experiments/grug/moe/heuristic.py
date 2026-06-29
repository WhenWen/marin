# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the AdamH MoE scaling heuristic."""

from experiments.grug.moe import heuristic_v1

DEFAULT_TARGET_STEPS = heuristic_v1.DEFAULT_TARGET_STEPS
MIN_BATCH_SIZE = heuristic_v1.MIN_BATCH_SIZE
SEQ_LEN = heuristic_v1.SEQ_LEN
MoeAdamHHeuristic = heuristic_v1.MoeAdamHHeuristic
build_from_heuristic = heuristic_v1.build_from_heuristic
compute_flops_per_token = heuristic_v1.compute_flops_per_token
compute_tokens_and_batch = heuristic_v1.compute_tokens_and_batch
moe_adamh_heuristic = heuristic_v1.moe_adamh_heuristic
