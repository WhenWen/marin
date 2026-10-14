# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in execution bindings for the qualified experimental TPU combination.

Bindings apply for the complete single-run process, including train/eval tracing.
The shared model and library defaults remain unchanged outside this context.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import partial
from unittest.mock import patch

import jax
from levanter.grug._moe import ep_ragged_all_to_all as ragged

from experiments.grug.fast_track import belay_tpu_ops, model, qb_balancing
from experiments.grug.fast_track.moe_saved_output import saved_output_routed_experts
from experiments.grug.fast_track.router_lookup import unrolled_router_logits
from experiments.grug.moe import kda

QUALIFICATION = "https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6176472787"
SELECTED = {
    "layer_backward": "save",
    "attention_remat": "off",
    "kda_state": "pallas",
    "kda_state_precision": "high",
    "projection_weights": "gather",
    "accepted_mask": "cutoff",
    "moe_scalar_permutation": "sort",
    "moe_weight_permutation": "sort",
    "qb_quantile": "two-stage",
    "router_lookup": "unrolled",
    "cross_entropy": "pallas512",
    "second_embedding": "column",
    "moe_weight_gradient": "saved-output",
    "moe_dispatch": "serialized",
    "expert_backward": "recompute",
}


def selected_model_config(config: model.GrugModelConfig) -> model.GrugModelConfig:
    return replace(config, attn_res_layer_backward=model.AttnResLayerBackward.SAVE, attn_res_remat_attention=False)


@contextmanager
def selected_execution():
    """Bind the selected implementation for one isolated experimental run."""
    with ExitStack() as stack:
        for module, name, implementation in (
            (model, "_proj", belay_tpu_ops.gathered_projection),
            (model, "_replicated_fp32_gather", belay_tpu_ops.column_embedding_gather),
            (model, "fused_linear_softmax_cross_entropy_loss", belay_tpu_ops.pallas_cross_entropy),
            (model, "_selected_router_logits", unrolled_router_logits),
            (kda, "_parallel_chunk_recurrence", partial(belay_tpu_ops.tpu_recurrence, precision=jax.lax.Precision.HIGH)),
            (ragged, "_accepted_assignments", ragged._accepted_assignments_cutoff),
            (ragged, "_restore_assignment_scalars", ragged._restore_assignment_scalars_sort),
            (ragged, "_permute_assignment_weights", ragged._permute_assignment_weights_sort),
            (ragged, "_routed_experts", saved_output_routed_experts),
            (
                qb_balancing,
                "bincount_upper_quantile",
                partial(qb_balancing.bincount_upper_quantile, histogram_method="two-stage"),
            ),
        ):
            stack.enter_context(patch.object(module, name, implementation))
        yield
