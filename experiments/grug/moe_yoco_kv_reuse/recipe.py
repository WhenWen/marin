# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-July recipes for the parameter-preserving midpoint K/V reuse experiment."""

import dataclasses
from dataclasses import dataclass

from experiments.grug.moe.heuristic import MoeHeuristic as BaselineMoeHeuristic
from experiments.grug.moe.model import GrugModelConfig as BaselineModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig as BaselineOptimizerConfig
from experiments.grug.moe_yoco_kv_reuse.heuristic import MoeHeuristic
from experiments.grug.moe_yoco_kv_reuse.model import GrugModelConfig
from experiments.grug.moe_yoco_kv_reuse.optimizer import GrugMoeMuonHConfig

SEQ_LEN: int = 8192


@dataclass(frozen=True)
class ExperimentPoint:
    hidden_dim: int
    batch_size: int
    num_steps: int
    compute_budget: float


POINTS: tuple[ExperimentPoint, ...] = (
    ExperimentPoint(hidden_dim=512, batch_size=16, num_steps=10_980, compute_budget=3.82e17),
    ExperimentPoint(hidden_dim=768, batch_size=32, num_steps=16_875, compute_budget=2.81e18),
    ExperimentPoint(hidden_dim=1024, batch_size=64, num_steps=16_080, compute_budget=1.16e19),
    ExperimentPoint(hidden_dim=1280, batch_size=128, num_steps=14_325, compute_budget=3.46e19),
)

# The current MoE overtraining baseline uses 750 tokens per active parameter
# (issue #8062). The exact-July d512 model has 20,730,368 active parameters
# excluding the embedding and LM head, so this rounds to 118,620 full batches.
OVERTRAIN_TOKENS_PER_ACTIVE_PARAMETER: int = 750
OVERTRAIN_D512_ACTIVE_PARAMETERS: int = 20_730_368
OVERTRAIN_D512_750_TPP = ExperimentPoint(
    hidden_dim=512,
    batch_size=16,
    num_steps=118_620,
    compute_budget=6.48e18,
)


def point_for_hidden_dim(hidden_dim: int) -> ExperimentPoint:
    matches = [point for point in POINTS if point.hidden_dim == hidden_dim]
    if len(matches) != 1:
        raise ValueError(f"Expected one experiment point for hidden_dim={hidden_dim}, found {len(matches)}")
    return matches[0]


def with_midpoint_kv_reuse(model: GrugModelConfig) -> GrugModelConfig:
    """Reuse the last first-half layer output for K/V throughout the second half."""
    if model.num_layers < 2:
        raise ValueError(f"Midpoint K/V reuse requires at least two layers, got {model.num_layers}")
    reuse_start_layer = (model.num_layers + 1) // 2
    return dataclasses.replace(model, kv_reuse_start_layer=reuse_start_layer)


def with_classical_yoco(model: GrugModelConfig) -> GrugModelConfig:
    """Share one projected midpoint K/V pair across all second-half layers."""
    if model.num_layers < 2:
        raise ValueError(f"Classical YOCO requires at least two layers, got {model.num_layers}")
    reuse_start_layer = (model.num_layers + 1) // 2
    return dataclasses.replace(
        model,
        kv_reuse_start_layer=None,
        shared_projected_kv_start_layer=reuse_start_layer,
    )


def classical_yoco_recipe(
    point: ExperimentPoint,
    *,
    parameter_match: str | None = None,
) -> tuple[GrugModelConfig, GrugMoeMuonHConfig]:
    """Return classical YOCO with optional localized parameter reinvestment."""
    heuristic = MoeHeuristic()
    model = with_classical_yoco(
        dataclasses.replace(
            heuristic.build_model_config(point.hidden_dim, seq_len=SEQ_LEN),
            disable_pko=True,
            disable_long_rope=True,
        )
    )
    start_layer = model.shared_projected_kv_start_layer
    assert start_layer is not None
    removed_projection_layers = model.num_layers - start_layer - 1
    if parameter_match == "expert":
        head_dim = point.hidden_dim // model.num_heads
        removed_parameters = removed_projection_layers * 2 * point.hidden_dim * model.num_kv_heads * head_dim
        shared_expert_parameters_per_dim = 3 * point.hidden_dim
        expert_intermediate_dim = round(removed_parameters / shared_expert_parameters_per_dim)
        model = dataclasses.replace(
            model,
            additional_shared_expert_layer=start_layer,
            additional_shared_expert_intermediate_dim=expert_intermediate_dim,
        )
    elif parameter_match == "heads":
        model = dataclasses.replace(
            model,
            additional_query_head_layers=tuple(range(start_layer, model.num_layers - 1)),
        )
    elif parameter_match is not None:
        raise ValueError(f"Unknown parameter_match={parameter_match!r}")

    tokens = float(point.num_steps * point.batch_size * SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(
        point.batch_size,
        tokens,
        point.hidden_dim,
        seq_len=SEQ_LEN,
    )
    return model, optimizer


def variant_recipe(point: ExperimentPoint) -> tuple[GrugModelConfig, GrugMoeMuonHConfig]:
    """Return a midpoint K/V reuse recipe at an exact July compute-optimal cell."""
    heuristic = MoeHeuristic()
    model = with_midpoint_kv_reuse(
        dataclasses.replace(
            heuristic.build_model_config(point.hidden_dim, seq_len=SEQ_LEN),
            disable_pko=True,
            disable_long_rope=True,
        )
    )
    tokens = float(point.num_steps * point.batch_size * SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(
        point.batch_size,
        tokens,
        point.hidden_dim,
        seq_len=SEQ_LEN,
    )
    return model, optimizer


def baseline_recipe(point: ExperimentPoint) -> tuple[BaselineModelConfig, BaselineOptimizerConfig]:
    """Return the unmodified July recipe for comparison and contract tests."""
    heuristic = BaselineMoeHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(point.hidden_dim, seq_len=SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    tokens = float(point.num_steps * point.batch_size * SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(
        point.batch_size,
        tokens,
        point.hidden_dim,
        seq_len=SEQ_LEN,
    )
    return model, optimizer
