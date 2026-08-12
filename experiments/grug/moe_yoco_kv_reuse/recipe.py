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
