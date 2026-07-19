# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate 1 cells for the July-baseline half-RoPE plus relative-attention model."""

import dataclasses
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe_rope_relative_position.heuristic import MoeHeuristic
from experiments.grug.moe_rope_relative_position.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_rope_relative_position.model import (
    AttentionQKNormalization,
    GrugModelConfig,
    RelativePositionEmbeddingInit,
    RelativePositionEmbeddingOrder,
    RelativeQueryProjectionInit,
)
from experiments.grug.moe_rope_relative_position.optimizer import (
    GrugMoeMuonHConfig,
    RelativeQueryProjectionOptimizer,
)
from experiments.grug.moe_rope_relative_position.train import GrugEvalConfig, GrugTrainerConfig


@dataclass(frozen=True)
class Gate1Point:
    """One fixed-width Gate 1 comparison cell."""

    experiment_id: str
    hidden_dim: int
    budget: float
    batch_size: int
    num_steps: int
    july_baseline_macro_loss: float
    july_baseline_tokens_per_second: float


_SEQ_LEN = 8192
_ISSUE_NUMBER = 7208
_WANDB_GROUP = "MOE-JULY-ROPE-RPE-INKP-gate1-issue-7208"
_GATE_1_RESOURCES = ResourceConfig.with_tpu("v5p-8")
GATE_1_POINTS: tuple[Gate1Point, ...] = (
    Gate1Point("MOE-JULY-ROPE-RPE-INKP-001", 512, 3.82e17, 16, 10_980, 3.5667, 352_609),
    Gate1Point("MOE-JULY-ROPE-RPE-INKP-002", 768, 2.81e18, 32, 16_875, 3.2272, 249_954),
)


def gate_1_recipe(point: Gate1Point) -> tuple[GrugModelConfig, GrugMoeMuonHConfig]:
    """Return the exact model and optimizer for one Gate 1 cell."""
    heuristic = MoeHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(point.hidden_dim, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
        qk_mult=128**-0.5,
        attention_qk_normalization=AttentionQKNormalization.LEARNED_RMS,
        relative_position_dim=16,
        relative_position_extent=1024,
        relative_position_initializer_std=0.02,
        relative_position_embedding_order=RelativePositionEmbeddingOrder.DIRECT,
        relative_position_embedding_init=RelativePositionEmbeddingInit.NORMAL,
        relative_query_projection_init=RelativeQueryProjectionInit.NORMAL,
        attention_block_size=128,
    )
    if model.inferred_head_dim != 128:
        raise ValueError(f"Gate 1 requires head_dim=128, got {model.inferred_head_dim}")
    tokens = float(point.num_steps * point.batch_size * _SEQ_LEN)
    optimizer = dataclasses.replace(
        heuristic.build_optimizer_config(
            point.batch_size,
            tokens,
            point.hidden_dim,
            seq_len=_SEQ_LEN,
        ),
        relative_query_projection_optimizer=RelativeQueryProjectionOptimizer.ADAM,
    )
    return model, optimizer


def gate_1_step(point: Gate1Point) -> ExecutorStep[GrugMoeLaunchConfig]:
    """Build one fresh July-baseline RoPE plus relative-attention training step."""
    model, optimizer = gate_1_recipe(point)
    run_id = f"{point.experiment_id}-d{point.hidden_dim}"
    return ExecutorStep(
        name=f"grug/moe_july_rope_relative_position_inkling_gate1_d{point.hidden_dim}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=versioned(run_id),
            resources=versioned(_GATE_1_RESOURCES),
            steps=versioned(point.num_steps),
            batch_size=versioned(point.batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=[
                    "MOE-JULY-ROPE-RPE-INKP",
                    f"issue-{_ISSUE_NUMBER}",
                    "gate1",
                    "july-baseline",
                    "half-rope",
                    "long-layer-rope-disabled",
                    "inkling-relative-position",
                    "direct-distance-order",
                    "inkling-qk-scale",
                    "learned-qk-rmsnorm",
                    "wr-adam",
                    "wr-normal-init",
                    "relative-table-adam",
                    "relative-table-normal-init",
                    f"d{point.hidden_dim}",
                ],
                group=_WANDB_GROUP,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=0.0,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=256,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
        description=(
            "Real July baseline with its half-RoPE policy preserved and "
            "Inkling-parameterized learned relative attention added."
        ),
    )


gate_1_steps = [gate_1_step(point) for point in GATE_1_POINTS]


if __name__ == "__main__":
    executor_main(
        steps=gate_1_steps,
        description=(
            "Issue #7208 Gate 1: real July baseline with half-RoPE plus "
            "Inkling-parameterized relative attention at d512 and d768."
        ),
    )
