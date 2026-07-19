# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build throughput-qualified two-RMSNorm Over-Encoding scaling cells."""

import dataclasses
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe_july_over_encoding.heuristic import MoeHeuristic
from experiments.grug.moe_july_over_encoding.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_july_over_encoding.lr_sweep import over_encoding_vocab_size
from experiments.grug.moe_july_over_encoding.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_july_over_encoding.train import GrugEvalConfig, GrugTrainerConfig

_ISSUE_NUMBER = 7368
_SEQ_LEN = 8192
_TPU = "v5p-8"
_OVER_ENCODING_LR_MULTIPLIER = 0.5
_D768_WANDB_GROUP = "MOE-OE-JULY-normsum-sc-gate2-issue-7368"


@dataclass(frozen=True)
class Gate2Point:
    """One canonical July scaling point for the two-RMSNorm OE variant."""

    hidden_dim: int
    batch_size: int
    num_steps: int


D768_POINT = Gate2Point(hidden_dim=768, batch_size=32, num_steps=16_875)


def build_gate2_step(point: Gate2Point, *, wandb_group: str) -> ExecutorStep:
    """Build one canonical July two-RMSNorm Gate 2 cell."""
    run_id = f"MOE-OE-JULY-NORMSUM-SC-GATE2-d{point.hidden_dim}"

    heuristic = MoeHeuristic()
    july_model = dataclasses.replace(
        heuristic.build_model_config(point.hidden_dim, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    model = dataclasses.replace(
        july_model,
        over_encoding_vocab_size=over_encoding_vocab_size(point.hidden_dim, july_model.vocab_size),
        over_encoding_splits=4,
        over_encoding_num_grams=3,
    )

    tokens = float(point.num_steps * point.batch_size * _SEQ_LEN)
    july_optimizer = heuristic.build_optimizer_config(
        point.batch_size,
        tokens,
        point.hidden_dim,
        seq_len=_SEQ_LEN,
    )
    if not isinstance(july_optimizer, GrugMoeMuonHConfig):
        raise TypeError(f"expected canonical MuonH config, got {type(july_optimizer)}")
    optimizer = dataclasses.replace(
        july_optimizer,
        over_encoding_lr_multiplier=_OVER_ENCODING_LR_MULTIPLIER,
    )

    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(point.num_steps),
            batch_size=versioned(point.batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=[
                    "MOE-OE-JULY",
                    f"issue-{_ISSUE_NUMBER}",
                    "normalized-input-streams",
                    "sparsecore-gradient",
                    "gate-2",
                    "oe-lr-0.5x",
                    f"d{point.hidden_dim}",
                ],
                group=wandb_group,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
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
    )


def build_step() -> ExecutorStep:
    """Build the existing canonical July d768 intermediate scaling cell."""
    return build_gate2_step(D768_POINT, wandb_group=_D768_WANDB_GROUP)


if __name__ == "__main__":
    executor_main(
        steps=[build_step()],
        description="Throughput-qualified July d768 two-RMSNorm Over-Encoding Gate 2.",
        max_concurrent=1,
    )
