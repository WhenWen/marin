# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tune only the added Over-Encoding table LR on canonical July Baseline d512.

The branch base is the exact July Baseline commit used by #6882. Every cell
keeps the canonical model, data, optimizer schedule, batch, seed, and training
budget fixed. The only sweep dimension is the Adam learning-rate multiplier
for ``over_encoding.tables``; the base token embedding remains on the canonical
Adam schedule and the OE projections remain on canonical MuonH.
"""

import dataclasses
import math
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
from experiments.grug.moe_july_over_encoding.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_july_over_encoding.train import GrugEvalConfig, GrugTrainerConfig

_ISSUE_NUMBER = 7368
_SEQ_LEN = 8192
_TPU = "v5p-8"
_HIDDEN_DIM = 512
_BATCH_SIZE = 16
_NUM_STEPS = 10_980
_WANDB_GROUP = "MOE-OE-JULY-lr-sweep-issue-7368"
_OE_REFERENCE_WIDTH = 1280
_OE_REFERENCE_VOCAB_SIZE = 6_400_000
OVER_ENCODING_SCALE_CONSTANT = _OE_REFERENCE_VOCAB_SIZE / (_OE_REFERENCE_WIDTH**1.5)


@dataclass(frozen=True)
class LrSweepPoint:
    """One OE-table learning-rate multiplier at fixed model/data/training config."""

    experiment_id: str
    over_encoding_lr_multiplier: float


LR_SWEEP_POINTS: tuple[LrSweepPoint, ...] = (
    LrSweepPoint("MOE-OE-JULY-LR025", 0.25),
    LrSweepPoint("MOE-OE-JULY-LR050", 0.5),
    LrSweepPoint("MOE-OE-JULY-LR100", 1.0),
    LrSweepPoint("MOE-OE-JULY-LR200", 2.0),
    LrSweepPoint("MOE-OE-JULY-LR400", 4.0),
)


def over_encoding_vocab_size(hidden_dim: int, base_vocab_size: int) -> int:
    """Return the nearest base-vocabulary-coprime integer to C * width^1.5."""
    raw_vocab_size = OVER_ENCODING_SCALE_CONSTANT * hidden_dim**1.5
    rounded_vocab_size = round(raw_vocab_size)
    for offset in range(rounded_vocab_size):
        lower = rounded_vocab_size - offset
        if lower > 0 and math.gcd(lower, base_vocab_size) == 1:
            return lower
        upper = rounded_vocab_size + offset
        if math.gcd(upper, base_vocab_size) == 1:
            return upper
    raise AssertionError("failed to find a positive coprime Over-Encoding vocabulary size")


def _build_step(point: LrSweepPoint) -> ExecutorStep:
    heuristic = MoeHeuristic()
    july_model = dataclasses.replace(
        heuristic.build_model_config(_HIDDEN_DIM, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    model = dataclasses.replace(
        july_model,
        over_encoding_vocab_size=over_encoding_vocab_size(_HIDDEN_DIM, july_model.vocab_size),
        over_encoding_splits=4,
        over_encoding_num_grams=3,
    )
    tokens = float(_NUM_STEPS * _BATCH_SIZE * _SEQ_LEN)
    july_optimizer = heuristic.build_optimizer_config(
        _BATCH_SIZE,
        tokens,
        _HIDDEN_DIM,
        seq_len=_SEQ_LEN,
    )
    if not isinstance(july_optimizer, GrugMoeMuonHConfig):
        raise TypeError(f"expected canonical MuonH config, got {type(july_optimizer)}")
    optimizer = dataclasses.replace(
        july_optimizer,
        over_encoding_lr_multiplier=point.over_encoding_lr_multiplier,
    )

    run_id = f"{point.experiment_id}-d{_HIDDEN_DIM}"
    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(_NUM_STEPS),
            batch_size=versioned(_BATCH_SIZE),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=[
                    "MOE-OE-JULY",
                    f"issue-{_ISSUE_NUMBER}",
                    "lr-sweep",
                    f"oe-lr-{point.over_encoding_lr_multiplier:g}x",
                    f"d{_HIDDEN_DIM}",
                ],
                group=_WANDB_GROUP,
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


if __name__ == "__main__":
    executor_main(
        steps=[_build_step(point) for point in LR_SWEEP_POINTS],
        description=(
            "Canonical July Baseline d512 with fixed Over-Encoding C and an OE-table-only learning-rate sweep."
        ),
        max_concurrent=len(LR_SWEEP_POINTS),
    )
