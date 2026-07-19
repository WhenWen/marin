# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test independently normalized token and OE streams on canonical July d512.

The cell keeps the selected fixed-C Over-Encoding configuration and 0.5x table
learning rate. The only model change is the input-stream fusion:

    (RMSNorm(token_embedding) + RMSNorm(over_encoding)) / sqrt(2)
"""

import dataclasses

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
_HIDDEN_DIM = 512
_BATCH_SIZE = 16
_NUM_STEPS = 10_980
_OVER_ENCODING_LR_MULTIPLIER = 0.5
_RUN_ID = "MOE-OE-JULY-NORMSUM-LR050-d512"
_WANDB_GROUP = "MOE-OE-JULY-normsum-lr050-issue-7368"


def build_step() -> ExecutorStep:
    """Build the normalized-input-stream d512 OE comparison cell."""
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
        over_encoding_lr_multiplier=_OVER_ENCODING_LR_MULTIPLIER,
    )

    return ExecutorStep(
        name=f"grug/{_RUN_ID}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=_RUN_ID,
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
                    "normalized-input-streams",
                    "oe-lr-0.5x",
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
        steps=[build_step()],
        description="Canonical July d512 OE with independently RMS-normalized token and OE input streams.",
        max_concurrent=1,
    )
