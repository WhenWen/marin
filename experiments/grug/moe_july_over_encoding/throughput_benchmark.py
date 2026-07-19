# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched d512 throughput gate for table-sharded Over-Encoding.

Both cells use this branch, the same canonical July model/data/optimizer setup,
and the same v5p-8 topology. The only model difference is whether hierarchical
Over-Encoding is enabled. A 50-step device profile is recorded after warmup;
the steady-state median throughput outside that interval is the gate metric.
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
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
from experiments.grug.moe_july_over_encoding.train import GrugTrainerConfig

_ISSUE_NUMBER = 7368
_SEQ_LEN = 8192
_TPU = "v5p-8"
_HIDDEN_DIM = 512
_BATCH_SIZE = 16
_NUM_STEPS = 700
_OVER_ENCODING_LR_MULTIPLIER = 0.5
_WANDB_GROUP = "MOE-OE-JULY-tablewise-throughput-issue-7368"
_BASELINE_RUN_ID = "MOE-JULY-TABLEWISE-PERF-BASELINE-d512"
_OVER_ENCODING_RUN_ID = "MOE-JULY-TABLEWISE-PERF-OE-d512"


def build_step(*, enable_over_encoding: bool) -> ExecutorStep:
    """Build one side of the matched baseline/OE throughput benchmark."""
    heuristic = MoeHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(_HIDDEN_DIM, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    if enable_over_encoding:
        model = dataclasses.replace(
            model,
            over_encoding_vocab_size=over_encoding_vocab_size(_HIDDEN_DIM, model.vocab_size),
            over_encoding_splits=4,
            over_encoding_num_grams=3,
        )

    tokens = float(_NUM_STEPS * _BATCH_SIZE * _SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(
        _BATCH_SIZE,
        tokens,
        _HIDDEN_DIM,
        seq_len=_SEQ_LEN,
    )
    if not isinstance(optimizer, GrugMoeMuonHConfig):
        raise TypeError(f"expected canonical MuonH config, got {type(optimizer)}")
    optimizer = dataclasses.replace(
        optimizer,
        over_encoding_lr_multiplier=_OVER_ENCODING_LR_MULTIPLIER,
    )

    run_id = _OVER_ENCODING_RUN_ID if enable_over_encoding else _BASELINE_RUN_ID
    variant_tag = "tablewise-oe" if enable_over_encoding else "no-oe"
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
                    "throughput-gate",
                    variant_tag,
                    f"d{_HIDDEN_DIM}",
                ],
                group=_WANDB_GROUP,
                name=None,
            ),
            optimizer=versioned(optimizer),
            profiler=ProfilerConfig(
                enabled=True,
                start_step=100,
                num_steps=50,
                profile_options=ProfileOptionsConfig(enable_hlo_proto=True),
            ),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
            eval=None,
            checkpointer=CheckpointerConfig(
                base_path="/tmp/moe-oe-tablewise-throughput-7368",
                save_interval=None,
                keep=None,
            ),
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[build_step(enable_over_encoding=False), build_step(enable_over_encoding=True)],
        description="Matched canonical July d512 baseline/OE throughput gate for table-sharded Over-Encoding.",
        max_concurrent=2,
    )
