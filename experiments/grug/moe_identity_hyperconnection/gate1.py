# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate 1 for four-stream Identity HC on the real July baseline."""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe_identity_hyperconnection.heuristic import MoeHeuristic
from experiments.grug.moe_identity_hyperconnection.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_identity_hyperconnection.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN = 8192
_TPU = "v5p-8"
_GROUP = "MOE-JULY-IHC-gate1-issue-7409"
_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 16, 10_980),
    (768, 32, 16_875),
)


def _build_step(hidden_dim: int, batch_size: int, num_steps: int, cell_id: int) -> ExecutorStep:
    heuristic = MoeHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(hidden_dim, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
        num_residual_streams=4,
        hyperconnection_alpha_init=0.01,
        hyperconnection_remat_layers=2,
    )
    tokens = float(num_steps * batch_size * _SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)
    run_id = f"MOE-JULY-IHC-G1-{cell_id:03d}-d{hidden_dim}"
    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=[
                    "moe",
                    "july_baseline",
                    "identity_hyperconnection",
                    "identity_hc_gate1",
                    f"d{hidden_dim}",
                    "issue_7409",
                ],
                group=_GROUP,
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
    steps = [_build_step(dim, batch, num_steps, index) for index, (dim, batch, num_steps) in enumerate(_POINTS, 1)]
    executor_main(
        steps=steps,
        description="Gate 1: July baseline plus four-stream Identity Hyper-Connections at d512 and d768.",
    )
