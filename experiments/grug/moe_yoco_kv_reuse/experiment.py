# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-July d512/d768 midpoint K/V reuse experiments.

Use ``--run_only '["grug/moe_yoco_kv_reuse_july_d512"]'`` for the first gate cell.
The d768 cell is defined by the same depth-derived recipe and can be selected
after d512 passes.
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe_yoco_kv_reuse.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_yoco_kv_reuse.recipe import POINTS, ExperimentPoint, variant_recipe
from experiments.grug.moe_yoco_kv_reuse.train import GrugEvalConfig, GrugTrainerConfig

_TPU: str = "v5p-8"
_WANDB_GROUP: str = "MOE-YOCO-KV-july-issue-8196"


def build_step(point: ExperimentPoint) -> ExecutorStep:
    model, optimizer = variant_recipe(point)
    run_id = f"MOE-YOCO-KV-JULY-001-d{point.hidden_dim}"
    return ExecutorStep(
        name=f"grug/moe_yoco_kv_reuse_july_d{point.hidden_dim}",
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
                tags=["MOE-YOCO-KV", "issue-8196", "july-baseline", f"d{point.hidden_dim}"],
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
        steps=[build_step(point) for point in POINTS],
        description=(
            "Parameter-preserving midpoint K/V reuse on the exact July d512/d768 MoE recipes. "
            "Select one gate cell with --run_only '[\"<step-regex>\"]'."
        ),
    )
