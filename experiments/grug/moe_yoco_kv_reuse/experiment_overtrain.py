# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched d512 control and midpoint K/V reuse runs at 750 tokens per active parameter."""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe_yoco_kv_reuse.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_yoco_kv_reuse.recipe import OVERTRAIN_D512_750_TPP, variant_recipe
from experiments.grug.moe_yoco_kv_reuse.train import GrugEvalConfig, GrugTrainerConfig

_TPU: str = "v5p-8"
_TPU_REGIONS: tuple[str, ...] = ("us-central1",)
_WANDB_GROUP: str = "MOE-YOCO-KV-overtrain-750tpp-issue-8196"


def build_step(*, fixed_yoco: bool) -> ExecutorStep:
    point = OVERTRAIN_D512_750_TPP
    model, optimizer = variant_recipe(point)
    variant_name = "fixed-yoco" if fixed_yoco else "control"
    if not fixed_yoco:
        model = dataclasses.replace(model, kv_reuse_start_layer=None)
    run_id = f"MOE-YOCO-KV-OVERTRAIN-750TPP-001-{variant_name}-d512"
    return ExecutorStep(
        name=f"grug/moe_yoco_kv_reuse_overtrain_750tpp_{variant_name}_d512",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU, regions=_TPU_REGIONS)),
            steps=versioned(point.num_steps),
            batch_size=versioned(point.batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=[
                    "MOE-YOCO-KV",
                    "issue-8196",
                    "july-baseline",
                    "overtrain-750tpp",
                    variant_name,
                    "d512",
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
        steps=[build_step(fixed_yoco=False), build_step(fixed_yoco=True)],
        description=(
            "Matched exact-July d512 control and fixed-YOCO runs at the Marin overtraining baseline of "
            "750 tokens per active parameter."
        ),
    )
