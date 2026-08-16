# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classical shared-projected-K/V YOCO variants at the d512 750-TPP horizon."""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe_yoco_kv_reuse.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_yoco_kv_reuse.recipe import OVERTRAIN_D512_750_TPP, classical_yoco_recipe
from experiments.grug.moe_yoco_kv_reuse.train import GrugEvalConfig, GrugTrainerConfig

_TPU: str = "v5p-8"
_TPU_REGIONS: tuple[str, ...] = ("us-central1",)
_WANDB_GROUP: str = "MOE-CLASSICAL-YOCO-overtrain-750tpp-issue-8196"
_VARIANTS: tuple[tuple[str, str | None], ...] = (
    ("classical", None),
    ("classical-expert-match", "expert"),
    ("classical-head-match", "heads"),
)


def build_step(variant_name: str, parameter_match: str | None) -> ExecutorStep:
    point = OVERTRAIN_D512_750_TPP
    model, optimizer = classical_yoco_recipe(point, parameter_match=parameter_match)
    run_id = f"MOE-CLASSICAL-YOCO-OVERTRAIN-750TPP-001-{variant_name}-d512"
    return ExecutorStep(
        name=f"grug/moe_classical_yoco_overtrain_750tpp_{variant_name}_d512",
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
                    "MOE-CLASSICAL-YOCO",
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
        steps=[build_step(variant_name, parameter_match) for variant_name, parameter_match in _VARIANTS],
        description=(
            "Classical YOCO shared projected K/V at d512 and 750 tokens per active parameter, with bare, "
            "localized shared-expert, and localized query-head parameter-reinvestment variants."
        ),
    )
