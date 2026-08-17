# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classical shared-projected-K/V YOCO on regular July cells."""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe_yoco_kv_reuse.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_yoco_kv_reuse.recipe import ExperimentPoint, classical_yoco_recipe, point_for_hidden_dim
from experiments.grug.moe_yoco_kv_reuse.train import GrugEvalConfig, GrugTrainerConfig

_TPU: str = "v5p-8"
_TPU_REGIONS: tuple[str, ...] = ("us-central1",)
_WANDB_GROUP: str = "MOE-CLASSICAL-YOCO-july-issue-8196"
_POINTS: tuple[ExperimentPoint, ...] = (
    point_for_hidden_dim(512),
    point_for_hidden_dim(768),
    point_for_hidden_dim(1024),
)
_VARIANTS: tuple[tuple[str, str | None], ...] = (
    ("classical", None),
    ("classical-expert-match", "expert"),
    ("classical-head-match", "heads"),
)


def build_step(point: ExperimentPoint, variant_name: str, parameter_match: str | None) -> ExecutorStep:
    model, optimizer = classical_yoco_recipe(point, parameter_match=parameter_match)
    run_id = f"MOE-CLASSICAL-YOCO-JULY-001-{variant_name}-d{point.hidden_dim}"
    return ExecutorStep(
        name=f"grug/moe_classical_yoco_july_{variant_name}_d{point.hidden_dim}",
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
                    variant_name,
                    f"d{point.hidden_dim}",
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
        steps=[
            build_step(point, variant_name, parameter_match)
            for point in _POINTS
            for variant_name, parameter_match in _VARIANTS
        ],
        description=(
            "Classical YOCO shared projected K/V on regular July cells, with bare, "
            "localized shared-expert, and localized query-head parameter-reinvestment variants."
        ),
    )
