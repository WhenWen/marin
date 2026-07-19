# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry the d1024 two-RMSNorm OE Gate 2 cell with checkpoint-safe host RAM."""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, versioned

from experiments.grug.moe_july_over_encoding.normsum_sparsecore_gate2 import Gate2Point, build_gate2_step

_RUN_ID = "MOE-OE-JULY-NORMSUM-SC-GATE2-RAM400-d1024"
_WANDB_GROUP = "MOE-OE-JULY-normsum-sc-gate2-d1024-ram400-issue-7368"
_D1024_POINT = Gate2Point(hidden_dim=1024, batch_size=64, num_steps=16_080)


def build_step() -> ExecutorStep:
    """Build the exact d1024 Gate 2 cell with additional host RAM for checkpoint serialization."""
    base = build_gate2_step(_D1024_POINT, wandb_group=_WANDB_GROUP)
    if not isinstance(base.config.tracker, WandbConfig):
        raise TypeError(f"expected W&B tracker config, got {type(base.config.tracker)}")

    config = dataclasses.replace(
        base.config,
        run_id=_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8", ram="400g")),
        tracker=dataclasses.replace(base.config.tracker, group=_WANDB_GROUP),
    )
    return dataclasses.replace(base, name=f"grug/{_RUN_ID}", config=config)


if __name__ == "__main__":
    executor_main(
        steps=[build_step()],
        description="Checkpoint-safe July d1024 two-RMSNorm Over-Encoding Gate 2 recovery.",
        max_concurrent=1,
    )
