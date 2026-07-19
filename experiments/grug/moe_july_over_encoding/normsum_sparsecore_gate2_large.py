# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the d1024 and d1280 two-RMSNorm Over-Encoding Gate 2 cells."""

from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep

from experiments.grug.moe_july_over_encoding.normsum_sparsecore_gate2 import Gate2Point, build_gate2_step

_WANDB_GROUP = "MOE-OE-JULY-normsum-sc-gate2-large-issue-7368"

GATE2_POINTS: tuple[Gate2Point, ...] = (
    Gate2Point(hidden_dim=1024, batch_size=64, num_steps=16_080),
    Gate2Point(hidden_dim=1280, batch_size=128, num_steps=14_325),
)


def build_steps() -> list[ExecutorStep]:
    """Build the two larger canonical July Gate 2 cells."""
    return [build_gate2_step(point, wandb_group=_WANDB_GROUP) for point in GATE2_POINTS]


if __name__ == "__main__":
    executor_main(
        steps=build_steps(),
        description="Throughput-qualified July d1024/d1280 two-RMSNorm Over-Encoding Gate 2.",
        max_concurrent=len(GATE2_POINTS),
    )
