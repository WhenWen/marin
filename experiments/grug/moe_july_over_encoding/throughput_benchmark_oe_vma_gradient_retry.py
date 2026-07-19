# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank-64 OE validation with complete custom-VJP mesh metadata."""

from marin.execution.executor import executor_main

from experiments.grug.moe_july_over_encoding.throughput_benchmark import build_step

_RUN_ID = "MOE-JULY-VMA-GRAD-PERF-OE-d512"


if __name__ == "__main__":
    executor_main(
        steps=[build_step(enable_over_encoding=True, run_id=_RUN_ID)],
        description="Unchanged rank-64 OE throughput validation with complete custom-VJP mesh metadata.",
    )
