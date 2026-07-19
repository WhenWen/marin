# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact rank-64 OE-only recovery for the matched d512 throughput gate."""

from marin.execution.executor import executor_main

from experiments.grug.moe_july_over_encoding.throughput_benchmark import build_step

_RUN_ID = "MOE-JULY-EXPLICIT-GRID-PERF-OE-d512"


if __name__ == "__main__":
    executor_main(
        steps=[build_step(enable_over_encoding=True, run_id=_RUN_ID)],
        description="Rank-64 OE-only recovery for the matched canonical July d512 throughput gate.",
    )
