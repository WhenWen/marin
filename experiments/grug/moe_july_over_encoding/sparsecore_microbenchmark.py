# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the rank-64 OE gradient microbenchmark on one v5p-8 slice."""

import subprocess
import sys
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep


@dataclass(frozen=True)
class SparseCoreMicrobenchmarkConfig:
    num_rows: int
    num_indices: int
    embedding_dim: int
    repeats: int


def run_sparsecore_microbenchmark(config: SparseCoreMicrobenchmarkConfig) -> dict[str, int]:
    """Run the standalone kernel benchmark and return its exact shape."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.grug.moe_july_over_encoding.bench_sparsecore_embedding",
            "--num-rows",
            str(config.num_rows),
            "--num-indices",
            str(config.num_indices),
            "--embedding-dim",
            str(config.embedding_dim),
            "--repeats",
            str(config.repeats),
        ],
        check=True,
    )
    return {
        "num_rows": config.num_rows,
        "num_indices": config.num_indices,
        "embedding_dim": config.embedding_dim,
    }


microbenchmark = ExecutorStep(
    name="grug/MOE-OE-SPARSECORE-MICROBENCH-RANK64-SHARED-VALUES-V9-7368",
    fn=run_sparsecore_microbenchmark,
    config=SparseCoreMicrobenchmarkConfig(
        num_rows=3_238_400,
        num_indices=262_144,
        embedding_dim=64,
        repeats=10,
    ),
    resources=ResourceConfig.with_tpu("v5p-8", regions=["us-east5"]),
)


if __name__ == "__main__":
    executor_main(
        steps=[microbenchmark],
        description="Rank-64 OE dense-gradient SparseCore correctness and latency microbenchmark.",
    )
