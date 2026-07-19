# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark XLA and SparseCore dense embedding-gradient accumulation on TPU."""

import argparse
import functools
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp

from experiments.grug.moe_july_over_encoding.sparsecore_embedding import embedding_scatter_add


def _timed_call(fn, ids, updates, *, repeats: int) -> tuple[jax.Array, float, float]:
    started = time.perf_counter()
    result = fn(ids, updates)
    result.block_until_ready()
    compile_time = time.perf_counter() - started

    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn(ids, updates)
        result.block_until_ready()
        durations.append(time.perf_counter() - started)
    return result, compile_time, statistics.median(durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rows", type=int, default=3_238_400)
    parser.add_argument("--num-indices", type=int, default=262_144)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    ids = jax.random.randint(key, (args.num_indices,), 0, args.num_rows, dtype=jnp.int32)
    updates = jax.random.normal(key, (args.num_indices, args.embedding_dim), dtype=jnp.bfloat16)
    implementations = ("xla", "sparsecore")
    results = {}

    for implementation in implementations:
        fn = jax.jit(
            functools.partial(
                embedding_scatter_add,
                num_rows=args.num_rows,
                implementation=implementation,
            )
        )
        result, compile_time, steady_state_time = _timed_call(fn, ids, updates, repeats=args.repeats)
        results[implementation] = result
        print(
            json.dumps(
                {
                    "backend": jax.default_backend(),
                    "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                    "block_sizes": {"scatter_window_size": 128},
                    "compile_time": compile_time,
                    "device_count": jax.device_count(),
                    "device_type": jax.devices()[0].device_kind,
                    "dtype": str(updates.dtype),
                    "error": None,
                    "git_sha": os.environ.get("GIT_COMMIT", "unknown"),
                    "implementation": implementation,
                    "kernel": "over_encoding_embedding_scatter_add",
                    "shape": {
                        "embedding_dim": args.embedding_dim,
                        "num_indices": args.num_indices,
                        "num_rows": args.num_rows,
                    },
                    "steady_state_time": steady_state_time,
                    "xla_flags": os.environ.get("XLA_FLAGS", ""),
                },
                sort_keys=True,
            )
        )

    absolute_error = jnp.abs(results["sparsecore"].astype(jnp.float32) - results["xla"].astype(jnp.float32))
    print(
        json.dumps(
            {
                "correctness": {
                    "max_absolute_error": float(jnp.max(absolute_error)),
                    "mean_absolute_error": float(jnp.mean(absolute_error)),
                }
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
