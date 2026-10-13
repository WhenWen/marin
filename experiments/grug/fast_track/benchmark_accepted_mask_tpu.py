# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare exact routing acceptance masks on one TPU v4 chip."""

import argparse
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
from levanter.grug._moe.ep_ragged_all_to_all import _accepted_assignments_inverse, _accepted_assignments_scatter


def acceptance(selected, order, sizes, accepted_sizes, *, implementation, reuse_inverse):
    positions = jnp.argsort(order)
    if implementation == "reference":
        mask = _accepted_assignments_scatter(selected, order, sizes, accepted_sizes)
    else:
        count = sizes.shape[0]
        starts = jnp.cumsum(sizes) - sizes
        if implementation == "inverse_direct":
            mask = _accepted_assignments_inverse(selected, order, sizes, accepted_sizes)
        else:
            sorted_experts = selected[order]
            experts = jnp.minimum(sorted_experts, count - 1)
            rank = jnp.arange(order.size, dtype=jnp.int32) - starts[experts]
            sorted_mask = (sorted_experts < count) & (rank < accepted_sizes[experts])
            if implementation == "integer_scatter":
                mask = (
                    jnp.zeros(order.shape, jnp.int32)
                    .at[order]
                    .set(sorted_mask.astype(jnp.int32), unique_indices=True, mode="promise_in_bounds")
                    .astype(jnp.bool_)
                )
            elif implementation == "inverse_gather":
                mask = sorted_mask[positions]
            else:
                raise ValueError(implementation)
    return (mask, positions) if reuse_inverse else mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    device = jax.local_devices()[0]
    if device.device_kind != "TPU v4":
        raise RuntimeError("This benchmark requires TPU v4")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        parser.error("Artifacts must stay in us-central2")
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    records = []
    with jax.default_device(device):
        for assignments, experts in ((2048, 16), (262144, 512)):
            for distribution in ("uniform_partial", "concentrated_invalid", "all_dropped", "all_accepted"):
                selected = jax.random.randint(jax.random.key(0), (assignments,), 0, experts)
                if distribution == "concentrated_invalid":
                    selected = selected % 4
                    selected = selected.at[::31].set(experts)
                order = jnp.argsort(selected, stable=True)
                sizes = jnp.bincount(selected, length=experts + 1)[:experts]
                accepted = sizes * 3 // 4
                if distribution == "all_dropped":
                    accepted = jnp.zeros_like(sizes)
                elif distribution == "all_accepted":
                    accepted = sizes
                inputs = (selected, order, sizes, accepted)
                for reuse in (False, True):
                    reference = None
                    for implementation in ("reference", "integer_scatter", "inverse_gather", "inverse_direct"):
                        record = {
                            "kernel": "moe_accepted_assignments",
                            "implementation": implementation,
                            "shape": [assignments, experts],
                            "distribution": distribution,
                            "reuse_inverse": reuse,
                            "dtype": "int32_to_bool",
                            "seed": 0,
                            "backend": "tpu",
                            "device_type": device.device_kind,
                            "device_count": 1,
                            "block_sizes": {},
                            "git_sha": args.revision,
                            "jax_version": jax.__version__,
                            "jaxlib_version": jaxlib.__version__,
                            "xla_flags": os.environ.get("XLA_FLAGS", ""),
                            "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                            "error": None,
                        }
                        try:
                            fn = partial(acceptance, implementation=implementation, reuse_inverse=reuse)
                            start = time.monotonic()
                            compiled = jax.jit(fn).lower(*inputs).compile()
                            record["compile_time"] = time.monotonic() - start
                            actual = jax.block_until_ready(compiled(*inputs))
                            if implementation == "reference":
                                reference = actual
                            if reference is None:
                                raise RuntimeError("Reference failed")
                            record["mismatches"] = [
                                int(jnp.sum(a != b))
                                for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True)
                            ]
                            if any(record["mismatches"]):
                                raise AssertionError("Acceptance mask or inverse positions differ")
                            record["temporary_bytes"] = compiled.memory_analysis().temp_size_in_bytes
                            if assignments == 262144 and distribution == "uniform_partial":
                                hlo = output / f"{implementation}-reuse{reuse}-hlo.txt"
                                hlo.write_text(compiled.as_text())
                                fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
                            for _ in range(2):
                                jax.block_until_ready(compiled(*inputs))
                            samples = []
                            for _ in range(5):
                                start = time.monotonic()
                                jax.block_until_ready(compiled(*inputs))
                                samples.append(time.monotonic() - start)
                            record.update(steady_state_time=statistics.median(samples), steady_state_samples=samples)
                        except Exception as exc:
                            record["error"] = repr(exc)
                        records.append(record)
                        print(json.dumps(record), flush=True)
                        report = output / "accepted-mask-benchmark.json"
                        report.write_text(json.dumps(records, indent=2) + "\n")
                        fs.put_file(str(report), f"{prefix}/{report.name}")
    if any(r["error"] for r in records):
        raise RuntimeError("An acceptance candidate failed; see the retained report")


if __name__ == "__main__":
    main()
