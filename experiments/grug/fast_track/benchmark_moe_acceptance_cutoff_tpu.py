# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test one expert-cutoff lookup against two routing-acceptance lookups."""

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
import numpy as np
from levanter.grug._moe.ep_ragged_all_to_all import _accepted_assignments_inverse


def accepted_cutoff(selected, order, sizes, accepted_sizes):
    """Use one cutoff per expert; counts and assignment positions fit int32."""
    positions = jnp.argsort(order)
    cutoff = jnp.cumsum(sizes) - sizes + accepted_sizes
    expert = jnp.minimum(selected, sizes.shape[0] - 1)
    return (selected < sizes.shape[0]) & (positions < cutoff[expert])


def acceptance(selected, order, sizes, accepted_sizes, *, cutoff, reuse_inverse):
    fn = accepted_cutoff if cutoff else _accepted_assignments_inverse
    mask = fn(selected, order, sizes, accepted_sizes)
    return (mask, jnp.argsort(order)) if reuse_inverse else mask


def fixture(seed, assignments, experts, distribution):
    rng = np.random.default_rng(seed)
    selected = rng.integers(experts, size=assignments, dtype=np.int32)
    if distribution == "concentrated_invalid":
        selected %= 4
        selected[::31] = experts
    order = np.argsort(selected, stable=True).astype(np.int32)
    sizes = np.bincount(selected, minlength=experts + 1)[:experts].astype(np.int32)
    accepted_sizes = sizes * 3 // 4
    if distribution == "all_dropped":
        accepted_sizes[:] = 0
    elif distribution == "all_accepted":
        accepted_sizes = sizes.copy()
    mask = np.zeros(assignments, bool)
    for expert, count in enumerate(accepted_sizes):
        mask[np.flatnonzero(selected == expert)[:count]] = True
    return (selected, order, sizes, accepted_sizes), mask, np.argsort(order).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    device = jax.local_devices(backend="tpu")[0]
    if device.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Use TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "acceptance-cutoff"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "revision": args.revision,
        "status": "running",
        "device_type": device.device_kind,
        "device_count_used": 1,
        "backend": "tpu",
        "dtype": "int32_to_bool",
        "block_sizes": "XLA selected",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
        "records": [],
    }

    def save():
        path = folder / "acceptance-cutoff.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    compiled = {}
    save()
    with jax.default_device(device):
        for assignments, experts in ((2048, 16), (262144, 512)):
            for seed in (0, 1, 2):
                for distribution in ("uniform_partial", "concentrated_invalid", "all_dropped", "all_accepted"):
                    host, expected_mask, expected_inverse = fixture(seed, assignments, experts, distribution)
                    path = folder / f"fixture-{assignments}-{seed}-{distribution}.npz"
                    np.savez(path, selected=host[0], order=host[1], sizes=host[2], accepted_sizes=host[3], mask=expected_mask)
                    fs.put_file(str(path), f"{prefix}/{path.name}")
                    arrays = tuple(jax.device_put(a, device) for a in host)
                    for reuse in (False, True):
                        row = {
                            "shape": [assignments, experts],
                            "seed": seed,
                            "distribution": distribution,
                            "reuse_inverse": reuse,
                            "implementations": {},
                            "samples": [],
                        }
                        for name in ("reference", "cutoff"):
                            key = (assignments, experts, reuse, name)
                            if key not in compiled:
                                fn = jax.jit(partial(acceptance, cutoff=name == "cutoff", reuse_inverse=reuse))
                                start = time.monotonic()
                                executable = fn.lower(*arrays).compile()
                                compiled[key] = (executable, time.monotonic() - start)
                                path = folder / f"hlo-{assignments}-reuse{reuse}-{name}.txt"
                                path.write_text(executable.as_text())
                                fs.put_file(str(path), f"{prefix}/{path.name}")
                            executable, compile_time = compiled[key]
                            actual = jax.block_until_ready(executable(*arrays))
                            expected = (expected_mask, expected_inverse) if reuse else expected_mask
                            mismatch = [int(np.count_nonzero(np.asarray(a) != b)) for a, b in zip(
                                jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
                            )]
                            row["implementations"][name] = {
                                "mismatches": mismatch,
                                "compile_seconds": compile_time,
                                "compiler_temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
                            }
                            for _ in range(2):
                                jax.block_until_ready(executable(*arrays))
                        for repeat in range(8):
                            for name in (("reference", "cutoff") if repeat % 2 == 0 else ("cutoff", "reference")):
                                executable, _ = compiled[(assignments, experts, reuse, name)]
                                start = time.monotonic()
                                jax.block_until_ready(executable(*arrays))
                                row["samples"].append({"implementation": name, "repeat": repeat,
                                                       "seconds": time.monotonic() - start})
                        row["median_seconds"] = {name: statistics.median(
                            r["seconds"] for r in row["samples"] if r["implementation"] == name
                        ) for name in ("reference", "cutoff")}
                        row["passed_exact"] = all(not any(r["mismatches"]) for r in row["implementations"].values())
                        report["records"].append(row)
                        save()
                        print(json.dumps(row), flush=True)
    report["status"] = "completed"
    save()
    if not all(r["passed_exact"] for r in report["records"]):
        raise RuntimeError("Acceptance mask or inverse positions differed from the independent oracle")


if __name__ == "__main__":
    main()
