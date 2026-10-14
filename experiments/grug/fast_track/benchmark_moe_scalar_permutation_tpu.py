# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare inverse gather, key/value sort, and scatter for MoE scalar gradients."""

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


IMPLEMENTATIONS = ("reference", "sort", "scatter")
DISTRIBUTIONS = ("uniform_partial", "concentrated_invalid", "identity_accepted", "reverse_dropped")


def restore_scalars(order, values, *, implementation):
    if implementation == "reference":
        return values[jnp.argsort(order)]
    if implementation == "sort":
        return jax.lax.sort_key_val(order, values)[1]
    if implementation == "scatter":
        return jnp.zeros_like(values).at[order].set(values, unique_indices=True)
    raise ValueError(f"Unknown implementation: {implementation}")


def weight_gradient(order, values, weights, accepted, *, implementation, reuse_inverse):
    restored = restore_scalars(order, values, implementation=implementation)
    divisible = accepted & (weights != 0)
    gradient = jnp.where(divisible, restored / jnp.where(divisible, weights, 1), 0)
    return (gradient, jnp.argsort(order)) if reuse_inverse else gradient


def fixture(seed, assignments, experts, distribution):
    rng = np.random.default_rng(seed)
    selected = rng.integers(experts, size=assignments, dtype=np.int32)
    accepted = rng.random(assignments) < 0.75
    if distribution == "concentrated_invalid":
        selected %= 4
        selected[::31] = experts
        accepted[::31] = False
    order = np.argsort(selected, stable=True).astype(np.int32)
    if distribution == "identity_accepted":
        order = np.arange(assignments, dtype=np.int32)
        accepted[:] = True
    elif distribution == "reverse_dropped":
        order = np.arange(assignments - 1, -1, -1, dtype=np.int32)
        accepted[:] = False
    values = rng.normal(size=assignments).astype(np.float32)
    values[::17] = -0.0
    weights = rng.uniform(0.1, 0.9, size=assignments).astype(np.float32)
    weights[::19] = 0
    weights[1::19] = 2**-20
    restored = np.empty_like(values)
    restored[order] = values
    inverse = np.empty_like(order)
    inverse[order] = np.arange(assignments, dtype=np.int32)
    divisible = accepted & (weights != 0)
    gradient = np.zeros_like(values)
    np.divide(restored, weights, out=gradient, where=divisible)
    return (order, values, weights, accepted), (restored, inverse, gradient)


def check(actual, expected, reuse_inverse):
    gradient = np.asarray(actual[0] if reuse_inverse else actual)
    inverse_exact = not reuse_inverse or np.array_equal(np.asarray(actual[1]), expected[1])
    return {
        "finite": bool(np.isfinite(gradient).all()),
        "inverse_exact": bool(inverse_exact),
        "max_abs_error": float(np.max(np.abs(gradient - expected[2]))),
        "gradient_close": bool(np.allclose(gradient, expected[2], atol=1e-5, rtol=1e-5)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    device = jax.local_devices(backend="tpu")[0]
    if device.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Use TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "scalar-permutation"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "revision": args.revision,
        "status": "running",
        "device_type": device.device_kind,
        "device_count_used": 1,
        "backend": "tpu",
        "dtype": "float32",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "records": [],
    }

    def save():
        path = folder / "scalar-permutation.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    compiled = {}
    permutations = {}
    save()
    with jax.default_device(device):
        for assignments, experts in ((2048, 16), (262144, 512)):
            for seed in (0, 1, 2):
                for distribution in DISTRIBUTIONS:
                    host, expected = fixture(seed, assignments, experts, distribution)
                    path = folder / f"fixture-{assignments}-{seed}-{distribution}.npz"
                    np.savez(path, order=host[0], values=host[1], weights=host[2], accepted=host[3])
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
                        for name in IMPLEMENTATIONS:
                            key = (assignments, reuse, name)
                            if key not in compiled:
                                fn = jax.jit(partial(weight_gradient, implementation=name, reuse_inverse=reuse))
                                start = time.monotonic()
                                executable = fn.lower(*arrays).compile()
                                compiled[key] = (executable, time.monotonic() - start)
                                path = folder / f"hlo-{assignments}-reuse{reuse}-{name}.txt"
                                path.write_text(executable.as_text())
                                fs.put_file(str(path), f"{prefix}/{path.name}")
                            if (assignments, name) not in permutations:
                                permutations[(assignments, name)] = jax.jit(
                                    partial(restore_scalars, implementation=name)
                                ).lower(*arrays[:2]).compile()
                            restored = np.asarray(permutations[(assignments, name)](*arrays[:2]))
                            executable, compile_time = compiled[key]
                            actual = jax.block_until_ready(executable(*arrays))
                            metrics = check(actual, expected, reuse)
                            metrics["permutation_exact"] = bool(
                                np.array_equal(restored.view(np.uint32), expected[0].view(np.uint32))
                            )
                            metrics["compile_seconds"] = compile_time
                            metrics["compiler_temporary_bytes"] = executable.memory_analysis().temp_size_in_bytes
                            row["implementations"][name] = metrics
                            for _ in range(2):
                                jax.block_until_ready(executable(*arrays))
                        for repeat in range(8):
                            names = IMPLEMENTATIONS if repeat % 2 == 0 else IMPLEMENTATIONS[::-1]
                            for name in names:
                                executable, _ = compiled[(assignments, reuse, name)]
                                start = time.monotonic()
                                jax.block_until_ready(executable(*arrays))
                                row["samples"].append(
                                    {"implementation": name, "repeat": repeat, "seconds": time.monotonic() - start}
                                )
                        row["median_seconds"] = {
                            name: statistics.median(r["seconds"] for r in row["samples"] if r["implementation"] == name)
                            for name in IMPLEMENTATIONS
                        }
                        row["passed_numerics"] = all(
                            r["finite"] and r["inverse_exact"] and r["permutation_exact"] and r["gradient_close"]
                            for r in row["implementations"].values()
                        )
                        report["records"].append(row)
                        save()
                        print(json.dumps(row), flush=True)
    report["status"] = "completed"
    save()
    if not all(r["passed_numerics"] for r in report["records"]):
        raise RuntimeError("Scalar permutation or weight gradients failed the independent oracle")


if __name__ == "__main__":
    main()
