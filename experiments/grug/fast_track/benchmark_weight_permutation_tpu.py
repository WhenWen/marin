# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the scalar weight permutation inside the MoE backward combine."""

import argparse
import hashlib
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
import ml_dtypes
import numpy as np
from levanter.grug._moe.ep_ragged_all_to_all import _permute_assignment_weights_sort

from experiments.grug.fast_track.benchmark_moe_scalar_permutation_tpu import DISTRIBUTIONS
from experiments.grug.fast_track.benchmark_moe_scalar_permutation_tpu import fixture as routing_fixture

IMPLEMENTATIONS = ("reference", "sort", "scatter")


def permute_weights(order, weights, *, implementation, shared_inverse):
    inverse = jnp.argsort(order) if shared_inverse or implementation != "reference" else None
    if implementation == "reference":
        result = weights[order]
    elif implementation == "sort":
        result = _permute_assignment_weights_sort(weights, order)
    elif implementation == "scatter":
        result = jnp.zeros_like(weights).at[inverse].set(weights, unique_indices=True)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    return result, inverse


def run(order, weights, cotangent, *, implementation, shared_inverse, composed):
    values, inverse = permute_weights(order, weights, implementation=implementation, shared_inverse=shared_inverse)
    if composed:
        token_cotangent = cotangent[order // 8].astype(jnp.float32)
        values = (token_cotangent * values[:, None]).astype(cotangent.dtype)
    return (values, inverse) if shared_inverse else (values,)


def fixture(seed, assignments, experts, distribution, width):
    (order, _, weights, accepted), (_, inverse, _) = routing_fixture(seed, assignments, experts, distribution)
    weights[::23] = -0.0
    weights = np.where(accepted, weights, np.float32(0))
    rng = np.random.default_rng(seed + 100)
    cotangent = rng.normal(size=(assignments // 8, max(1, width))).astype(ml_dtypes.bfloat16)
    cotangent.flat[::29] = -0.0
    values = weights[order]
    if width:
        values = (cotangent[order // 8].astype(np.float32) * values[:, None]).astype(ml_dtypes.bfloat16)
    return (order, weights, cotangent, accepted), (values, inverse)


def check(actual, expected):
    rows = []
    for value, oracle in zip(actual, expected, strict=True):
        value = np.ascontiguousarray(value)
        oracle = np.ascontiguousarray(oracle)
        error = np.abs(value.astype(np.float32) - oracle.astype(np.float32))
        rows.append(
            dict(
                exact=value.dtype == oracle.dtype and bool(np.array_equal(value.view(np.uint8), oracle.view(np.uint8))),
                finite=bool(np.isfinite(value).all()),
                max_abs_error=float(error.max()),
                mean_abs_error=float(error.mean()),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    device = jax.local_devices(backend="cpu" if args.cpu_check else "tpu")[0]
    if not args.cpu_check and device.device_kind != "TPU v4":
        raise ValueError("Use a TPU v4 chip")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Artifacts must remain in us-central2")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "weight-permutation"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "revision": args.revision,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_source_sha256": (
            hashlib.sha256(Path(__file__).with_name("benchmark_moe_scalar_permutation_tpu.py").read_bytes()).hexdigest()
        ),
        "device_kind": device.device_kind,
        "device_count_used": 1,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "cpu_check": args.cpu_check,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "status": "running",
        "compilation": [],
        "records": [],
    }

    def save():
        path = folder / "weight-permutation.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    compiled = {}
    shapes = [(64, 16, (0, 16, 32)), (2048, 16, (0, 16, 32))]
    if not args.cpu_check:
        shapes.append((262144, 512, (0, 256, 512)))
    save()
    with jax.default_device(device):
        for assignments, experts, widths in shapes:
            for width in widths:
                for seed in (0, 1, 2):
                    for distribution in DISTRIBUTIONS:
                        host, expected = fixture(seed, assignments, experts, distribution, width)
                        path = folder / f"fixture-{assignments}-{width}-{seed}-{distribution}.npz"
                        np.savez(path, order=host[0], weights=host[1], cotangent=host[2], accepted=host[3])
                        fs.put_file(str(path), f"{prefix}/{path.name}")
                        arrays = tuple(jax.device_put(a, device) for a in host[:3])
                        for shared in (False, True):
                            row = dict(
                                assignments=assignments,
                                experts=experts,
                                width=width,
                                seed=seed,
                                distribution=distribution,
                                shared_inverse=shared,
                                implementations={},
                                samples=[],
                            )
                            for name in IMPLEMENTATIONS:
                                key = (assignments, width, shared, name)
                                if key not in compiled:
                                    fn = jax.jit(
                                        partial(run, implementation=name, shared_inverse=shared, composed=bool(width))
                                    )
                                    start = time.monotonic()
                                    executable = fn.lower(*arrays).compile()
                                    compiled[key] = executable
                                    report["compilation"].append(
                                        dict(
                                            assignments=assignments,
                                            width=width,
                                            shared_inverse=shared,
                                            implementation=name,
                                            seconds=time.monotonic() - start,
                                            temporary_bytes=executable.memory_analysis().temp_size_in_bytes,
                                        )
                                    )
                                    if assignments == 262144:
                                        path = folder / f"hlo-{width}-shared{shared}-{name}.txt"
                                        path.write_text(executable.as_text())
                                        fs.put_file(str(path), f"{prefix}/{path.name}")
                                output = jax.block_until_ready(compiled[key](*arrays))
                                row["implementations"][name] = check(output, expected if shared else expected[:1])
                            row["passed_numerics"] = all(
                                m["exact"] and m["finite"] for ms in row["implementations"].values() for m in ms
                            )
                            if assignments == 262144:
                                for name in IMPLEMENTATIONS:
                                    for _ in range(2):
                                        jax.block_until_ready(compiled[(assignments, width, shared, name)](*arrays))
                                for repeat in range(8):
                                    names = IMPLEMENTATIONS if repeat % 2 == 0 else IMPLEMENTATIONS[::-1]
                                    for name in names:
                                        start = time.monotonic()
                                        jax.block_until_ready(compiled[(assignments, width, shared, name)](*arrays))
                                        row["samples"].append(
                                            dict(implementation=name, repeat=repeat, seconds=time.monotonic() - start)
                                        )
                                row["median_seconds"] = {
                                    name: statistics.median(
                                        r["seconds"] for r in row["samples"] if r["implementation"] == name
                                    )
                                    for name in IMPLEMENTATIONS
                                }
                            report["records"].append(row)
                            save()
                            print(json.dumps({k: v for k, v in row.items() if k not in ("samples", "implementations")}))
    report["numerical_gate"] = all(r["passed_numerics"] for r in report["records"])
    report["qualification"] = {}
    if not args.cpu_check:
        for name in IMPLEMENTATIONS[1:]:
            cases = []
            for row in report["records"]:
                if row["assignments"] != 262144 or row["distribution"] != "uniform_partial":
                    continue
                ratio = row["median_seconds"][name] / row["median_seconds"]["reference"]
                cases.append(
                    dict(
                        seed=row["seed"],
                        width=row["width"],
                        shared_inverse=row["shared_inverse"],
                        candidate_reference_ratio=ratio,
                        passed=ratio <= (1.01 if row["width"] else 0.95),
                    )
                )
            assert len(cases) == 18
            report["qualification"][name] = dict(
                passed=report["numerical_gate"] and all(c["passed"] for c in cases), cases=cases
            )
    report["status"] = "completed"
    save()
    print(json.dumps({k: report[k] for k in ("numerical_gate", "qualification")}))
    if not report["numerical_gate"]:
        raise RuntimeError("Weight permutation failed the independent oracle")


if __name__ == "__main__":
    main()
