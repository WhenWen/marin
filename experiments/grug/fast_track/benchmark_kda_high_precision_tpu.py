# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen three-pass TPU KDA arithmetic against the frozen FP64 oracle grid."""

import argparse
import functools
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from experiments.grug.fast_track import belay_tpu_ops
from experiments.grug.fast_track import benchmark_kda_state_tpu as benchmark
from experiments.grug.fast_track import diagnose_kda_state_precision as diagnostic
from experiments.grug.fast_track.diagnose_kda_state_precision import errors, fixture_inputs, oracle
from experiments.grug.moe import kda, kda_state_tpu

ARMS = ("highest", "high")
NAMES = ("output", "final_state", "objective", "dq", "dk", "dv", "dg", "dbeta", "dinitial")
FIXTURES = (
    ("random", (1, 2, 128, 128), 37),
    ("repeated", (1, 2, 128, 128), 37),
    ("strong-decay", (1, 2, 128, 128), 37),
    ("long-packed", (1, 1, 1024, 128), 83),
    ("uninterrupted", (1, 1, 256, 128), 256),
)


def host_leaves(value):
    return tuple(np.asarray(x) for x in jax.tree.leaves(jax.device_get(value)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    tpu, cpu = jax.devices("tpu")[0], jax.devices("cpu")[0]
    if tpu.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("This screen requires TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    sources = (
        Path(__file__),
        Path(kda_state_tpu.__file__),
        Path(kda.__file__),
        Path(benchmark.__file__),
        Path(belay_tpu_ops.__file__),
        Path(diagnostic.__file__),
    )
    report = {
        "revision": args.revision,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "device": str(tpu),
        "status": "running",
        "compilations": [],
        "numerics": [],
        "timings": [],
    }

    def upload(path):
        fs.put_file(str(path), f"{prefix}/{path.name}")

    def save():
        path = folder / "high-precision.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        upload(path)

    compiled = {}

    def executable(arm, mode, arguments):
        shape = tuple(arguments[0][0].shape)
        key = (arm, mode, shape)
        if key not in compiled:
            precision = jax.lax.Precision.HIGHEST if arm == "highest" else jax.lax.Precision.HIGH
            fn = jax.jit(functools.partial(benchmark.evaluate, mode=mode))
            start = time.monotonic()
            # The partial passed into Pallas carries precision in its cache key.
            with (
                patch.object(
                    kda,
                    "_parallel_chunk_recurrence",
                    functools.partial(belay_tpu_ops.tpu_recurrence, precision=precision),
                ),
            ):
                compiled[key] = fn.lower(*arguments).compile()
            duration = time.monotonic() - start
            label = f"{'-'.join(map(str, shape))}-{arm}-{mode}"
            hlo = folder / f"{label}.hlo.txt"
            hlo.write_text(compiled[key].as_text())
            upload(hlo)
            report["compilations"].append(
                {
                    "arm": arm,
                    "mode": mode,
                    "shape": shape,
                    "compile_time": duration,
                    "temporary_bytes": compiled[key].memory_analysis().temp_size_in_bytes,
                    "hlo": hlo.name,
                }
            )
            save()
        return compiled[key]

    save()
    with jax.default_device(tpu):
        for seed in range(3):
            for fixture, shape, doc in FIXTURES:
                arguments = fixture_inputs(seed, fixture, shape, doc)
                represented = host_leaves(arguments)
                fixture_path = folder / f"{fixture}-seed-{seed}.npz"
                # NPZ does not retain ml_dtypes metadata: store BF16 bits explicitly.
                np.savez(
                    fixture_path,
                    **{
                        name: value.view(np.uint16) if i < 3 else value
                        for i, (name, value) in enumerate(
                            zip(
                                (
                                    "q_bf16_bits",
                                    "k_bf16_bits",
                                    "v_bf16_bits",
                                    "g",
                                    "beta",
                                    "initial",
                                    "ct",
                                    "ft",
                                    "segments",
                                ),
                                represented,
                                strict=True,
                            )
                        )
                    },
                )
                upload(fixture_path)
                actual = {}
                for arm in ARMS:
                    outputs = host_leaves(jax.block_until_ready(executable(arm, "forward", arguments)(*arguments)))
                    gradients = host_leaves(
                        jax.block_until_ready(executable(arm, "value_and_grad", arguments)(*arguments))
                    )
                    actual[arm] = (*outputs, *gradients)
                with jax.enable_x64(True), jax.default_device(cpu):
                    cpu_args = tuple(jnp.asarray(x, dtype=jnp.float64) for x in represented[:-1])
                    reference = host_leaves(
                        jax.block_until_ready(
                            jax.jit(oracle)(cpu_args[:6], cpu_args[6], cpu_args[7], jnp.asarray(represented[-1]))
                        )
                    )
                comparisons = {
                    arm: {name: errors(a, b) for name, a, b in zip(NAMES, actual[arm], reference, strict=True)}
                    for arm in ARMS
                }
                passed = all(comparisons[arm][name]["finite"] for arm in ARMS for name in NAMES) and all(
                    comparisons["high"][name][metric] <= 1.25 * comparisons["highest"][name][metric] + 0.001
                    for name in NAMES
                    for metric in ("max_abs_error", "relative_l2")
                )
                row = {
                    "fixture": fixture,
                    "shape": shape,
                    "document_length": doc,
                    "seed": seed,
                    "fixture_file": fixture_path.name,
                    "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                    "oracle_errors": comparisons,
                    "candidate_vs_reference": {
                        name: errors(a, b) for name, a, b in zip(NAMES, actual["high"], actual["highest"], strict=True)
                    },
                    "passed": passed,
                }
                report["numerics"].append(row)
                save()
                print(json.dumps({"fixture": fixture, "seed": seed, "passed": passed}), flush=True)
        report["numerics_passed"] = all(row["passed"] for row in report["numerics"])
        if not report["numerics_passed"]:
            report["status"] = "rejected_numerics"
            save()
            return
        for seed in range(3):
            for doc in (83, 1024, 4096):
                arguments = fixture_inputs(seed, "full", (8, 4, 4096, 128), doc)
                for mode in ("forward", "value_and_grad"):
                    functions = {arm: executable(arm, mode, arguments) for arm in ARMS}
                    values = {}
                    for _ in range(2):
                        for arm in ARMS:
                            values[arm] = jax.block_until_ready(functions[arm](*arguments))
                    samples = {arm: [] for arm in ARMS}
                    order = []
                    for round_id in range(8):
                        for arm in ARMS if round_id % 2 == 0 else ARMS[::-1]:
                            start = time.monotonic()
                            values[arm] = jax.block_until_ready(functions[arm](*arguments))
                            samples[arm].append(time.monotonic() - start)
                            order.append({"round": round_id, "arm": arm, "seconds": samples[arm][-1]})
                    host_values = {arm: host_leaves(values[arm]) for arm in ARMS}
                    direct = [errors(a, b) for a, b in zip(host_values["high"], host_values["highest"], strict=True)]
                    medians = {arm: statistics.median(samples[arm]) for arm in ARMS}
                    row = {
                        "seed": seed,
                        "document_length": doc,
                        "mode": mode,
                        "shape": [8, 4, 4096, 128],
                        "samples": order,
                        "medians": medians,
                        "candidate_reference_ratio": medians["high"] / medians["highest"],
                        "candidate_vs_reference": dict(
                            zip(NAMES[:2] if mode == "forward" else NAMES[2:], direct, strict=True)
                        ),
                        "finite": all(e["finite"] for e in direct),
                        "speed_passed": medians["high"] <= 0.9 * medians["highest"],
                    }
                    report["timings"].append(row)
                    save()
                    print(
                        json.dumps({k: v for k, v in row.items() if k not in ("candidate_vs_reference", "samples")}),
                        flush=True,
                    )
        report["passed"] = all(row["finite"] for row in report["timings"]) and all(
            row["speed_passed"] for row in report["timings"] if row["mode"] == "value_and_grad"
        )
        report["status"] = "completed"
        save()


if __name__ == "__main__":
    main()
