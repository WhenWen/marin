# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen origin-side MoE weight gradients with operator oracles and TPU timings."""

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug._moe import ep_ragged_all_to_all as ragged
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.benchmark_moe_transport_tpu import _NAMES, _SPECS, evaluate, fixture, oracle
from experiments.grug.fast_track.diagnose_kda_state_precision import errors
from experiments.grug.fast_track.moe_saved_output import saved_output_routed_experts
from experiments.grug.fast_track.moe_expert_residuals import saved_expert_backward, saved_expert_forward

_DISTRIBUTIONS = ("uniform", "sparse", "imbalanced", "empty", "zero-weight", "tiny-weight")


def compile_operator(arrays, mesh, capacity, implementation, comparison):
    with ExitStack() as stack:
        stack.enter_context(patch.object(ragged, "_accepted_assignments", ragged._accepted_assignments_cutoff))
        stack.enter_context(patch.object(ragged, "_restore_assignment_scalars", ragged._restore_assignment_scalars_sort))
        stack.enter_context(patch.object(ragged, "_permute_assignment_weights", ragged._permute_assignment_weights_sort))
        if comparison == "expert-residuals" or implementation == "saved-output":
            stack.enter_context(patch.object(ragged, "_routed_experts", saved_output_routed_experts))
        if implementation == "expert-residuals":
            stack.enter_context(patch.object(ragged._RaggedDotExpertMlp, "forward", saved_expert_forward))
            stack.enter_context(patch.object(ragged._RaggedDotExpertMlp, "backward", saved_expert_backward))
        return jax.jit(partial(evaluate, mesh=mesh, capacity_factor=capacity)).lower(*arrays).compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--comparison", choices=("saved-output", "expert-residuals"), default="saved-output")
    args = parser.parse_args()
    implementations = ("reference", args.comparison)
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Keep artifacts in us-central2")
    if not args.cpu_check:
        require_tpu_rounding()
        DistributedConfig().initialize()
    devices = jax.devices()
    if len(devices) != 16 or any(d.platform != ("cpu" if args.cpu_check else "tpu") for d in devices):
        raise ValueError("Use 16 CPU devices for non-training preflight or one TPU v4-32 slice")
    if not args.cpu_check and any(d.device_kind != "TPU v4" for d in devices):
        raise ValueError("The screen is frozen for TPU v4")
    rank = jax.process_index()
    name = "cpu-check" if args.cpu_check else f"rank-{rank}"
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / f"saved-output-{name}"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(f"{args.artifact_prefix}/{name}")
    report = dict(
        revision=args.revision,
        comparison=args.comparison,
        rank=rank,
        platform=devices[0].platform,
        status="running",
        numerics=[],
        timings=[],
        full_shape_checks=[],
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        source_hashes={
            filename: hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
            for filename in (
                "moe_saved_output.py",
                "moe_expert_residuals.py",
                "benchmark_moe_saved_output_tpu.py",
                "benchmark_moe_transport_tpu.py",
            )
        },
    )

    def save():
        path = folder / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    mesh = compact_grug_mesh(expert_axis_size=16, replica_axis_size=1)
    specs = (*_SPECS, P("expert", None), P("expert", None))
    save()
    with jax.set_mesh(mesh):
        compiled = {}
        for seed in ((0,) if args.cpu_check else (0, 1, 2)):
            for distribution in _DISTRIBUTIONS:
                for width in ((128,) if args.cpu_check else (128, 256)):
                    host = fixture(seed, distribution, width)
                    if distribution == "zero-weight":
                        host[1][:, ::2] = 0
                    elif distribution == "tiny-weight":
                        host[1][:, ::2] *= 2.0**-20
                    if rank == 0:
                        path = folder / f"fixture-{seed}-{distribution}-{width}.npz"
                        names = ("x", "weights", "valid", "up", "down", "selected", "cotangent")
                        np.savez(path, **dict(zip(names, host, strict=True)))
                        fs.put_file(str(path), f"{prefix}/{path.name}")
                    for capacity in (0.5, 1.05):
                        expected, expected_drops = oracle(*host, capacity, 16)
                        # Preserve the existing custom-VJP convention at exactly zero weight.
                        expected[2][host[1] == 0] = 0
                        arrays = tuple(
                            jax.make_array_from_callback(a.shape, NamedSharding(mesh, spec), lambda index, a=a: a[index])
                            for a, spec in zip(host, specs, strict=True)
                        )
                        row = dict(seed=seed, distribution=distribution, width=width, capacity=capacity, arms={})
                        outputs = {}
                        for arm in implementations:
                            key = (width, capacity, arm)
                            if key not in compiled:
                                compiled[key] = compile_operator(arrays, mesh, capacity, arm, args.comparison)
                            result, drops = jax.block_until_ready(compiled[key](*arrays))
                            outputs[arm] = tuple(
                                np.asarray(multihost_utils.process_allgather(a, tiled=True)) for a in result
                            )
                            row["arms"][arm] = dict(
                                tensors={
                                    key: errors(a, b) for key, a, b in zip(_NAMES, outputs[arm], expected, strict=True)
                                },
                                drops=int(drops.sender_dropped),
                                receiver_drops=int(drops.receiver_dropped),
                            )
                        reference, candidate = (row["arms"][arm] for arm in implementations)
                        row["exact_nonrouter"] = {
                            name: outputs["reference"][index].tobytes() == outputs[args.comparison][index].tobytes()
                            for index, name in enumerate(_NAMES)
                            if name != "dweights"
                        }
                        row["direct_errors"] = {
                            name: errors(a, b)
                            for name, a, b in zip(_NAMES, outputs[args.comparison], outputs["reference"], strict=True)
                        }
                        row["passed"] = bool(
                            reference["drops"] == candidate["drops"] == expected_drops
                            and reference["receiver_drops"] == candidate["receiver_drops"] == 0
                            and (args.comparison == "expert-residuals" or all(row["exact_nonrouter"].values()))
                            and all(
                                reference["tensors"][name]["finite"]
                                and candidate["tensors"][name]["finite"]
                                and all(
                                    candidate["tensors"][name][metric]
                                    <= 1.25 * reference["tensors"][name][metric] + 1e-3
                                    for metric in ("max_abs_error", "relative_l2")
                                )
                                for name in _NAMES
                            )
                        )
                        report["numerics"].append(row)
                        save()
                        print(
                            json.dumps(
                                dict(
                                    seed=seed,
                                    distribution=distribution,
                                    width=width,
                                    capacity=capacity,
                                    passed=row["passed"],
                                )
                            ),
                            flush=True,
                        )
        if not all(row["passed"] for row in report["numerics"]):
            report["status"] = "numerical-screen-failed"
            save()
            raise ValueError("Numerical screen failed; preserve reports and do not time/promote")
        compiled.clear()
        if args.cpu_check:
            report["status"] = "completed"
            save()
            return
        for width in (256, 512):
            shapes = (
                (524288, 256),
                (524288, 8),
                (524288,),
                (512, 256, 384),
                (512, 384, width),
                (524288, 8),
                (524288, width),
            )
            dtypes = (jnp.bfloat16, jnp.float32, jnp.bool_, jnp.float32, jnp.float32, jnp.int32, jnp.bfloat16)
            for seed in (0, 1, 2):
                arrays = []
                for index, (shape, spec, dtype) in enumerate(zip(shapes, specs, dtypes, strict=True)):
                    sharding = NamedSharding(mesh, spec)
                    if index == 2:
                        a = jnp.ones(shape, dtype, out_sharding=sharding)
                    elif index == 1:
                        a = jnp.full(shape, 1 / 8, dtype, out_sharding=sharding)
                    elif index == 5:
                        ids = (
                            (np.arange(shape[0])[:, None] * 17 + np.arange(8)[None, :] * 31 + seed * 13) % 512
                        ).astype(np.int32)
                        a = jax.make_array_from_callback(shape, sharding, lambda index, ids=ids: ids[index])
                    else:
                        a = jax.random.normal(
                            jax.random.key(seed * 100 + 10 + index), shape, dtype, out_sharding=sharding
                        )
                        if index in (3, 4):
                            a = a / np.sqrt(shape[1])
                    arrays.append(a)
                for arm in implementations:
                    if arm not in compiled:
                        start = time.monotonic()
                        compiled[arm] = compile_operator(arrays, mesh, 1.05, arm, args.comparison)
                        report.setdefault("compilation", []).append(
                            dict(
                                width=width,
                                arm=arm,
                                seconds=time.monotonic() - start,
                                compiler_temporary_bytes=compiled[arm].memory_analysis().temp_size_in_bytes,
                            )
                        )
                        if rank == 0:
                            path = folder / f"width-{width}-{arm}-hlo.txt"
                            path.write_text(compiled[arm].as_text())
                            fs.put_file(str(path), f"{prefix}/{path.name}")
                    for _ in range(2):
                        result, drops = jax.block_until_ready(compiled[arm](*arrays))
                    finite = bool(jax.jit(lambda xs: jnp.stack([jnp.isfinite(a).all() for a in xs]).all())(result))
                    report["full_shape_checks"].append(
                        dict(
                            width=width,
                            seed=seed,
                            arm=arm,
                            finite=finite,
                            drops=int(drops.sender_dropped),
                            receiver_drops=int(drops.receiver_dropped),
                        )
                    )
                for repeat in range(8):
                    for arm in (implementations if repeat % 2 == 0 else implementations[::-1]):
                        multihost_utils.sync_global_devices(f"saved-output-{width}-{seed}-{repeat}-{arm}")
                        start = time.monotonic()
                        jax.block_until_ready(compiled[arm](*arrays))
                        report["timings"].append(
                            dict(width=width, seed=seed, arm=arm, repeat=repeat, seconds=time.monotonic() - start)
                        )
                save()
            compiled.clear()
        report["median_seconds"] = [
            dict(
                width=width,
                seed=seed,
                arm=arm,
                seconds=statistics.median(
                    r["seconds"] for r in report["timings"] if (r["width"], r["seed"], r["arm"]) == (width, seed, arm)
                ),
            )
            for width in (256, 512)
            for seed in (0, 1, 2)
            for arm in implementations
        ]
        report["status"] = "completed"
        save()


if __name__ == "__main__":
    main()
