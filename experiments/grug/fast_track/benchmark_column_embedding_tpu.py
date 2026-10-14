# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen column-sharded lookup with an independent oracle and complete VJP timing.

Protocol: https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6172765675
No optimizer or training loop runs in this benchmark.
"""

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
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.column_sharded_embedding import column_sharded_embedding
from experiments.grug.fast_track.model import _BATCH_AXES, _FSDP_AXES, _replicated_fp32_gather

SHAPES = ((2048, 128, 16, 128), (524288, 512, 128, 4096))
DISTRIBUTIONS = ("uniform", "boundaries", "repeated64", "one_row")
IMPLEMENTATIONS = {"reference": _replicated_fp32_gather, "column": column_sharded_embedding}


def fixture(shape, seed, distribution):
    rows, width, batch, sequence = shape
    tokens = batch * sequence
    rng = np.random.default_rng(seed)
    starts = np.arange(16, dtype=np.int32) * (rows // 16)
    boundaries = np.concatenate((starts, starts + rows // 16 - 1))
    if distribution == "uniform":
        ids = rng.integers(rows, size=tokens, dtype=np.int32)
        ids[:32] = boundaries
    elif distribution == "boundaries":
        ids = np.resize(np.roll(boundaries, seed), tokens)
    elif distribution == "repeated64":
        available = np.setdiff1d(np.arange(rows, dtype=np.int32), boundaries)
        pool = np.concatenate((boundaries, rng.choice(available, 32, replace=False)))
        ids = pool[rng.integers(64, size=tokens)]
        ids[:64] = pool
    elif distribution == "one_row":
        ids = np.full(tokens, (0, rows // 2, rows - 1)[seed], dtype=np.int32)
    else:
        raise ValueError(distribution)
    arrays = {"ids": ids.reshape(batch, sequence)}
    if shape == SHAPES[0]:
        cot = rng.integers(-8, 9, size=(batch, sequence, width)).astype(np.float64) / 8
        oracle = np.zeros((rows, width), dtype=np.float64)
        np.add.at(oracle, ids, cot.reshape(tokens, width))
        arrays.update(cotangent=cot, gradient_oracle=oracle)
    else:
        token = ((np.arange(tokens) + 5 * seed) % 17 - 8).astype(np.float64) / 8
        column = ((np.arange(width) + seed) % 5 - 2).astype(np.float64) / 2
        row = np.bincount(ids, weights=token, minlength=rows)
        arrays.update(token_coefficients=token, column_coefficients=column, row_coefficients=row)
    return arrays


def make_arrays(shape, seed, fixture_arrays, mesh):
    rows, width, batch, sequence = shape
    row_spec = NamedSharding(mesh, P(_FSDP_AXES, None))
    token_spec = NamedSharding(mesh, P(_BATCH_AXES, None, None))
    ids = fixture_arrays["ids"]
    # Callback operands contain only the locally owned rows, including on four-host TPU.
    row_terms = (17 * np.arange(rows, dtype=np.int32) + seed)[:, None]
    column_terms = 3 * np.arange(width, dtype=np.int32)[None, :]

    def table_values(index):
        return (((row_terms[index[0]] + column_terms[:, index[1]]) % 257 - 128) / 128).astype(ml_dtypes.bfloat16)

    def output_values(index):
        selected = row_terms[ids[index[0], index[1]], 0]
        return (((selected[..., None] + column_terms[:, index[2]]) % 257 - 128) / 128).astype(ml_dtypes.bfloat16)

    def cotangent_values(index):
        if shape == SHAPES[0]:
            return fixture_arrays["cotangent"][index].astype(ml_dtypes.bfloat16)
        token = fixture_arrays["token_coefficients"].reshape(batch, sequence)[index[:2]]
        return (token[..., None] * fixture_arrays["column_coefficients"][index[2]]).astype(ml_dtypes.bfloat16)

    def gradient_values(index):
        if shape == SHAPES[0]:
            return fixture_arrays["gradient_oracle"][index].astype(np.float32)
        row = fixture_arrays["row_coefficients"][index[0]]
        column = fixture_arrays["column_coefficients"][index[1]]
        return (row[:, None] * column[None, :]).astype(np.float32)

    table = jax.make_array_from_callback((rows, width), row_spec, table_values)
    device_ids = jax.make_array_from_callback(
        (batch, sequence), NamedSharding(mesh, P(_BATCH_AXES, None)), lambda index: ids[index]
    )
    cotangent = jax.make_array_from_callback((batch, sequence, width), token_spec, cotangent_values)
    expected_output = jax.make_array_from_callback((batch, sequence, width), token_spec, output_values)
    expected_gradient = jax.make_array_from_callback((rows, width), row_spec, gradient_values)
    return (table, device_ids, cotangent), (expected_output, expected_gradient)


def evaluate(table, ids, cotangent, *, core, component):
    if component == "forward":
        return core(table, ids)
    output, backward = jax.vjp(lambda t: core(t, ids), table)
    gradient = backward(cotangent)[0]
    if component == "backward":
        return gradient
    return output, gradient


@jax.jit
def numerical_metrics(result, expected):
    output, gradient = result
    expected_output, expected_gradient = expected
    difference = gradient.astype(jnp.float32) - expected_gradient
    bits = jax.lax.bitcast_convert_type(output, jnp.uint16)
    expected_bits = jax.lax.bitcast_convert_type(expected_output, jnp.uint16)
    return dict(
        forward_bit_mismatches=jnp.sum(bits != expected_bits),
        finite=jnp.all(jnp.isfinite(output)) & jnp.all(jnp.isfinite(gradient)),
        relative_l2=jnp.sqrt(jnp.sum(difference**2)) / jnp.maximum(jnp.sqrt(jnp.sum(expected_gradient**2)), 1e-30),
        max_abs_error=jnp.max(jnp.abs(difference)),
    )


@jax.jit
def forward_mismatches(left, right):
    return jnp.sum(jax.lax.bitcast_convert_type(left, jnp.uint16) != jax.lax.bitcast_convert_type(right, jnp.uint16))


def paired_timings(compiled, inputs, label):
    for name, executable in compiled.items():
        for warmup in range(2):
            multihost_utils.sync_global_devices(f"{label}-{name}-warmup-{warmup}")
            jax.block_until_ready(executable(*inputs))
    samples = {name: [] for name in compiled}
    for repeat in range(8):
        order = list(compiled) if repeat % 2 == 0 else list(reversed(compiled))
        for name in order:
            multihost_utils.sync_global_devices(f"{label}-{repeat}-{name}")
            start = time.perf_counter()
            jax.block_until_ready(compiled[name](*inputs))
            samples[name].append(time.perf_counter() - start)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    worst = {
        name: float(np.max(multihost_utils.process_allgather(np.asarray(value)))) for name, value in medians.items()
    }
    return dict(samples=samples, host_medians=medians, worst_host_medians=worst)


def numerical_gate(metrics, pair_mismatches):
    reference, candidate = metrics["reference"], metrics["column"]
    return (
        pair_mismatches == 0
        and all(x["finite"] and x["forward_bit_mismatches"] == 0 for x in metrics.values())
        and candidate["relative_l2"] <= 1.25 * reference["relative_l2"] + 1e-3
        and candidate["max_abs_error"] <= 1.25 * reference["max_abs_error"] + 1e-6
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args()
    if not args.cpu_preflight:
        require_tpu_rounding()
        DistributedConfig().initialize()
        if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
            raise ValueError("Keep TPU artifacts in us-central2")
    devices = jax.devices()
    platform = "cpu" if args.cpu_preflight else "tpu"
    if len(devices) != 16 or any(d.platform != platform for d in devices):
        raise ValueError("Use 16 CPU devices for preflight or one TPU v4-32 slice")
    if not args.cpu_preflight and any(d.device_kind != "TPU v4" for d in devices):
        raise ValueError("The frozen screen uses TPU v4")
    rank = jax.process_index()
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 16, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    name = "cpu" if args.cpu_preflight else f"rank-{rank}"
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp/belay-column-embedding")) / name
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(f"{args.artifact_prefix}/{name}")
    fs.makedirs(prefix, exist_ok=True)
    paths = [
        Path(__file__),
        Path(__file__).with_name("column_sharded_embedding.py"),
        Path(__file__).with_name("model.py"),
    ]
    report = dict(
        issue_url="https://github.com/WhenWen/astra-d8-research/issues/96",
        protocol_url="https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6172765675",
        revision=args.revision,
        source_sha256={str(p.relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        rank=rank,
        process_count=jax.process_count(),
        device_count=len(devices),
        device_type=devices[0].device_kind,
        mesh=dict(mesh.shape),
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        numpy_version=np.__version__,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        cpu_preflight=args.cpu_preflight,
        status="running",
        compilation=[],
        cases=[],
    )

    def save():
        path = folder / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    def compile_functions(inputs, shape, component):
        compiled = {}
        table_spec = P(_FSDP_AXES, None)
        token_spec = P(_BATCH_AXES, None, None)
        output_spec = (
            (token_spec, table_spec)
            if component == "complete"
            else (token_spec if component == "forward" else table_spec)
        )
        for implementation, core in IMPLEMENTATIONS.items():
            label = f"{shape[0]}-{component}-{implementation}"
            function = jax.jit(partial(evaluate, core=core, component=component), out_shardings=output_spec)
            start = time.perf_counter()
            executable = function.lower(*inputs).compile()
            compiled[implementation] = executable
            hlo = folder / f"{label}.hlo.txt"
            hlo.write_text(executable.as_text())
            fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
            report["compilation"].append(
                dict(label=label, seconds=time.perf_counter() - start, memory=str(executable.memory_analysis()))
            )
        save()
        return compiled

    save()
    if rank == 0:
        for path in paths:
            fs.put_file(str(path), f"{prefix}/{path.name}")
    with jax.set_mesh(mesh):
        for shape in SHAPES[:1] if args.cpu_preflight else SHAPES:
            compiled = None
            for seed in range(3):
                for distribution in DISTRIBUTIONS:
                    label = f"{shape[0]}-{seed}-{distribution}"
                    arrays = fixture(shape, seed, distribution)
                    if rank == 0:
                        path = folder / f"{label}-fixture.npz"
                        np.savez_compressed(path, **arrays)
                        fs.put_file(str(path), f"{prefix}/{path.name}")
                    inputs, expected = make_arrays(shape, seed, arrays, mesh)
                    jax.block_until_ready((inputs, expected))
                    if compiled is None:
                        compiled = compile_functions(inputs, shape, "complete")
                    results = {name: executable(*inputs) for name, executable in compiled.items()}
                    metrics = {
                        name: jax.tree.map(lambda x: np.asarray(x).item(), numerical_metrics(result, expected))
                        for name, result in results.items()
                    }
                    mismatches = int(forward_mismatches(results["reference"][0], results["column"][0]))
                    case = dict(
                        shape=shape,
                        seed=seed,
                        distribution=distribution,
                        metrics=metrics,
                        pair_forward_mismatches=mismatches,
                        numerical_pass=numerical_gate(metrics, mismatches),
                    )
                    del results
                    if distribution == "uniform" and not args.cpu_preflight:
                        case["timing"] = paired_timings(compiled, inputs, label)
                        medians = case["timing"]["worst_host_medians"]
                        case["timing_pass"] = medians["column"] <= 0.95 * medians["reference"]
                        if shape == SHAPES[1] and seed == 0:
                            case["components"] = {}
                            for component in ("forward", "backward"):
                                component_compiled = compile_functions(inputs, shape, component)
                                case["components"][component] = paired_timings(
                                    component_compiled, inputs, f"{label}-{component}"
                                )
                    report["cases"].append(case)
                    save()
                    print(json.dumps(dict(case=label, numerical_pass=case["numerical_pass"])), flush=True)
    report["numerical_pass"] = all(c["numerical_pass"] for c in report["cases"])
    report["timing_pass"] = (
        None
        if args.cpu_preflight
        else all(c["timing_pass"] for c in report["cases"] if c["shape"] == SHAPES[1] and c["distribution"] == "uniform")
    )
    report["status"] = "completed"
    save()
    if not report["numerical_pass"]:
        raise ValueError("Embedding numerical gate failed; inspect preserved per-case reports")


if __name__ == "__main__":
    main()
