# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare fused CE forward/backward paths without an optimizer or training loop.

Protocol: https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6171525144
"""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from functools import lru_cache, partial
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from levanter.kernels.pallas.fused_cross_entropy_loss.config import BlockSizes
from levanter.kernels.pallas.fused_cross_entropy_loss.pallas_tpu import (
    linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu,
    linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu,
    linear_softmax_cross_entropy_loss_pallas,
)
from levanter.kernels.pallas.fused_cross_entropy_loss.xla import (
    _linear_softmax_cross_entropy_loss_streaming_bwd_scan,
    linear_softmax_cross_entropy_loss_xla,
)

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.benchmark_inkling_block_inputs_tpu import finite_errors

SOFT_CAP = 10.0
Z_LOSS = 1e-4
IMPLEMENTATIONS = ("reference", "Pallas-512", "Hybrid-512", "Hybrid-1024")
INPUTS = ("x", "w", "labels", "weights")
TENSORS = ("ce", "lse", "objective", "dx", "dw")


def fast_backward(x, labels, w, lse, dce, dlse):
    return _linear_softmax_cross_entropy_loss_streaming_bwd_scan(
        x, labels, w, lse, dce, dlse,
        block_size=min(8192, w.shape[1] // 2),
        dtype=jnp.float32,
        batch_block_size=x.shape[0],
        logit_soft_cap=SOFT_CAP,
        precision=None,
    )


@lru_cache(maxsize=None)
def hybrid_core(vocab_tile):
    """Pair the existing Pallas forward with the existing BF16-delta backward."""
    forward = partial(
        linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu,
        block_sizes=BlockSizes(1024, 512, vocab_tile),
        dtype=jnp.float32,
        logit_soft_cap=SOFT_CAP,
    )

    @jax.custom_vjp
    def core(x, labels, w):
        return forward(x, labels, w)

    def fwd(x, labels, w):
        ce, lse = forward(x, labels, w)
        return (ce, lse), (x, labels, w, lse)

    def bwd(residual, cotangents):
        dx, dw = fast_backward(*residual, *cotangents)
        return dx, None, dw

    core.defvjp(fwd, bwd)
    return core


def reference(x, labels, w):
    return linear_softmax_cross_entropy_loss_xla(
        x, labels, w,
        block_sizes=BlockSizes(65536, 512, min(8192, w.shape[1] // 2)),
        dtype=jnp.float32,
        logit_soft_cap=SOFT_CAP,
        fast_backward=True,
    )


def oracle64(x, labels, w):
    """Dense independent oracle, called only with FP64 operands on CPU."""
    logits = SOFT_CAP * jnp.tanh((x @ w) / SOFT_CAP)
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    ce = lse - logits[jnp.arange(x.shape[0]), labels]
    return ce, lse


def objective(x, w, labels, weights, *, core):
    ce, lse = core(x, labels, w)
    loss = jnp.sum(weights * (ce + Z_LOSS * lse**2)) / jnp.sum(weights)
    return loss, (ce, lse)


def values_and_gradients(x, w, labels, weights, *, core):
    (loss, (ce, lse)), (dx, dw) = jax.value_and_grad(
        partial(objective, core=core), argnums=(0, 1), has_aux=True
    )(x, w, labels, weights)
    return ce, lse, loss, dx, dw


@lru_cache(maxsize=None)
def core_functions():
    return {
        "reference": reference,
        "Pallas-512": partial(
            linear_softmax_cross_entropy_loss_pallas,
            block_sizes=BlockSizes(1024, 512, 512),
            dtype=jnp.float32,
            logit_soft_cap=SOFT_CAP,
        ),
        "Hybrid-512": hybrid_core(512),
        "Hybrid-1024": hybrid_core(1024),
    }


def backward_component(x, w, labels, weights, lse, *, implementation):
    dce = weights / jnp.sum(weights)
    dlse = 2 * Z_LOSS * lse * dce
    if implementation == "Pallas-512":
        return linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
            dce, dlse, lse, x, labels, w,
            block_sizes=BlockSizes(1024, 512, 512),
            dtype=jnp.float32,
            logit_soft_cap=SOFT_CAP,
        )
    return fast_backward(x, labels, w, lse, dce, dlse)


def fixture(shape, seed, scale, weighting):
    n, h, v = shape
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n, h)) * scale).astype(jnp.bfloat16)
    w = (rng.standard_normal((h, v)) / np.sqrt(h)).astype(jnp.bfloat16)
    labels = rng.integers(0, v, size=n, dtype=np.int32)
    fractional = rng.uniform(0.25, 1.0, size=n).astype(np.float32)
    labels[:4] = (0, v - 1, v // 2 - 1, v // 2)
    weights = np.ones(n, dtype=np.float32)
    if weighting == "boundary":
        weights[1023::1024] = 0
    elif weighting == "fractional":
        weights = fractional
    elif weighting != "ones":
        raise ValueError(weighting)
    return x, w, labels, weights


def numerical_gate(candidate, baseline):
    return all(
        candidate[key]["finite"]
        and baseline[key]["finite"]
        and candidate[key]["relative_l2"] <= 1.25 * baseline[key]["relative_l2"] + 1e-3
        and candidate[key]["max_abs_error"] <= 1.25 * baseline[key]["max_abs_error"] + (
            1e-6 if key in ("dx", "dw") else 1e-4
        )
        for key in TENSORS
    )


def measurement_order(names, repeat):
    shift = repeat % len(names)
    order = names[shift:] + names[:shift]
    return order[::-1] if (repeat // len(names)) % 2 else order


@jax.jit
def all_finite(values):
    return jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(values)]).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args()
    for key in ("LEVANTER_CE_XLA_FAST_BWD", "LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH"):
        if os.environ.get(key) is not None:
            raise ValueError(f"Remove backend override {key}")
    cpu = jax.local_devices(backend="cpu")[0]
    if args.cpu_preflight:
        device = cpu
    else:
        require_tpu_rounding()
        device = jax.local_devices(backend="tpu")[0]
        if device.device_kind != "TPU v4" or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
            raise ValueError("Use TPU v4 and us-central2 artifacts")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp/belay-fused-ce")) / (
        "cpu" if args.cpu_preflight else "tpu"
    )
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    fs.makedirs(prefix, exist_ok=True)
    source_paths = (
        Path(__file__).relative_to(Path.cwd()),
        Path("lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py"),
        Path("lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py"),
        Path("lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/reference.py"),
    )
    report = {
        "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/96",
        "protocol_url": "https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6171525144",
        "revision": args.revision,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "status": "running",
        "cpu_preflight": args.cpu_preflight,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "device_type": device.device_kind,
        "device_count_used": 1,
        "local_device_count_available": len(jax.local_devices(backend=device.platform)),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "numerics": [], "compilation": [], "compilation_failures": [], "fixtures": [],
        "warmups": [], "timings": [], "performance": [], "components": [],
    }
    disabled = set()
    compiled = {}
    functions = {
        name: jax.jit(partial(values_and_gradients, core=core))
        for name, core in core_functions().items()
        if not args.cpu_preflight or name == "reference"
    }
    oracle_fn = jax.jit(partial(values_and_gradients, core=oracle64))

    def upload(path):
        fs.put_file(str(path), f"{prefix}/{path.name}")

    def save():
        path = folder / "fused-ce.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        upload(path)

    def preserve_fixture(host, case):
        stem = "-".join(str(case[key]) for key in ("n", "h", "v", "seed", "scale", "weighting"))
        path = folder / f"fixture-{stem}.npz"
        # NumPy represents BF16 as opaque bytes in NPZ; save raw uint16 explicitly.
        np.savez(path, x_bits=host[0].view(np.uint16), w_bits=host[1].view(np.uint16),
                 labels=host[2], weights=host[3])
        upload(path)
        report["fixtures"].append({
            **case, "path": path.name,
            "sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                       for name, value in zip(INPUTS, host, strict=True)},
        })

    def executable(name, phase, fn, inputs, case):
        key = (name, phase, case["n"], case["h"], case["v"])
        if name in disabled:
            return None
        if key in compiled:
            return compiled[key]
        report["phase"] = {**case, "implementation": name, "stage": f"compile-{phase}"}
        save()
        start = time.monotonic()
        try:
            result = fn.lower(*inputs).compile()
        except Exception as error:
            # Compilation failures disqualify this implementation, not its peers.
            report["compilation_failures"].append({
                **report["phase"], "error": repr(error), "traceback": traceback.format_exc(),
            })
            disabled.add(name)
            save()
            if name == "reference":
                raise
            return None
        memory = result.memory_analysis()
        report["compilation"].append({
            **report["phase"], "seconds": time.monotonic() - start,
            "temporary_bytes": memory.temp_size_in_bytes,
            "argument_bytes": memory.argument_size_in_bytes,
            "output_bytes": memory.output_size_in_bytes,
            "alias_bytes": memory.alias_size_in_bytes,
        })
        path = folder / f"{name}-{phase}-{case['n']}-{case['h']}-{case['v']}-hlo.txt"
        path.write_text(result.as_text())
        upload(path)
        compiled[key] = result
        save()
        return result

    def time_case(available, inputs_by_name, case, phase):
        names = list(available)
        for name in names:
            for repeat in range(2):
                values = jax.block_until_ready(available[name](*inputs_by_name[name]))
                finite = bool(all_finite(values))
                report["warmups"].append({**case, "phase": phase, "implementation": name,
                                          "repeat": repeat, "finite": finite})
                if not finite:
                    save()
                    raise ValueError(f"Nonfinite {phase} outputs from {name}: {case}")
        samples = {name: [] for name in names}
        for repeat in range(8):
            for name in measurement_order(names, repeat):
                start = time.monotonic()
                jax.block_until_ready(available[name](*inputs_by_name[name]))
                elapsed = time.monotonic() - start
                samples[name].append(elapsed)
                report["timings"].append({**case, "phase": phase, "implementation": name,
                                          "repeat": repeat, "seconds": elapsed})
            save()
        return {name: statistics.median(values) for name, values in samples.items()}

    save()
    try:
        for shape in ((1024, 512, 2048), (2048, 512, 4096), (1024, 512, 16384)):
            for seed in (0, 1, 2):
                for scale in (1, 8):
                    for weighting in ("ones", "boundary", "fractional"):
                        case = dict(zip(("n", "h", "v"), shape, strict=True)) | {
                            "seed": seed, "scale": scale, "weighting": weighting,
                        }
                        host = fixture(shape, seed, scale, weighting)
                        preserve_fixture(host, case)
                        with jax.enable_x64(True), jax.default_device(cpu):
                            host64 = tuple(x.astype(np.int32 if i == 2 else np.float64)
                                           for i, x in enumerate(host))
                            oracle = jax.device_get(oracle_fn(*(jax.device_put(x, cpu) for x in host64)))
                        actuals, deviations = {}, {}
                        with jax.default_device(device):
                            inputs = tuple(jax.device_put(x, device) for x in host)
                            for name, fn in functions.items():
                                run = executable(name, "complete", fn, inputs, case)
                                if run is None:
                                    continue
                                actuals[name] = jax.device_get(jax.block_until_ready(run(*inputs)))
                                deviations[name] = {
                                    key: finite_errors(actual, expected)
                                    for key, actual, expected in zip(TENSORS, actuals[name], oracle, strict=True)
                                }
                        row = {**case, "errors": deviations, "direct_errors": {}, "passed": {}}
                        for name in deviations:
                            row["passed"][name] = numerical_gate(deviations[name], deviations["reference"])
                            if name != "reference":
                                row["direct_errors"][name] = {
                                    key: finite_errors(actual, expected)
                                    for key, actual, expected in zip(
                                        TENSORS, actuals[name], actuals["reference"], strict=True
                                    )
                                }
                        report["numerics"].append(row)
                        save()
                        print(json.dumps({**case, "passed": row["passed"]}), flush=True)
        if not args.cpu_preflight:
            with jax.default_device(device):
                for n in (1024, 32768):
                    for seed in (10, 20, 30):
                        for scale, weighting in ((1, "ones"), (1, "boundary"), (8, "ones")):
                            case = {"n": n, "h": 512, "v": 16384, "seed": seed,
                                    "scale": scale, "weighting": weighting}
                            host = fixture((n, 512, 16384), seed, scale, weighting)
                            preserve_fixture(host, case)
                            inputs = tuple(jax.device_put(x, device) for x in host)
                            available = {}
                            for name, fn in functions.items():
                                run = executable(name, "complete", fn, inputs, case)
                                if run is not None:
                                    available[name] = run
                            medians = time_case(available, dict.fromkeys(available, inputs), case, "complete")
                            result = {**case, "medians": medians, "reductions": {
                                name: 1 - value / medians["reference"] for name, value in medians.items()
                                if name != "reference"
                            }}
                            report["performance"].append(result)
                            print(json.dumps(result), flush=True)
                            save()
                            if (n, seed, scale, weighting) == (32768, 10, 1, "ones"):
                                for phase in ("forward", "backward"):
                                    component_runs, component_inputs = {}, {}
                                    for name in available:
                                        if phase == "forward":
                                            fn = jax.jit(partial(objective, core=core_functions()[name]))
                                            values = inputs
                                        else:
                                            lse = jax.block_until_ready(available[name](*inputs))[1]
                                            fn = jax.jit(partial(backward_component, implementation=name))
                                            values = (*inputs, lse)
                                        run = executable(name, phase, fn, values, case)
                                        if run is not None:
                                            component_runs[name] = run
                                            component_inputs[name] = values
                                    medians = time_case(component_runs, component_inputs, case, phase)
                                    report["components"].append({**case, "phase": phase, "medians": medians})
                                    save()
        report["qualification"] = {
            name: (
                name not in disabled
                and len(report["numerics"]) == 54
                and all(row["passed"].get(name, False) for row in report["numerics"])
                and not args.cpu_preflight
                and len([row for row in report["performance"] if row["n"] == 32768]) == 9
                and all(row["reductions"].get(name, -1) >= 0.05
                        for row in report["performance"] if row["n"] == 32768)
            ) for name in IMPLEMENTATIONS if name != "reference"
        }
        report["status"] = "completed"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
