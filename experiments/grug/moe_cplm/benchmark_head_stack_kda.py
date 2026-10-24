# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Equal-state head-stack correctness and separate TPU forward/backward timing."""

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from experiments.grug.fast_track import model
from experiments.grug.moe import head_stack_kda as stack
from experiments.grug.moe import kda
from experiments.grug.moe_cplm.benchmark_frame_kda import write_record
from experiments.grug.moe_cplm.train_belay import tpu_recurrence


def fixture(length, rank=128, width=128, *, mixed=False, heads=4):
    rng = np.random.default_rng(261023 + length)

    def normal(shape):
        return jnp.asarray(rng.normal(size=shape), jnp.float32)

    def unit(x):
        return x * jax.lax.rsqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)

    shape = (1, length, heads, rank)
    logits = 0.2 * normal((1, length, 3))
    if not mixed:
        logits += jnp.array([0, 0, 6])
    inputs = (
        jax.nn.softmax(logits, -1),
        unit(normal(shape)) / rank**0.5,
        unit(normal(shape)),
        normal((1, length, heads, width)),
        -0.1 * jax.nn.softplus(normal(shape)),
        jax.nn.sigmoid(normal((1, length, heads))),
        jnp.ones((1, length), jnp.float32).at[:, length // 2].set(0),
    )
    initial = 0.1 * normal((1, heads, rank, width))
    return inputs, initial


def operator_inputs(inputs):
    _, q, key, value, g, beta, keep = inputs
    q, key, value, g, beta = (jnp.swapaxes(x, 1, 2) for x in (q, key, value, g, beta))
    segments = jnp.cumsum(1 - keep, axis=-1).astype(jnp.int32)[:, None, :]
    return (q * q.shape[-1] ** 0.5, key, value, g, beta), segments


def native(inputs, state, *, serial=False):
    args, segments = operator_inputs(inputs)
    kwargs = {} if serial else {"chunk_size": model.KDA_CHUNK_SIZE, "scan_impl": "parallel_stacked_split"}
    fn = kda.recurrent_kda if serial else kda.chunk_kda
    output, final = fn(
        *args,
        initial_state=state,
        use_qk_l2norm=False,
        segment_ids=segments,
        **kwargs,
    )
    return jnp.swapaxes(output, 1, 2), final


def candidate(inputs, state, chunk, precision="fp32"):
    args, segments = operator_inputs(inputs)
    output, final = stack.chunk_kda(
        *args,
        actions=inputs[0],
        initial_state=state,
        use_qk_l2norm=False,
        segment_ids=segments,
        chunk_size=chunk,
        matmul_dtype=jnp.bfloat16 if precision == "mixed" else jnp.float32,
    )
    return jnp.swapaxes(output, 1, 2), final


def objective(fn, inputs, state):
    out, final = fn(inputs, state)
    return jnp.sum(jnp.sin(out)) + jnp.sum(jnp.sin(final))


def reference_interface(inputs, state):
    # Native document ids are integer controls, with no mask cotangent.
    return stack.reference((*inputs[:-1], jax.lax.stop_gradient(inputs[-1])), state)


def correctness(chunk):
    result = {}
    fn = lambda x, s: candidate(x, s, chunk)  # noqa: E731
    for mixed in (False, True):
        xs, initial = fixture(128, mixed=mixed)
        actual = jax.jit(jax.value_and_grad(lambda x, s: objective(fn, x, s), argnums=(0, 1)))(xs, initial)
        expected = jax.jit(jax.value_and_grad(lambda x, s: objective(reference_interface, x, s), argnums=(0, 1)))(
            xs, initial
        )
        actual = (*jax.jit(fn)(xs, initial), actual)
        expected = (*jax.jit(reference_interface)(xs, initial), expected)
        errors = []
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            a, b = np.asarray(a), np.asarray(b)
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
            errors.append(float(np.max(np.abs(a - b))))
        result["mixed" if mixed else "near_stay"] = errors
    xs = (jnp.broadcast_to(jnp.array([0, 0, 1], jnp.float32), xs[0].shape), *xs[1:])
    with jax.default_matmul_precision("highest"):
        native_reference = native(xs, initial, serial=True)
    candidate_reference = candidate(xs, initial, chunk)
    result["native_stay_max_abs"] = [
        float(jnp.max(jnp.abs(a - b))) for a, b in zip(candidate_reference, native_reference, strict=True)
    ]
    print(json.dumps({"strict_check_diagnostics": result}), flush=True)
    for a, b in zip(candidate_reference, native_reference, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
    return result


def precision_errors(chunk):
    """Measure BF16 approximation separately; do not weaken strict FP32 checks."""
    result = {}
    for mixed_actions in (False, True):
        xs, initial = fixture(128, mixed=mixed_actions)
        variants = {}
        for precision in ("fp32", "mixed"):
            fn = lambda x, s, p=precision: candidate(x, s, chunk, p)  # noqa: E731
            values = jax.jit(fn)(xs, initial)
            grads = jax.jit(jax.grad(lambda x, s, f=fn: objective(f, x, s), argnums=(0, 1)))(xs, initial)
            variants[precision] = jax.tree.leaves((values, grads))
        errors = []
        for actual, expected in zip(variants["mixed"], variants["fp32"], strict=True):
            actual, expected = np.asarray(actual), np.asarray(expected)
            if not np.isfinite(actual).all():
                raise ValueError("Nonfinite mixed-precision value or adjoint")
            errors.append(
                {
                    "max_abs": float(np.max(np.abs(actual - expected))),
                    "relative_l2": float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)),
                }
            )
        result["mixed_actions" if mixed_actions else "near_stay"] = errors
    return result


def timed(fn, args):
    started = time.perf_counter()
    executable = jax.jit(fn).lower(*args).compile()
    compilation = time.perf_counter() - started
    value = executable(*args)
    if not all(np.isfinite(x).all() for x in jax.tree.leaves(jax.device_get(value))):
        raise ValueError("Nonfinite benchmark output")
    times = []
    for _ in range(7):
        started = time.perf_counter()
        jax.block_until_ready(executable(*args))
        times.append(time.perf_counter() - started)
    memory = executable.memory_analysis()
    return {
        "seconds": times,
        "median_seconds": float(np.median(times)),
        "compile_seconds": compilation,
        "temp_bytes": memory.temp_size_in_bytes,
    }


def benchmark(fn, inputs, initial):
    # Close over the VJP's primal residuals as explicit executable inputs, so
    # backward timing cannot fold them into constants or include forward work.
    output, pullback = jax.vjp(fn, inputs, initial)
    cotangent = jax.tree.map(jnp.cos, output)
    residuals, structure = jax.tree.flatten(pullback)

    def backward(residuals, dy):
        return jax.tree.unflatten(structure, residuals)(dy)

    return {
        "forward": timed(fn, (inputs, initial)),
        "backward": timed(backward, (residuals, cotangent)),
        "forward_backward": timed(
            jax.value_and_grad(lambda x, s: objective(fn, x, s), argnums=(0, 1)), (inputs, initial)
        ),
        "backward_residual_bytes": sum(x.size * x.dtype.itemsize for x in residuals),
        "state_bytes": initial.size * initial.dtype.itemsize,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--precisions", choices=("fp32", "mixed"), nargs="+", default=["fp32", "mixed"])
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    source = Path(__file__).parent
    settings = json.loads((source / "belay_recipe.json").read_text())["trainer"]["trainer"]["jax_config"]
    settings["jax_enable_pgle"] = False
    for name, value in settings.items():
        jax.config.update(name, value)
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
    if any(device.device_kind != "TPU v4" for device in jax.local_devices()):
        raise ValueError("Head-stack qualification requires TPU v4")
    attempt = uuid.uuid4().hex
    output, root = args.output / attempt, args.artifact_root.rstrip("/") + "/" + attempt
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "training_updates": 0,
        "execution_id": attempt,
        "artifact_root": root,
        "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/59",
        "motivating_task": "scoped_binding@1 (#58), not evaluated",
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_settings": settings,
        "shape": {"batch": 1, "length": args.length, "heads_and_depth": 4, "rank": 128, "width": 128},
        "baseline": "native KDA TPU Pallas state pass, mixed precision, C16",
        "candidate": "three shared actions, four slots; FP32 reference and BF16/FP32 hybrid",
        "source_sha256": {
            str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                source.parent / "moe/head_stack_kda.py",
                source.parent / "moe/stack_kda.py",
                source.parent / "moe/kda.py",
                source.parent / "moe/kda_state_tpu.py",
                source / "head_stack_kda_model.py",
                Path(__file__),
                source / "belay_recipe.json",
            )
        },
    }
    write_record(output, root, "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    with jax.default_device(jax.local_devices()[0]), patch.object(kda, "_parallel_chunk_recurrence", tpu_recurrence):
        inputs, initial = fixture(args.length)
        baseline = benchmark(native, inputs, initial)
        write_record(output, root, "baseline.json", baseline)
        print(json.dumps({"baseline": baseline}), flush=True)
        for chunk in args.chunks:
            checks = {"errors": correctness(chunk), "precision_errors": precision_errors(chunk)}
            write_record(output, root, f"checks-chunk-{chunk}.json", checks)
            print(json.dumps({"chunk": chunk, **checks}), flush=True)
            for precision in args.precisions:
                jax.clear_caches()
                result = {"chunk": chunk, "precision": precision}
                result.update(benchmark(lambda x, s, c=chunk, p=precision: candidate(x, s, c, p), inputs, initial))
                write_record(output, root, f"precision-{precision}-chunk-{chunk}.json", result)
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
