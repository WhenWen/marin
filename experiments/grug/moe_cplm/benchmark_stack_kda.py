# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify depth-four KDA against the native TPU recurrence without training."""

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
from experiments.grug.moe import kda, stack_kda
from experiments.grug.moe_cplm.benchmark_frame_kda import write_record
from experiments.grug.moe_cplm.stack_kda_validation import fixture, objective, validate_stack_kda
from experiments.grug.moe_cplm.train_belay import tpu_recurrence


def native_baseline(sequence_length):
    def baseline(xs, state):
        _, q, key, value, decay, beta, keep = xs
        segments = jnp.cumsum(1 - keep, axis=-1).astype(jnp.int32)
        return kda.chunk_kda(
            q * q.shape[-1] ** 0.5,
            key,
            value,
            jnp.log(decay),
            beta,
            initial_state=state[..., 0, :],
            use_qk_l2norm=False,
            segment_ids=segments,
            chunk_size=model.KDA_CHUNK_SIZE,
            scan_impl="parallel_stacked_split",
        )

    with jax.default_device(jax.local_devices()[0]), patch.object(kda, "_parallel_chunk_recurrence", tpu_recurrence):
        inputs, initial = fixture(sequence_length)
        fn = jax.jit(jax.value_and_grad(lambda x, s: objective(baseline, x, s), argnums=(0, 1)))
        started = time.perf_counter()
        executable = fn.lower(inputs, initial).compile()
        compilation = time.perf_counter() - started
        values = jax.device_get(executable(inputs, initial))
        if not all(np.isfinite(x).all() for x in jax.tree.leaves(values)):
            raise ValueError("Native baseline benchmark is nonfinite")
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            jax.block_until_ready(executable(inputs, initial))
            timings.append(time.perf_counter() - started)
        forward = jax.jit(baseline).lower(inputs, initial).compile()
        jax.block_until_ready(forward(inputs, initial))
        forward_times = []
        for _ in range(5):
            started = time.perf_counter()
            jax.block_until_ready(forward(inputs, initial))
            forward_times.append(time.perf_counter() - started)
        return {
            "forward_backward_seconds": timings,
            "forward_seconds": forward_times,
            "compile_seconds": compilation,
            "compiled_temp_bytes": executable.memory_analysis().temp_size_in_bytes,
            "chunk": model.KDA_CHUNK_SIZE,
            "execution": "native TPU Pallas state_pass; inherited mixed precision",
        }


def phase_timings(sequence_length, chunk):
    with jax.default_device(jax.local_devices()[0]):
        inputs, initial = fixture(sequence_length)
        prepare = lambda x, s: stack_kda._prepare(stack_kda._split(x, chunk))  # noqa: E731

        def boundaries(prep, state):
            axis = state.ndim - 3
            blocks = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0), prep)

            def step(state, block):
                final, _ = stack_kda._advance(block, state)
                return final, state

            final, incoming = jax.lax.scan(step, jnp.swapaxes(state, -3, -2), blocks)
            return final, jnp.moveaxis(incoming, 0, axis)

        def readout(prep, state):
            _, innovation = stack_kda._advance(prep, state)
            return stack_kda._emit(prep, state, innovation)

        prepared = jax.jit(prepare)(inputs, initial)
        _, incoming = jax.jit(boundaries)(prepared, initial)
        result = {}
        for name, fn, args in (
            ("prepare", prepare, (inputs, initial)),
            ("boundaries", boundaries, (prepared, initial)),
            ("readout", readout, (prepared, incoming)),
            ("forward", lambda x, s: stack_kda.chunked(x, s, chunk), (inputs, initial)),
        ):
            executable = jax.jit(fn).lower(*args).compile()
            jax.block_until_ready(executable(*args))
            times = []
            for _ in range(5):
                started = time.perf_counter()
                jax.block_until_ready(executable(*args))
                times.append(time.perf_counter() - started)
            result[name] = {"seconds": times, "temp_bytes": executable.memory_analysis().temp_size_in_bytes}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, nargs="+", default=[32])
    parser.add_argument("--backward-batches", type=int, nargs="+", default=[stack_kda.BACKWARD_BATCH])
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
        raise ValueError("Stack KDA performance qualification requires TPU v4")
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
        "source_sha256": {
            str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                source.parent / "moe/stack_kda.py",
                source / "stack_kda_validation.py",
                source / "stack_kda_model.py",
                source / "benchmark_stack_kda.py",
                source / "train_belay.py",
                source / "belay_recipe.json",
            )
        },
    }
    write_record(output, root, "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    baseline = native_baseline(args.length)
    write_record(output, root, "baseline.json", baseline)
    print(json.dumps({"baseline": baseline}), flush=True)
    for batch in args.backward_batches:
        stack_kda.BACKWARD_BATCH = batch
        for chunk in args.chunks:
            jax.clear_caches()
            result = validate_stack_kda(args.length, chunk)
            result["phases"] = phase_timings(args.length, chunk)
            write_record(output, root, f"batch-{batch}-chunk-{chunk}.json", result)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
