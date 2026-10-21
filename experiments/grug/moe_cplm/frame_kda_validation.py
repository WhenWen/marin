# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero-update device checks and timings for the protected-frame recurrence."""

import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_cplm import frame_kda


def validate_frame_kda(sequence_length: int, *, chunk: int | None = None):
    """Check values and all adjoints, then measure a native-length local shard."""
    frame = frame_kda.FRAME_CONFIG
    heads, depth, rank, width = frame.heads, frame.depth, frame.rank, frame.width
    chunk = frame.chunk if chunk is None else chunk
    check_length = 4 * frame.chunk
    if chunk <= 0 or check_length % chunk or sequence_length % chunk:
        raise ValueError("Chunk size must divide both the reference length and native sequence length")
    rng = np.random.default_rng(261021)

    def make_inputs(length):
        def normal(size):
            return jnp.asarray(rng.normal(size=(1, length, heads, size)), jnp.float32)

        def unit(x):
            return x * jax.lax.rsqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)

        return (
            jax.nn.softmax(normal(4), -1),
            unit(normal(rank)),
            unit(normal(rank)),
            normal(width),
            jax.nn.sigmoid(normal(rank)),
            jax.nn.sigmoid(normal(1)[..., 0]),
            jax.nn.softmax(normal(depth), -1),
            jnp.ones((1, length, heads)).at[:, length // 2].set(0),
        )

    def reference(xs, state):
        final, out = frame_kda.recurrent(tuple(jnp.swapaxes(x, 0, 1) for x in xs), state)
        return jnp.swapaxes(out, 0, 1), final

    def objective(fn, xs, state):
        output, final = fn(xs, state)
        return jnp.sum(jnp.sin(output)) + jnp.sum(jnp.sin(final))

    def fast(xs, state):
        return frame_kda.chunked(xs, state, chunk)

    with jax.default_device(jax.local_devices()[0]):
        initial = jnp.asarray(rng.normal(size=(1, heads, depth, rank, width)), jnp.float32)
        # Fixed length keeps the reference and native random inputs identical
        # when comparing different parallel schedules.
        xs = make_inputs(check_length)
        fused = jax.jit(jax.value_and_grad(lambda x, s: objective(fast, x, s), argnums=(0, 1)))
        serial = jax.jit(jax.value_and_grad(lambda x, s: objective(reference, x, s), argnums=(0, 1)))
        got, want = jax.device_get(fused(xs, initial)), jax.device_get(serial(xs, initial))
        errors = []
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
            errors.append(float(np.max(np.abs(a - b))))
        serial.clear_cache()
        fused.clear_cache()
        xs = make_inputs(sequence_length)
        # Compile once, warm once, and drain every measured execution.
        started = time.perf_counter()
        executable = fused.lower(xs, initial).compile()
        compile_seconds = time.perf_counter() - started
        native_result = jax.device_get(executable(xs, initial))
        if not all(np.isfinite(x).all() for x in jax.tree.leaves(native_result)):
            raise ValueError("Nonfinite native-length objective or gradient")
        durations = []
        for _ in range(5):
            started = time.perf_counter()
            jax.block_until_ready(executable(xs, initial))
            durations.append(time.perf_counter() - started)
        memory = executable.memory_analysis()
        result = {
            "training_updates": 0,
            "device": jax.local_devices()[0].device_kind,
            "shape": [1, sequence_length, heads, depth, rank, width],
            "chunk": chunk,
            "reference_length": check_length,
            "value_and_gradient_max_errors": errors,
            "native_objective": float(native_result[0]),
            "native_inputs_sha256": hashlib.sha256(
                b"".join(np.asarray(x).tobytes() for x in jax.tree.leaves(jax.device_get((xs, initial))))
            ).hexdigest(),
            "compile_seconds": compile_seconds,
            "forward_backward_seconds": durations,
            "compiled_temp_bytes": memory.temp_size_in_bytes,
            "compiled_argument_bytes": memory.argument_size_in_bytes,
            "compiled_output_bytes": memory.output_size_in_bytes,
            "xla_cost_analysis": executable.cost_analysis(),
        }
        fused.clear_cache()
        return result
