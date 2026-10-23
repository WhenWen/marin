# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training numerical and device qualification for layer-internal stack KDA."""

import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe import stack_kda
from experiments.grug.moe_cplm import frame_kda


def reference(inputs, initial):
    actions, query, key, value, decay, beta, keep = (jnp.swapaxes(x, 1, 2) for x in inputs)
    address = jnp.broadcast_to(jnp.array([1, 0, 0, 0], jnp.float32), actions.shape)
    xs = (actions, query, key, value, decay, beta, address, keep)
    final, output = frame_kda.recurrent(
        tuple(jnp.swapaxes(x, 0, 1) for x in xs), jnp.swapaxes(initial, -3, -2)
    )
    return jnp.transpose(output, (1, 2, 0, 3)), jnp.swapaxes(final, -3, -2)


def fixture(length, rank=128, width=128, heads=4):
    rng = np.random.default_rng(261023 + length)

    def normal(size):
        return jnp.asarray(rng.normal(size=(1, heads, length, size)), jnp.float32)

    def unit(x):
        return x * jax.lax.rsqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)

    inputs = (
        jax.nn.softmax(0.2 * normal(4) + jnp.array([0, 0, 6, 0]), -1),
        unit(normal(rank)) / rank**0.5,
        unit(normal(rank)),
        normal(width),
        jnp.exp(-0.1 * jax.nn.softplus(normal(rank))),
        jax.nn.sigmoid(normal(1)[..., 0]),
        jnp.ones((1, heads, length), jnp.float32).at[:, :, length // 2].set(0),
    )
    initial = jnp.asarray(0.1 * rng.normal(size=(1, heads, rank, 4, width)), jnp.float32)
    return inputs, initial


def objective(fn, inputs, state):
    output, final = fn(inputs, state)
    return jnp.sum(jnp.sin(output)) + jnp.sum(jnp.sin(final))


def validate_stack_kda(sequence_length=8192, chunk=stack_kda.CHUNK):
    """Compare all reference adjoints, then time five drained native-shape passes."""
    def fast(xs, state):
        return stack_kda.chunked(xs, state, chunk)

    with jax.default_device(jax.local_devices()[0]):
        inputs, initial = fixture(128)
        check = jax.jit(jax.value_and_grad(lambda x, s: objective(fast, x, s), argnums=(0, 1)))
        oracle = jax.jit(jax.value_and_grad(lambda x, s: objective(reference, x, s), argnums=(0, 1)))
        got, want = jax.device_get(check(inputs, initial)), jax.device_get(oracle(inputs, initial))
        errors = []
        for actual, expected in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
            errors.append(float(np.max(np.abs(actual - expected))))
        for actual, expected in zip(jax.jit(fast)(inputs, initial), reference(inputs, initial), strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        oracle.clear_cache()
        check.clear_cache()
        inputs, initial = fixture(sequence_length)
        digest = hashlib.sha256()
        for item in jax.tree.leaves((inputs, initial)):
            digest.update(np.asarray(item).tobytes())
        started = time.perf_counter()
        executable = check.lower(inputs, initial).compile()
        compile_seconds = time.perf_counter() - started
        result = jax.device_get(executable(inputs, initial))
        if not all(np.isfinite(x).all() for x in jax.tree.leaves(result)):
            raise ValueError("Stack KDA native forward or adjoints are nonfinite")
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            jax.block_until_ready(executable(inputs, initial))
            timings.append(time.perf_counter() - started)
        memory = executable.memory_analysis()
        return {
            "training_updates": 0,
            "shape": {"batch": 1, "heads": 4, "length": sequence_length, "rank": 128, "depth": 4, "width": 128},
            "chunk": chunk,
            "reference_length": 128,
            "reference_max_abs_errors": errors,
            "native_inputs_sha256": digest.hexdigest(),
            "native_objective": float(result[0]),
            "forward_backward_seconds": timings,
            "compile_seconds": compile_seconds,
            "compiled_temp_bytes": memory.temp_size_in_bytes,
            "device": jax.local_devices()[0].device_kind,
        }
