# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training checks against independent stack branch enumeration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe import kda, stack_kda
from experiments.grug.moe_cplm.test_frame_kda import fixture, reference


def inputs_and_state():
    xs, initial = fixture(depth=4, rank=3, width=2)
    inputs = tuple(jnp.swapaxes(x, 1, 2) for x in (*xs[:6], xs[-1]))
    return inputs, jnp.swapaxes(initial, -3, -2)


def branch_reference(inputs, initial):
    xs = tuple(jnp.swapaxes(x, 1, 2) for x in inputs)
    address = jnp.broadcast_to(jnp.array([1, 0, 0, 0], jnp.float32), (*xs[0].shape[:-1], 4))
    output, final = reference((*xs[:6], address, xs[-1]), jnp.swapaxes(initial, -3, -2))
    return jnp.swapaxes(output, 1, 2), jnp.swapaxes(final, -3, -2)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("chunk", [2, 3, 6])
def test_structured_values_and_all_adjoints(chunk):
    inputs, initial = inputs_and_state()

    def fast(xs, state):
        return stack_kda.chunked(xs, state, chunk)

    def loss(fn, xs, state):
        output, final = fn(xs, state)
        return jnp.sum(jnp.sin(output)) + jnp.sum(jnp.sin(final))

    for got, want in zip(jax.jit(fast)(inputs, initial), branch_reference(inputs, initial), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-4)
    actual = jax.jit(jax.grad(lambda xs, s: loss(fast, xs, s), argnums=(0, 1)))(inputs, initial)
    expected = jax.jit(jax.grad(lambda xs, s: loss(branch_reference, xs, s), argnums=(0, 1)))(inputs, initial)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-4)


def test_streaming_causality_and_document_reset():
    inputs, initial = inputs_and_state()
    full, final = stack_kda.chunked(inputs, initial, 3)
    first, middle = stack_kda.chunked(tuple(x[:, :, :6] for x in inputs), initial, 3)
    second, streamed_final = stack_kda.chunked(tuple(x[:, :, 6:] for x in inputs), middle, 3)
    np.testing.assert_allclose(jnp.concatenate((first, second), axis=2), full, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(streamed_final, final, atol=1e-5, rtol=1e-4)
    changed = list(inputs)
    changed[3] = inputs[3].at[:, :, 9:].add(100)
    future, _ = stack_kda.chunked(tuple(changed), initial, 3)
    np.testing.assert_allclose(future[:, :, :9], full[:, :, :9], atol=1e-5, rtol=1e-4)
    changed[3] = inputs[3].at[0, :, :5].add(100)
    reset, _ = stack_kda.chunked(tuple(changed), initial, 3)
    np.testing.assert_allclose(reset[0, :, 5:], full[0, :, 5:], atol=1e-5, rtol=1e-4)


def test_hard_pop_restores_parent_without_child_decay():
    actions = jnp.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], jnp.float32)[None, None]
    qk = jnp.ones((1, 1, 4, 1), jnp.float32)
    values = jnp.array([2, 7, 99, 0], jnp.float32).reshape(1, 1, 4, 1)
    decay = jnp.array([1, 1, 0.01, 1], jnp.float32).reshape(1, 1, 4, 1)
    beta = jnp.array([1, 1, 1, 0], jnp.float32).reshape(1, 1, 4)
    xs = (actions, qk, qk, values, decay, beta, jnp.ones((1, 1, 4)))
    output, _ = stack_kda.chunked(xs, jnp.zeros((1, 1, 1, 4, 1)), 2)
    np.testing.assert_allclose(output.reshape(-1), [2, 7, 2, 2], atol=1e-6)


def test_hard_stay_native_normalization_and_packed_documents():
    inputs, _ = inputs_and_state()
    _, q, k, v, decay, beta, _ = inputs
    segments = jnp.broadcast_to(jnp.arange(12) // 4, (2, 12))
    actions = jnp.broadcast_to(jnp.array([0, 0, 1, 0], jnp.float32), (2, 12, 2, 4))
    args = tuple(jnp.swapaxes(x, 1, 2) for x in (q, k, v, jnp.log(decay), beta))
    got = stack_kda.native_kernel(*args, actions, segments, chunk=3)
    want, _ = kda.recurrent_kda(q, k, v, jnp.log(decay), beta, segment_ids=segments[:, None])
    np.testing.assert_allclose(jnp.swapaxes(got, 1, 2), want, atol=1e-5, rtol=1e-4)
