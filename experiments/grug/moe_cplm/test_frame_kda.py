# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training value, adjoint, causality and save/restore checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe_cplm.frame_kda import chunked
from experiments.grug.moe_cplm.frame_kda_model import local_read


def reference(inputs, initial):
    """Explicitly form four possible stacks and average their outcomes."""

    def token(state, xs):
        action, query, key, value, decay, beta, address, keep = xs
        state = state * keep[..., None, None, None]
        rank = key.shape[-1]
        matrix = jnp.eye(rank) - beta[..., None, None] * key[..., :, None] * key[..., None, :]
        updated = jnp.einsum("...ij,...jv->...iv", matrix, decay[..., :, None] * state[..., 0, :, :])
        write = beta[..., None, None] * key[..., :, None] * value[..., None, :]
        updated = updated + write
        push = jnp.concatenate((updated[..., None, :, :], state[..., :-1, :, :]), axis=-3)
        pop = jnp.concatenate((state[..., 1:, :, :], jnp.zeros_like(state[..., :1, :, :])), axis=-3)
        stay = jnp.concatenate((updated[..., None, :, :], state[..., 1:, :, :]), axis=-3)
        replace = jnp.concatenate((write[..., None, :, :], state[..., 1:, :, :]), axis=-3)
        state = sum(action[..., i, None, None, None] * branch for i, branch in enumerate((push, pop, stay, replace)))
        read = jnp.sum(state * address[..., :, None, None], axis=-3)
        output = jnp.sum(read * query[..., :, None], axis=-2)
        return state, output

    final, outputs = jax.lax.scan(token, initial, tuple(jnp.swapaxes(x, 0, 1) for x in inputs))
    return jnp.swapaxes(outputs, 0, 1), final


def fixture(length=12, depth=3, rank=2, width=3):
    rng = np.random.default_rng(128)
    batch, heads = 2, 2

    def normal(size):
        return jnp.asarray(rng.normal(size=(batch, length, heads, size)), jnp.float32)

    def unit(x):
        return x / jnp.sqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)

    inputs = (
        jax.nn.softmax(normal(4), -1),
        unit(normal(rank)),
        unit(normal(rank)),
        normal(width),
        jax.nn.sigmoid(normal(rank)),
        jax.nn.sigmoid(normal(1)[..., 0]),
        jax.nn.softmax(normal(depth), -1),
        jnp.ones((batch, length, heads), jnp.float32),
    )
    keep = inputs[-1].at[0, 5].set(0).at[1, 8].set(0)
    inputs = (*inputs[:-1], keep)
    initial = jnp.asarray(rng.normal(size=(batch, heads, depth, rank, width)), jnp.float32)
    return inputs, initial


@pytest.mark.parametrize("chunk", [1, 3, 4, 6, 12])
def test_chunked_outputs_and_all_gradients_match_branch_reference(chunk):
    inputs, initial = fixture()
    rng = np.random.default_rng(45)
    dy = jnp.asarray(rng.normal(size=(2, 12, 2, 3)), jnp.float32)
    df = jnp.asarray(rng.normal(size=initial.shape), jnp.float32)

    def objective(fn, xs, state):
        out, final = fn(xs, state)
        return jnp.sum(out * dy) + jnp.sum(final * df)

    fast = partial_chunk(chunk)
    want = reference(inputs, initial)
    got = jax.jit(fast)(inputs, initial)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5)
    reference_grad = jax.grad(lambda xs, state: objective(reference, xs, state), argnums=(0, 1))(inputs, initial)
    actual_grad = jax.jit(jax.grad(lambda xs, state: objective(fast, xs, state), argnums=(0, 1)))(inputs, initial)
    for a, b in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(reference_grad), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)


def partial_chunk(chunk):
    return lambda xs, state: chunked(xs, state, chunk)


def test_parent_survives_child_overwrite_and_pop_skips_decay():
    # Write A; push inherited child and overwrite with B; pop while requesting
    # strong decay. The pop must restore A, including under beta=1.
    actions = jnp.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], jnp.float32)[None, :, None]
    qk = jnp.ones((1, 4, 1, 1), jnp.float32)
    values = jnp.array([2, 7, 99, 0], jnp.float32).reshape(1, 4, 1, 1)
    decay = jnp.array([1, 1, 0.01, 1], jnp.float32).reshape(1, 4, 1, 1)
    beta = jnp.array([1, 1, 1, 0], jnp.float32).reshape(1, 4, 1)
    address = jnp.broadcast_to(jnp.array([1, 0, 0], jnp.float32), (1, 4, 1, 3))
    inputs = (actions, qk, qk, values, decay, beta, address, jnp.ones((1, 4, 1)))
    initial = jnp.zeros((1, 1, 3, 1, 1))
    output, _ = chunked(inputs, initial, 2)
    np.testing.assert_array_equal(np.asarray(output).reshape(-1), [2, 7, 2, 2])


def test_chunked_causality_document_reset_and_streaming_continuation():
    inputs, initial = fixture()
    full, final = chunked(inputs, initial, 3)
    first, mid = chunked(tuple(x[:, :6] for x in inputs), initial, 3)
    second, final2 = chunked(tuple(x[:, 6:] for x in inputs), mid, 3)
    np.testing.assert_allclose(jnp.concatenate((first, second), 1), full, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(final2, final, atol=3e-6, rtol=3e-5)
    changed = list(inputs)
    changed[3] = changed[3].at[:, 9:].add(100)
    changed_out, _ = chunked(tuple(changed), initial, 3)
    np.testing.assert_array_equal(changed_out[:, :9], full[:, :9])
    changed[3] = inputs[3].at[0, :5].add(100)
    reset_out, _ = chunked(tuple(changed), initial, 3)
    np.testing.assert_allclose(reset_out[0, 5:], full[0, 5:], atol=3e-6, rtol=3e-5)


def test_residual_initialization_and_learned_control_gradients():
    rng = np.random.default_rng(73)
    hidden = jnp.asarray(rng.normal(size=(1, 12, 8)), jnp.float32)
    segments = jnp.array([[0] * 6 + [1] * 6])
    shapes = [(8, 2 * size) for size in (2, 2, 3, 2, 1, 4, 3)] + [(6, 8)]
    weights = tuple(jnp.asarray(rng.normal(size=shape) * 0.02, jnp.float32) for shape in shapes)
    bias = jnp.zeros((2, 4))

    def residual(x, w, gain, action_bias):
        return local_read(x, segments, w, gain, action_bias, heads=2, depth=3, rank=2, width=3, chunk=4)

    np.testing.assert_array_equal(residual(hidden, weights, jnp.zeros(8), bias), jnp.zeros_like(hidden))

    def objective(x, w, g, b):
        return jnp.sum(jnp.sin(residual(x, w, g, b)))

    initial_grad = jax.jit(jax.grad(objective, argnums=(0, 1, 2, 3)))(hidden, weights, jnp.zeros(8), bias)
    assert float(jnp.linalg.norm(initial_grad[2])) > 0
    for x in jax.tree.leaves((initial_grad[0], initial_grad[1], initial_grad[3])):
        np.testing.assert_array_equal(x, jnp.zeros_like(x))
    active_grad = jax.jit(jax.grad(objective, argnums=(0, 1, 2, 3)))(hidden, weights, jnp.ones(8), bias)
    for x in jax.tree.leaves(active_grad):
        assert np.isfinite(x).all() and float(jnp.linalg.norm(x)) > 0
    changed = hidden.at[:, :6].add(10)
    np.testing.assert_allclose(
        residual(changed, weights, jnp.ones(8), bias)[:, 6:],
        residual(hidden, weights, jnp.ones(8), bias)[:, 6:],
        atol=3e-6,
        rtol=3e-5,
    )
