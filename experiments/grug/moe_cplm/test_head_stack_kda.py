# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe import head_stack_kda as stack
from experiments.grug.fast_track import model
from experiments.grug.fast_track.test_kda_local import _config, _mesh
from experiments.grug.moe_cplm.benchmark_head_stack_kda import fixture, native, objective
from experiments.grug.moe_cplm.head_stack_kda_model import HeadStackKimiDeltaAttention, NativeKimiDeltaAttention
from experiments.grug.moe_cplm.test_stack_kda_model import serial_native


@pytest.mark.parametrize("chunk", [2, 3, 6])
def test_chunked_values_and_all_adjoints_match_literal_stack(chunk):
    inputs, initial = fixture(12, rank=5, width=3, mixed=True)
    fast = lambda x, s: stack.chunked(x, s, chunk)  # noqa: E731
    for fn in (lambda f: f, lambda f: jax.value_and_grad(lambda x, s: objective(f, x, s), argnums=(0, 1))):
        got = jax.jit(fn(fast))(inputs, initial)
        want = jax.jit(fn(stack.reference))(inputs, initial)
        for actual, expected in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_hard_stay_matches_native_outputs_and_state_without_expansion():
    inputs, initial = fixture(12, rank=5, width=3)
    inputs = (jnp.broadcast_to(jnp.array([0, 0, 1, 0], jnp.float32), inputs[0].shape), *inputs[1:])
    got, final = stack.chunked(inputs, initial, 3)
    want, native_final = native(inputs, initial, serial=True)
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(final, native_final, atol=1e-5, rtol=1e-4)
    assert final.shape == native_final.shape == (1, 4, 5, 3)
    assert final.nbytes == native_final.nbytes


def test_hard_push_pop_restores_parent_slots_and_discards_bottom():
    inputs, initial = fixture(2, rank=5, width=3)
    actions = jnp.array([[[1, 0, 0, 0], [0, 1, 0, 0]]], jnp.float32)
    inputs = (actions, *inputs[1:-1], jnp.ones_like(inputs[-1]))
    _, final = stack.chunked(inputs, initial, 2)
    np.testing.assert_allclose(final[:, :3], initial[:, :3], atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(final[:, 3], 0)


def test_streaming_and_document_reset_match_full_sequence():
    inputs, initial = fixture(12, rank=5, width=3, mixed=True)
    full, final = stack.chunked(inputs, initial, 3)
    first, boundary = stack.chunked(tuple(x[:, :6] for x in inputs), initial, 3)
    second, streamed = stack.chunked(tuple(x[:, 6:] for x in inputs), boundary, 3)
    np.testing.assert_allclose(jnp.concatenate((first, second), 1), full, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(streamed, final, atol=1e-5, rtol=1e-4)
    reset, reset_final = stack.chunked(tuple(x[:, 6:] for x in inputs), jnp.zeros_like(initial), 3)
    np.testing.assert_allclose(reset, second, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(reset_final, final, atol=1e-5, rtol=1e-4)


@pytest.mark.timeout(180)
def test_native_layer_limit_and_shared_action_gradients():
    cfg = _config(max_seq_len=32, num_heads=4, head_dim=8, kda_onchip_decay_gate=True)
    with jax.set_mesh(_mesh()):
        native_layer = NativeKimiDeltaAttention.init(cfg, key=jax.random.PRNGKey(8))
        stacked = HeadStackKimiDeltaAttention.init(cfg, key=jax.random.PRNGKey(8))
        x = jax.random.normal(jax.random.PRNGKey(9), (1, 32, 32))
        segments = (jnp.arange(32) >= 15)[None].astype(jnp.int32)
        stay = eqx.tree_at(lambda m: m.stack_action_bias, stacked, jnp.array([-100.0, -100.0, 100.0, -100.0]))
        with patch.object(model, "_kda_kernel", serial_native):
            expected = eqx.filter_jit(lambda m: m(x, segment_ids=segments)[0])(native_layer)
        actual = eqx.filter_jit(lambda m: m(x, segment_ids=segments)[0])(stay)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: jnp.sum(jnp.sin(m(x, segment_ids=segments)[0])))
        )(stacked)
        assert np.isfinite(loss)
        for value in (grads.w_stack_actions, grads.stack_action_bias):
            assert np.isfinite(value).all()
            assert np.max(np.abs(value)) > 0
