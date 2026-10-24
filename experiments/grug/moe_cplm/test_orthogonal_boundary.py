# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training tests for input-conditioned orthogonal head-state mixing."""

from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track import model
from experiments.grug.fast_track.test_kda_local import _config, _mesh
from experiments.grug.moe import boundary_stack_kda as stack
from experiments.grug.moe_cplm.boundary_stack_kda_model import NativeKimiDeltaAttention
from experiments.grug.moe_cplm.boundary_stack_validation import compare, evaluate
from experiments.grug.moe_cplm.orthogonal_boundary_kda_model import OrthogonalBoundaryKimiDeltaAttention, cayley_matrix
from experiments.grug.moe_cplm.orthogonal_boundary_validation import candidate, fixture, literal
from experiments.grug.moe_cplm.test_stack_kda_model import serial_native


def test_cayley_preserves_state_singular_values_and_input_sensitivity():
    coordinates = jax.random.normal(jax.random.PRNGKey(91), (8, 6)) * 0.2
    matrix = np.asarray(cayley_matrix(coordinates))
    np.testing.assert_allclose(
        np.swapaxes(matrix, -1, -2) @ matrix, np.broadcast_to(np.eye(4), matrix.shape), atol=1e-5, rtol=1e-4
    )
    states = np.random.default_rng(91).normal(size=(8, 4, 32)).astype(np.float32)
    np.testing.assert_allclose(
        np.linalg.svd(matrix @ states, compute_uv=False), np.linalg.svd(states, compute_uv=False), atol=1e-5, rtol=1e-4
    )
    jacobian = jax.jacfwd(cayley_matrix)(jnp.zeros(6)).reshape(16, 6)
    assert np.linalg.matrix_rank(np.asarray(jacobian)) == 6


@pytest.mark.parametrize("mixed", [False, True])
def test_matrix_kernel_adjoints_include_cayley_coordinates(mixed):
    xs, state = fixture(8, rank=8, chunk=4)
    actual = evaluate(lambda x, s: candidate(x, s, 4, mixed, True), xs, state)
    reference = evaluate(lambda x, s: candidate(x, s, 4, mixed, reference=True), xs, state)
    compare(actual, reference)
    if not mixed:
        compare(actual, evaluate(lambda x, s: literal(x, s, 4), xs, state))


def test_matrix_boundary_is_causal_and_chunk_continuation_matches():
    xs, state = fixture(8, rank=8, chunk=4)
    xs = (*xs[:-1], jnp.ones_like(xs[-1]))
    out, final = candidate(xs, state, 4, interpret=True)
    changed = (xs[0].at[:, 0].add(0.3), *xs[1:])
    other, _ = candidate(changed, state, 4, interpret=True)
    compare(out[:, :4], other[:, :4])
    assert np.max(np.abs(np.asarray(out[:, 4:] - other[:, 4:]))) > 1e-4
    left, middle = candidate((xs[0][:, :1], *(x[:, :4] for x in xs[1:])), state, 4, interpret=True)
    right, end = candidate((xs[0][:, 1:], *(x[:, 4:] for x in xs[1:])), middle, 4, interpret=True)
    compare((jnp.concatenate((left, right), axis=1), end), (out, final))


def test_identity_layer_matches_native_and_controller_receives_gradients():
    cfg = _config(max_seq_len=32, num_heads=4, head_dim=8, kda_onchip_decay_gate=True)

    def interpreted(q, k, v, g, beta, actions, segment_ids=None, **kwargs):
        out, _ = stack.chunk_kda(
            *(jnp.swapaxes(x, 1, 2) for x in (q, k, v, g, beta)),
            actions=actions,
            segment_ids=segment_ids,
            matmul_dtype=jnp.float32,
            interpret=True,
        )
        return jnp.swapaxes(out, 1, 2)

    with jax.set_mesh(_mesh()), patch.object(stack, "native_kernel", interpreted):
        key = jax.random.PRNGKey(92)
        baseline = NativeKimiDeltaAttention.init(cfg, key=key)
        layer = OrthogonalBoundaryKimiDeltaAttention.init(cfg, key=key, precision="fp32")
        x = jax.random.normal(jax.random.PRNGKey(93), (1, 32, 32))
        segments = (jnp.arange(32) >= 15)[None].astype(jnp.int32)
        with patch.object(model, "_kda_kernel", serial_native):
            expected = eqx.filter_jit(lambda m: m(x, segment_ids=segments)[0])(baseline)
        actual = eqx.filter_jit(lambda m: m(x, segment_ids=segments)[0])(layer)
        compare(actual, expected)
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: jnp.sum(jnp.sin(m(x, segment_ids=segments)[0])))
        )(layer)
        assert np.isfinite(loss)
        for value in (grads.w_boundary_matrix, grads.boundary_matrix_bias):
            assert np.isfinite(value).all() and np.max(np.abs(value)) > 0
        # A nonzero controller must produce different matrices for different inputs.
        weight = jax.random.normal(jax.random.PRNGKey(94), (32, 6)) * 0.02
        conditioned = eqx.tree_at(lambda m: m.w_boundary_matrix, layer, weight)
        matrices, _ = eqx.filter_jit(lambda m: m.boundary_actions(x))(conditioned)
        assert np.max(np.abs(np.asarray(matrices[:, 0] - matrices[:, 1]))) > 1e-4
