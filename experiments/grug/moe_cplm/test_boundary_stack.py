# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training numerical checks for fused boundary head mixing."""

from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track import model
from experiments.grug.fast_track.test_kda_local import _config, _mesh
from experiments.grug.moe import boundary_stack_kda as stack
from experiments.grug.moe_cplm.boundary_stack_kda_model import BoundaryStackKimiDeltaAttention, NativeKimiDeltaAttention
from experiments.grug.moe_cplm.boundary_stack_validation import candidate, compare, evaluate, fixture, literal
from experiments.grug.moe_cplm.test_stack_kda_model import serial_native


def test_native_layer_limit_and_boundary_action_gradients():
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
        baseline = NativeKimiDeltaAttention.init(cfg, key=jax.random.PRNGKey(8))
        layer = BoundaryStackKimiDeltaAttention.init(cfg, key=jax.random.PRNGKey(8), precision="fp32")
        x = jax.random.normal(jax.random.PRNGKey(9), (1, 32, 32))
        segments = (jnp.arange(32) >= 15)[None].astype(jnp.int32)
        stay = eqx.tree_at(lambda m: m.stack_action_bias, layer, jnp.array([-100.0, -100.0, 100.0]))
        with patch.object(model, "_kda_kernel", serial_native):
            expected = eqx.filter_jit(lambda m: m(x, segment_ids=segments)[0])(baseline)
        actual = eqx.filter_jit(lambda m: m(x, segment_ids=segments)[0])(stay)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: jnp.sum(jnp.sin(m(x, segment_ids=segments)[0])))
        )(layer)
        assert np.isfinite(loss)
        for value in (grads.w_stack_actions, grads.stack_action_bias):
            assert np.isfinite(value).all()
            assert np.max(np.abs(value)) > 0


def test_fp32_forward_and_all_adjoints_with_reset():
    xs, state = fixture(8, rank=8, chunk=4)
    compare(
        evaluate(lambda x, s: candidate(x, s, 4, interpret=True), xs, state),
        evaluate(lambda x, s: literal(x, s, 4), xs, state),
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_fused_reverse_matches_executed_native_scan(mixed):
    xs, state = fixture(8, rank=8, chunk=4)
    compare(
        evaluate(lambda x, s: candidate(x, s, 4, mixed, True), xs, state),
        evaluate(lambda x, s: candidate(x, s, 4, mixed, reference=True), xs, state),
    )


def test_push_pop_preserves_three_states_and_boundary_is_causal():
    xs, state = fixture(8, rank=8, chunk=4)
    actions = jnp.array([[[1, 0, 0], [0, 1, 0]]], jnp.float32)
    # Zero decay log and beta disable KDA writes; all slots retain their values.
    fixed = (actions, *xs[1:4], jnp.zeros_like(xs[4]), jnp.zeros_like(xs[5]), jnp.ones_like(xs[6]))
    out, final = candidate(fixed, state, 4, interpret=True)
    compare(final, jnp.concatenate((state[:, :3], jnp.zeros_like(state[:, :1])), axis=1))
    stay = (jnp.broadcast_to(jnp.array([0, 0, 1], jnp.float32), actions.shape), *fixed[1:])
    other, _ = candidate(stay, state, 4, interpret=True)
    compare(out[:, :4], other[:, :4])


def test_chunk_continuation():
    xs, state = fixture(8, rank=8, chunk=4)
    xs = (*xs[:-1], jnp.ones_like(xs[-1]))
    first = (xs[0][:, :1], *(x[:, :4] for x in xs[1:]))
    second = (xs[0][:, 1:], *(x[:, 4:] for x in xs[1:]))
    left, middle = candidate(first, state, 4, interpret=True)
    right, final = candidate(second, middle, 4, interpret=True)
    compare((jnp.concatenate((left, right), axis=1), final), candidate(xs, state, 4, interpret=True))
