# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training checks of native projections and layer selection."""

from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track import model
from experiments.grug.fast_track.test_kda_local import _config, _mesh
from experiments.grug.moe import kda
from experiments.grug.moe_cplm.stack_kda_model import NativeKimiDeltaAttention, StackKimiDeltaAttention


def serial_native(q, k, v, g, beta, extras, *, options):
    if "gate_p" in extras:
        g = model.gate_log_decay(
            model.GateFn(model.GateKind.SIGMOID, model.KDA_MIN_LOG_DECAY),
            g.astype(jnp.float32),
            extras["gate_p"],
            extras["gate_bias"],
        )
    args = tuple(jnp.swapaxes(x, 1, 2) for x in (q, k, v, g, beta))
    segments = extras["segment_ids"][:, None]
    return jnp.swapaxes(kda.recurrent_kda(*args, segment_ids=segments)[0], 1, 2)


@pytest.mark.timeout(180)
def test_native_layer_limit_and_action_gradients():
    cfg = _config(max_seq_len=32, kda_onchip_decay_gate=True)
    with jax.set_mesh(_mesh()):
        key = jax.random.PRNGKey(8)
        native = NativeKimiDeltaAttention.init(cfg, key=key)
        stacked = StackKimiDeltaAttention.init(cfg, key=key)
        for name in ("w_q", "w_k", "w_v", "w_g", "w_o", "w_beta", "w_a_down", "w_a_up"):
            np.testing.assert_array_equal(getattr(native, name), getattr(stacked, name))
        x = jax.random.normal(jax.random.PRNGKey(9), (1, 32, 32))
        segments = (jnp.arange(32) >= 15)[None].astype(jnp.int32)
        stay = eqx.tree_at(
            lambda m: m.stack_action_bias, stacked, jnp.broadcast_to(jnp.array([-100.0, -100.0, 100.0, -100.0]), (2, 4))
        )
        with patch.object(model, "_kda_kernel", serial_native):
            expected = eqx.filter_jit(lambda m: m(x, segment_ids=segments)[0])(native)
        actual = eqx.filter_jit(lambda m: m(x, segment_ids=segments)[0])(stay)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        loss, grads = eqx.filter_jit(
            eqx.filter_value_and_grad(lambda m: jnp.sum(jnp.sin(m(x, segment_ids=segments)[0])))
        )(stacked)
        assert np.isfinite(loss)
        for value in (grads.w_stack_actions, grads.stack_action_bias):
            assert np.isfinite(value).all()
            assert np.max(np.abs(value)) > 0


def test_only_four_native_kda_layers_change():
    with jax.set_mesh(_mesh()), patch.object(model, "KimiDeltaAttention", StackKimiDeltaAttention):
        network = model.Transformer.init(_config(max_seq_len=32), key=jax.random.PRNGKey(0))
        assert [type(layer.attn) for layer in network.layers()] == [
            StackKimiDeltaAttention,
            StackKimiDeltaAttention,
            StackKimiDeltaAttention,
            model.CausalSelfAttention,
            StackKimiDeltaAttention,
            model.CausalSelfAttention,
        ]
