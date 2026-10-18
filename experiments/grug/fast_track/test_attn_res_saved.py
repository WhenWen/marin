# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training VJP parity for saved AttnRes scheduling against ordinary AD.

This exercises the Belay backbone boundary; copy heads are omitted here and
remain covered separately by the full-model TPU launch preflight.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track import model
from experiments.grug.fast_track.attn_res_saved import saved_layer
from experiments.grug.fast_track.test_kda_local import _SEQ, _VOCAB, _model


@pytest.mark.parametrize("index", [0, 3])
@pytest.mark.parametrize("z_weight", [0.0, 0.1])
def test_saved_layer_matches_ordinary_ad_for_history_and_z_loss(index, z_weight):
    mesh, net = _model(attn_res_layer_backward=model.AttnResLayerBackward.SAVE)
    rng = jax.random.split(jax.random.key(83), 5)
    tokens = jax.random.randint(rng[0], (2, _SEQ), 0, _VOCAB)
    segments = jnp.broadcast_to(jnp.arange(_SEQ) // 12, tokens.shape)
    mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
    with jax.set_mesh(mesh):
        queries = jax.random.normal(rng[1], (2 * net.config.num_layers + 1, 32)) * 0.1
        blocks = tuple(
            jax.sharding.reshard(jax.random.normal(k, (2, _SEQ, 32)) * 0.1, model._batch_spec()) for k in rng[2:4]
        )
        logits = tuple(
            model._logit_rows(model._attn_res_source_logits(x, queries[2 * index :], 1e-6, False)) for x in blocks
        )
        running = (
            None
            if index == 0
            else jax.sharding.reshard(jax.random.normal(rng[4], (2, _SEQ, 32)) * 0.1, model._batch_spec())
        )
        args = (net.layers()[index], blocks, logits, running, queries, None)

        def loss(values, implementation):
            output, history, scores, stats = implementation(values, mask, tokens, index == 3, index, 1e-6, None)
            # Direct history cotangents exercise their addition to the layer VJP;
            # the z-loss checks the differentiable auxiliary-statistic path.
            return (
                jnp.mean(output**2)
                + 0.03 * sum(jnp.mean(x**2) for x in history)
                + 0.02 * sum(jnp.mean(x**2) for x in jax.tree.leaves(scores))
                + (z_weight * stats[model._ATTN_RES_Z] if z_weight else 0)
            )

        reference = eqx.filter_jit(eqx.filter_value_and_grad(lambda x: loss(x, model._attn_res_layer_passthrough)))
        implementation = partial(
            saved_layer,
            layer_forward=model._attn_res_layer,
            statistic_keys=(*model._DIFFERENTIABLE_STATS, model._ATTN_RES_Z),
        )
        scheduled = eqx.filter_jit(eqx.filter_value_and_grad(lambda x: loss(x, implementation)))
        want, want_grad = reference(args)
        got, got_grad = scheduled(args)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
        for actual, expected in zip(jax.tree.leaves(got_grad), jax.tree.leaves(want_grad), strict=True):
            actual, expected = np.asarray(actual), np.asarray(expected)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
