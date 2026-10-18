# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training VJP parity for saved AttnRes scheduling against ordinary AD.

This exercises the Belay backbone boundary; copy heads are omitted here and
remain covered separately by the full-model TPU launch preflight.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track import model
from experiments.grug.fast_track.test_kda_local import _SEQ, _VOCAB, _model


@pytest.mark.parametrize("index", [0, 3])
def test_saved_layer_matches_ordinary_ad_for_history_and_z_loss(index):
    mesh, net = _model(attn_res_layer_backward=model.AttnResLayerBackward.SAVE)
    rng = jax.random.split(jax.random.key(83), 5)
    tokens = jax.random.randint(rng[0], (2, _SEQ), 0, _VOCAB)
    segments = jnp.broadcast_to(jnp.arange(_SEQ) // 12, tokens.shape)
    mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
    with jax.set_mesh(mesh):
        queries = jax.random.normal(rng[1], (2 * net.config.num_layers + 1, 32)) * .1
        blocks = tuple(jax.random.normal(k, (2, _SEQ, 32)) * .1 for k in rng[2:4])
        logits = tuple(model._logit_rows(model._attn_res_source_logits(x, queries[2 * index:], 1e-6, False))
                       for x in blocks)
        partial = None if index == 0 else jax.random.normal(rng[4], (2, _SEQ, 32)) * .1
        args = (net.layers()[index], blocks, logits, partial, queries, None)

        def loss(values, implementation):
            output, history, scores, stats = implementation(values, mask, tokens, index == 3, index, 1e-6, None)
            # Direct history cotangents exercise their addition to the layer VJP;
            # the z-loss checks the differentiable auxiliary-statistic path.
            return (jnp.mean(output ** 2) + .03 * sum(jnp.mean(x ** 2) for x in history)
                    + .02 * sum(jnp.mean(x ** 2) for x in jax.tree.leaves(scores))
                    + .1 * stats[model._ATTN_RES_Z])

        reference = eqx.filter_jit(eqx.filter_value_and_grad(lambda x: loss(x, model._attn_res_layer_passthrough)))
        scheduled = eqx.filter_jit(eqx.filter_value_and_grad(lambda x: loss(x, model._attn_res_layer_saved)))
        want, want_grad = reference(args)
        got, got_grad = scheduled(args)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
        for actual, expected in zip(jax.tree.leaves(got_grad), jax.tree.leaves(want_grad), strict=True):
            actual, expected = np.asarray(actual), np.asarray(expected)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
