# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import haliax as hax
import jax
import numpy as np
from levanter.layers.attention import AttentionMask

from experiments.hybrid_transfer.model import Residual, ResidualAttention, TransferConfig


def test_zero_query_attends_to_all_prior_values():
    config = TransferConfig(hidden_dim=16)
    reader = ResidualAttention.init(config)
    values = [
        hax.named(np.random.default_rng(i).normal(size=(3, 16)), (hax.Axis("position", 3), config.Embed))
        for i in range(5)
    ]
    expected = np.mean(np.stack([v.array for v in values]), axis=0)
    np.testing.assert_allclose(reader(values).array, expected, rtol=1e-6, atol=1e-6)


def test_readout_scaling_matches_fused_loss_path_and_base_depth_equivalence():
    config = TransferConfig(
        hidden_dim=64,
        intermediate_dim=128,
        num_heads=1,
        num_kv_heads=1,
        num_layers=2,
        base_depth=2,
        base_width=32,
        max_seq_len=8,
        attn_backend="vanilla",
    )
    tokens = hax.named(np.arange(8, dtype=np.int32), hax.Axis("position", 8))
    outputs = []
    for residual in (Residual.STANDARD, Residual.COMPLETEP, Residual.DEPTH_MUP):
        model = dataclasses.replace(config, residual=residual).build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(0))
        activations = model.activations(tokens, AttentionMask.causal())
        fused_logits = hax.dot(activations, model.get_lm_head(), axis=config.Embed)
        direct_logits = model(tokens, AttentionMask.causal())
        np.testing.assert_allclose(fused_logits.array, direct_logits.array, rtol=1e-6, atol=1e-6)
        outputs.append(direct_logits.array)
    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])


def test_hidden_initialization_scales_with_width_while_embedding_and_readout_do_not():
    measurements = []
    for width in (128, 512):
        config = TransferConfig(
            hidden_dim=width,
            intermediate_dim=width * 7 // 2,
            num_heads=width // 64,
            num_kv_heads=width // 64,
            num_layers=2,
            attn_backend="vanilla",
        )
        model = config.build(hax.Axis("vocab", 256), key=jax.random.PRNGKey(11))
        measurements.append(
            np.array(
                [
                    np.std(model.transformer.layers.stacked.mlp.gate_proj.weight.array),
                    np.std(model.embeddings.token_embeddings.weight.array),
                    np.std(model.lm_head.weight.array),
                ]
            )
        )
    # Four times the fan-in halves hidden std, while the outer matrices keep
    # their scale. The 3% tolerance covers sampling error in these finite draws.
    np.testing.assert_allclose(measurements[1] / measurements[0], [0.5, 1.0, 1.0], rtol=0.03)
