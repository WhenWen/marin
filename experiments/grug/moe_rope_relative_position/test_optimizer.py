# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from experiments.grug.moe_rope_relative_position.optimizer import (
    GrugMoeMuonHConfig,
    RelativeQueryProjectionOptimizer,
)


def test_inkling_parameters_route_to_adam_without_changing_other_matrix_groups():
    params = {
        "blocks": {
            "0": {
                "attn": {
                    "w_q": jnp.ones((8, 8), dtype=jnp.float32),
                    "w_r": jnp.ones((8, 16), dtype=jnp.float32),
                    "relative_position_embeddings": jnp.ones((16, 32), dtype=jnp.float32),
                    "q_norm_weight": jnp.ones((8,), dtype=jnp.float32),
                    "k_norm_weight": jnp.ones((8,), dtype=jnp.float32),
                },
                "mlp": {"expert_mlp": {"w_gate": jnp.ones((4, 8, 16), dtype=jnp.float32)}},
            }
        },
        "output_proj": jnp.ones((8, 128), dtype=jnp.float32),
    }
    optimizer = GrugMoeMuonHConfig(
        relative_query_projection_optimizer=RelativeQueryProjectionOptimizer.ADAM,
    )

    mask = optimizer.create_mask(params)

    attention_mask = mask["blocks"]["0"]["attn"]
    assert attention_mask["w_q"] == "muonh"
    assert attention_mask["w_r"] == "adam"
    assert attention_mask["relative_position_embeddings"] == "adam"
    assert attention_mask["q_norm_weight"] == "adam"
    assert attention_mask["k_norm_weight"] == "adam"
    assert mask["blocks"]["0"]["mlp"]["expert_mlp"]["w_gate"] == "muonh"
    assert mask["output_proj"] == "adamh"
