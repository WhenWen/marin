# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from levanter.optim import OptimizerConfig

from experiments.grug.moe_identity_hyperconnection.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig


def test_grug_moe_adamh_mask_routes_expert_mlp_weights_to_expert_group():
    params = {
        "blocks": {
            "0": {
                "mlp": {
                    "router": jnp.ones((8, 4), dtype=jnp.float32),
                    "expert_mlp": {
                        "w_gate": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_up": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
                    },
                },
                "shared": {
                    "w_gate": jnp.ones((8, 16), dtype=jnp.float32),
                },
            },
        },
        "token_embed": jnp.ones((128, 8), dtype=jnp.float32),
    }

    mask = GrugMoeAdamHConfig().create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["expert_mlp"]["w_gate"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_up"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_gate"] == "adamh_expert"
    assert mask["token_embed"] == "adam"


def test_identity_hyperconnection_parameters_are_adam_routed():
    params = {
        "blocks": {
            "0": {
                "attn_hc": {
                    "mapping_projection": jnp.ones((32, 8), dtype=jnp.float32),
                    "mapping_bias": jnp.ones((8,), dtype=jnp.float32),
                    "mapping_scale": jnp.ones((2,), dtype=jnp.float32),
                },
                "mlp_hc": {
                    "mapping_projection": jnp.ones((32, 8), dtype=jnp.float32),
                    "mapping_bias": jnp.ones((8,), dtype=jnp.float32),
                    "mapping_scale": jnp.ones((2,), dtype=jnp.float32),
                },
            }
        }
    }

    for optimizer_config in (GrugMoeAdamHConfig(), GrugMoeMuonHConfig()):
        mask = optimizer_config.create_mask(params)
        assert set(mask["blocks"]["0"]["attn_hc"].values()) == {"adam"}
        assert set(mask["blocks"]["0"]["mlp_hc"].values()) == {"adam"}


def test_identity_hyperconnection_optimizer_registry_names_are_variant_specific():
    assert OptimizerConfig.get_choice_class("grug_moe_identity_hc_adamh_v1") is GrugMoeAdamHConfig
    assert OptimizerConfig.get_choice_class("grug_moe_identity_hc_muonh_v1") is GrugMoeMuonHConfig
