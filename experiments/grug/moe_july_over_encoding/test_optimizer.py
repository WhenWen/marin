# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_july_over_encoding.optimizer import GrugMoeMuonHConfig


def _embedding_params():
    return {
        "token_embed": jnp.ones((8, 2), dtype=jnp.float32),
        "over_encoding": {
            "tables": jnp.ones((1, 8, 2), dtype=jnp.float32),
            "projections": (jnp.ones((2, 2), dtype=jnp.float32),),
        },
    }


def test_muonh_mask_routes_only_over_encoding_tables_to_separate_adam():
    mask = GrugMoeMuonHConfig().create_mask(_embedding_params())

    assert mask["token_embed"] == "adam"
    assert mask["over_encoding"]["tables"] == "over_encoding_adam"
    assert mask["over_encoding"]["projections"][0] == "muonh"


def test_over_encoding_lr_multiplier_changes_only_table_update():
    params = _embedding_params()
    grads = jax.tree.map(jnp.ones_like, params)

    def first_update(multiplier: float):
        config = GrugMoeMuonHConfig(
            learning_rate=0.03,
            adam_lr=0.002,
            over_encoding_lr_multiplier=multiplier,
            warmup=0,
            decay=0,
            lr_schedule="constant",
            max_grad_norm=None,
        )
        optimizer = config.build(num_train_steps=10)
        state = optimizer.init(params)
        updates, _ = optimizer.update(grads, state, params)
        return updates

    unit_updates = first_update(1.0)
    quarter_updates = first_update(0.25)
    quadruple_updates = first_update(4.0)

    np.testing.assert_allclose(quarter_updates["token_embed"], unit_updates["token_embed"])
    np.testing.assert_allclose(quadruple_updates["token_embed"], unit_updates["token_embed"])
    np.testing.assert_allclose(
        quarter_updates["over_encoding"]["tables"],
        unit_updates["over_encoding"]["tables"] * 0.25,
    )
    np.testing.assert_allclose(
        quadruple_updates["over_encoding"]["tables"],
        unit_updates["over_encoding"]["tables"] * 4.0,
    )
    np.testing.assert_allclose(
        quarter_updates["over_encoding"]["projections"][0],
        unit_updates["over_encoding"]["projections"][0],
    )
    np.testing.assert_allclose(
        quadruple_updates["over_encoding"]["projections"][0],
        unit_updates["over_encoding"]["projections"][0],
    )
