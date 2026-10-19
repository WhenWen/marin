# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Value-embedding storage and gather VJPs; no optimizer updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.fast_track import model
from experiments.grug.fast_track.column_sharded_embedding import column_sharded_embedding


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_value_embedding_gather_matches_indexing_and_fp32_scatter(dtype):
    mesh = jax.make_mesh(
        (1, len(jax.devices()), 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    rng = np.random.default_rng(16)
    values = rng.normal(size=(64, 64)).astype(np.float32)
    # Repeated IDs span every batch shard, exercising the complete gradient sum.
    ids = np.tile(np.array([0, 1, 1, 63, 0, 17, 1, 63], np.int32), (len(jax.devices()) * 2, 1))
    cotangent = rng.normal(size=(*ids.shape, 64)).astype(np.float32)
    with jax.set_mesh(mesh):
        table = jax.device_put(jnp.asarray(values, dtype), NamedSharding(mesh, P(model._FSDP_AXES, None)))
        tokens = jax.device_put(ids, NamedSharding(mesh, P(model._BATCH_AXES, None)))
        cot = jax.device_put(jnp.asarray(cotangent, dtype), NamedSharding(mesh, P(model._BATCH_AXES, None, None)))

        @jax.jit
        def evaluate(t, i, g):
            output, pullback = jax.vjp(lambda x: column_sharded_embedding(x, i), t)
            return output, pullback(g)[0]

        output, gradient = evaluate(table, tokens, cot)
    expected = np.zeros(values.shape, np.float32)
    np.add.at(expected, ids, np.asarray(cot, np.float32))
    expected = np.asarray(jnp.asarray(expected, dtype), np.float32)
    np.testing.assert_array_equal(output, np.asarray(table)[ids])
    np.testing.assert_allclose(np.asarray(gradient, np.float32), expected, rtol=2e-6, atol=2e-6)
    assert gradient.sharding.spec == P(model._FSDP_AXES, None)


def test_mla_value_embedding_is_partitioned_without_changing_initial_values():
    mesh = jax.make_mesh(
        (1, len(jax.devices()), 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = model.GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        num_heads=2,
        head_dim=16,
        mla=True,
        mla_kv_latent_dim=16,
        value_embeds=model.ValueEmbeds.GATED,
    )
    key = jax.random.key(19)
    with jax.set_mesh(mesh):
        attention = jax.jit(lambda k: model.CausalSelfAttention.init(cfg, key=k, layer_index=jnp.array(0)))(key)
        value_key = jax.random.split(key, 7)[6]
        expected = model._init_weight(value_key, (64, 32), cfg.initializer_std)
    np.testing.assert_array_equal(attention.value_embed, expected)
    assert attention.value_embed.sharding.spec == P(model._FSDP_AXES, None)
    assert attention.value_embed.addressable_shards[0].data.size * len(jax.devices()) == attention.value_embed.size


def test_stacked_expert_initialization_keeps_seeded_values_in_sharded_storage():
    mesh = jax.make_mesh(
        (1, len(jax.devices()), 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    keys = jax.random.split(jax.random.key(31), 3)
    with jax.set_mesh(mesh):
        actual = jax.jit(
            jax.vmap(
                lambda key: MoEExpertMlp.init(
                    num_experts=8,
                    hidden_dim=32,
                    output_dim=32,
                    intermediate_dim=16,
                    initializer_std=0.02,
                    key=key,
                    implementation="ragged_all_to_all",
                )
            )
        )(keys)
        expected = jax.jit(
            jax.vmap(
                lambda key: tuple(
                    0.02 * jax.random.truncated_normal(k, -3, 3, shape)
                    for k, shape in zip(jax.random.split(key, 3), ((8, 32, 16), (8, 32, 16), (8, 16, 32)), strict=True)
                )
            )
        )(keys)
    for value, reference in zip((actual.w_gate, actual.w_up, actual.w_down), expected, strict=True):
        np.testing.assert_array_equal(value, reference)
        assert value.addressable_shards[0].data.size * len(jax.devices()) == value.size
