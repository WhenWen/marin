# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from levanter.grug.attention import AttentionMask

from experiments.grug.moe import model as baseline_model
from experiments.grug.moe_yoco_kv_reuse import model
from experiments.grug.moe_yoco_kv_reuse.recipe import POINTS, baseline_recipe, variant_recipe


def _single_device_mesh() -> Mesh:
    return Mesh(
        np.array([jax.devices()[0]]).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _tiny_config_kwargs(num_layers: int) -> dict[str, int]:
    return {
        "vocab_size": 32,
        "hidden_dim": 8,
        "intermediate_dim": 4,
        "shared_expert_intermediate_dim": 4,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_layers": num_layers,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "max_seq_len": 4,
        "sliding_window": 4,
    }


@pytest.mark.parametrize(
    ("hidden_dim", "expected_layers", "expected_start", "expected_source"),
    [(512, 6, 3, 2), (768, 8, 4, 3), (1024, 11, 6, 5)],
)
def test_recipe_derives_midpoint_from_model_depth(
    hidden_dim: int,
    expected_layers: int,
    expected_start: int,
    expected_source: int,
):
    point = next(point for point in POINTS if point.hidden_dim == hidden_dim)
    baseline_config, baseline_optimizer = baseline_recipe(point)
    variant_config, variant_optimizer = variant_recipe(point)

    assert variant_config.num_layers == expected_layers
    assert variant_config.kv_reuse_start_layer == expected_start
    assert variant_config.kv_reuse_start_layer - 1 == expected_source

    variant_model_fields = dataclasses.asdict(variant_config)
    del variant_model_fields["kv_reuse_start_layer"]
    assert variant_model_fields == dataclasses.asdict(baseline_config)
    assert dataclasses.asdict(variant_optimizer) == dataclasses.asdict(baseline_optimizer)


@pytest.mark.parametrize("num_layers", [6, 8, 11])
def test_midpoint_variant_preserves_every_july_parameter_at_initialization(num_layers: int):
    config_kwargs = _tiny_config_kwargs(num_layers)
    key = jax.random.key(0)

    with jax.set_mesh(_single_device_mesh()):
        baseline = baseline_model.Transformer.init(baseline_model.GrugModelConfig(**config_kwargs), key=key)
        variant = model.Transformer.init(
            model.GrugModelConfig(**config_kwargs, kv_reuse_start_layer=(num_layers + 1) // 2),
            key=key,
        )

    baseline_arrays = [leaf for leaf in jax.tree.leaves(baseline) if eqx.is_array(leaf)]
    variant_arrays = [leaf for leaf in jax.tree.leaves(variant) if eqx.is_array(leaf)]
    assert len(variant_arrays) == len(baseline_arrays)
    for variant_array, baseline_array in zip(variant_arrays, baseline_arrays, strict=True):
        np.testing.assert_array_equal(np.asarray(variant_array), np.asarray(baseline_array))


def test_attention_can_take_kv_from_a_different_activation():
    cfg = model.GrugModelConfig(**_tiny_config_kwargs(num_layers=6), kv_reuse_start_layer=3)
    query_input = jax.random.normal(jax.random.key(1), (1, 4, cfg.hidden_dim))
    other_kv_input = jax.random.normal(jax.random.key(2), query_input.shape)

    with jax.set_mesh(_single_device_mesh()):
        attention = model.CausalSelfAttention.init(cfg, key=jax.random.key(3))
        implicit_self_attention = attention(query_input, AttentionMask.causal(), disable_rope=True)
        explicit_self_attention = attention(
            query_input,
            AttentionMask.causal(),
            disable_rope=True,
            kv_input=query_input,
        )
        reused_kv_attention = attention(
            query_input,
            AttentionMask.causal(),
            disable_rope=True,
            kv_input=other_kv_input,
        )

    np.testing.assert_array_equal(np.asarray(implicit_self_attention), np.asarray(explicit_self_attention))
    assert not np.allclose(np.asarray(reused_kv_attention), np.asarray(explicit_self_attention))


@pytest.mark.parametrize("num_layers", [6, 8, 11])
def test_midpoint_reuse_changes_computation_without_changing_parameters(num_layers: int):
    config_kwargs = _tiny_config_kwargs(num_layers)
    token_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    key = jax.random.key(4)

    with jax.set_mesh(_single_device_mesh()):
        baseline = baseline_model.Transformer.init(baseline_model.GrugModelConfig(**config_kwargs), key=key)
        variant = model.Transformer.init(
            model.GrugModelConfig(**config_kwargs, kv_reuse_start_layer=(num_layers + 1) // 2),
            key=key,
        )
        baseline_logits = baseline.logits(token_ids)
        variant_logits = variant.logits(token_ids)

    assert baseline_logits.shape == variant_logits.shape
    assert np.all(np.isfinite(np.asarray(variant_logits)))
    assert not np.allclose(np.asarray(variant_logits), np.asarray(baseline_logits))


def test_midpoint_reuse_backpropagates_through_cached_source():
    config_kwargs = _tiny_config_kwargs(num_layers=6)
    token_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    with jax.set_mesh(_single_device_mesh()):
        variant = model.Transformer.init(
            model.GrugModelConfig(**config_kwargs, kv_reuse_start_layer=3),
            key=jax.random.key(5),
        )

        def squared_logits(candidate: model.Transformer) -> jax.Array:
            return jnp.mean(jnp.square(candidate.logits(token_ids)))

        loss, grads = eqx.filter_value_and_grad(squared_logits)(variant)

    assert np.isfinite(np.asarray(loss))
    grad_arrays = [leaf for leaf in jax.tree.leaves(grads) if eqx.is_array(leaf)]
    assert grad_arrays
    assert all(np.all(np.isfinite(np.asarray(grad))) for grad in grad_arrays)
    assert np.linalg.norm(np.asarray(grads.blocks[2].attn.w_k)) > 0
    assert np.linalg.norm(np.asarray(grads.blocks[3].attn.w_k)) > 0
