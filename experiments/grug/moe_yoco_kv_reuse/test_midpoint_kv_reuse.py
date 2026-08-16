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
from experiments.grug.moe_yoco_kv_reuse.experiment import build_step as build_scale_step
from experiments.grug.moe_yoco_kv_reuse.experiment_classical_yoco import build_step as build_classical_step
from experiments.grug.moe_yoco_kv_reuse.experiment_classical_yoco_july import (
    build_step as build_classical_july_step,
)
from experiments.grug.moe_yoco_kv_reuse.experiment_overtrain import build_step as build_overtrain_step
from experiments.grug.moe_yoco_kv_reuse.recipe import (
    OVERTRAIN_D512_750_TPP,
    OVERTRAIN_D512_ACTIVE_PARAMETERS,
    OVERTRAIN_TOKENS_PER_ACTIVE_PARAMETER,
    POINTS,
    ExperimentPoint,
    baseline_recipe,
    classical_yoco_recipe,
    variant_recipe,
)


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
    [(512, 6, 3, 2), (768, 8, 4, 3), (1024, 11, 6, 5), (1280, 13, 7, 6)],
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
    for field_name, baseline_value in dataclasses.asdict(baseline_config).items():
        assert variant_model_fields[field_name] == baseline_value
    assert dataclasses.asdict(variant_optimizer) == dataclasses.asdict(baseline_optimizer)


def test_overtrain_recipe_matches_750_tokens_per_active_parameter():
    point = OVERTRAIN_D512_750_TPP
    trained_tokens = point.num_steps * point.batch_size * 8192
    target_tokens = OVERTRAIN_D512_ACTIVE_PARAMETERS * OVERTRAIN_TOKENS_PER_ACTIVE_PARAMETER

    assert abs(trained_tokens - target_tokens) < point.batch_size * 8192


def test_new_runs_pin_resources_to_us_central1():
    d1280 = next(point for point in POINTS if point.hidden_dim == 1280)
    resources = [
        build_scale_step(d1280).config.resources.value,
        build_overtrain_step(fixed_yoco=False).config.resources.value,
        build_overtrain_step(fixed_yoco=True).config.resources.value,
    ]

    assert all(resource.regions == ("us-central1",) for resource in resources)


@pytest.mark.parametrize("num_layers", [6, 8, 11, 13])
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


@pytest.mark.parametrize("num_layers", [6, 8, 11, 13])
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


def _parameter_count(tree: object) -> int:
    return sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(tree) if hasattr(leaf, "shape"))


def test_classical_yoco_reuses_one_projected_kv_pair():
    cfg = model.GrugModelConfig(**_tiny_config_kwargs(num_layers=6), shared_projected_kv_start_layer=3)
    token_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    with jax.set_mesh(_single_device_mesh()):
        transformer = model.Transformer.init(cfg, key=jax.random.key(6))
        logits = transformer.logits(token_ids)

    assert transformer.blocks[3].attn.w_k is not None
    assert transformer.blocks[3].attn.w_v is not None
    assert transformer.blocks[4].attn.w_k is None
    assert transformer.blocks[4].attn.w_v is None
    assert transformer.blocks[5].attn.w_k is None
    assert transformer.blocks[5].attn.w_v is None
    assert np.all(np.isfinite(np.asarray(logits)))


@pytest.mark.parametrize(
    ("point", "bare_deficit", "expert_delta", "head_delta"),
    [
        (POINTS[0], 262_144, 512, 1_024),
        (POINTS[1], 589_824, 0, 2_304),
    ],
)
def test_classical_yoco_parameter_reinvestment_is_close_to_baseline(
    point: ExperimentPoint,
    bare_deficit: int,
    expert_delta: int,
    head_delta: int,
):
    classical_cfg, _ = classical_yoco_recipe(point)
    expert_cfg, _ = classical_yoco_recipe(point, parameter_match="expert")
    head_cfg, _ = classical_yoco_recipe(point, parameter_match="heads")
    baseline_cfg = dataclasses.replace(classical_cfg, shared_projected_kv_start_layer=None)

    with jax.set_mesh(_single_device_mesh()):
        baseline = jax.eval_shape(lambda: model.Transformer.init(baseline_cfg, key=jax.random.key(7)))
        classical = jax.eval_shape(lambda: model.Transformer.init(classical_cfg, key=jax.random.key(7)))
        expert = jax.eval_shape(lambda: model.Transformer.init(expert_cfg, key=jax.random.key(7)))
        heads = jax.eval_shape(lambda: model.Transformer.init(head_cfg, key=jax.random.key(7)))

    baseline_parameters = _parameter_count(baseline)
    assert baseline_parameters - _parameter_count(classical) == bare_deficit
    assert _parameter_count(expert) - baseline_parameters == expert_delta
    assert _parameter_count(heads) - baseline_parameters == head_delta


@pytest.mark.parametrize(
    ("variant_name", "parameter_match", "extra_expert_dim", "extra_head_layers"),
    [
        ("classical", None, 0, ()),
        ("classical-expert-match", "expert", 171, ()),
        ("classical-head-match", "heads", 0, (3, 4)),
    ],
)
def test_classical_yoco_launch_matrix(
    variant_name: str,
    parameter_match: str | None,
    extra_expert_dim: int,
    extra_head_layers: tuple[int, ...],
):
    step = build_classical_step(variant_name, parameter_match)
    launch = step.config
    model_config = launch.model.value

    assert launch.resources.value.regions == ("us-central1",)
    assert launch.steps.value == 118_620
    assert launch.batch_size.value == 16
    assert model_config.kv_reuse_start_layer is None
    assert model_config.shared_projected_kv_start_layer == 3
    assert model_config.additional_shared_expert_intermediate_dim == extra_expert_dim
    assert model_config.additional_query_head_layers == extra_head_layers
    assert launch.run_id.endswith(f"-{variant_name}-d512")


@pytest.mark.parametrize(
    ("point", "expected_steps", "expected_batch_size", "expected_start_layer", "expert_dim", "head_layers"),
    [
        (POINTS[0], 10_980, 16, 3, 171, (3, 4)),
        (POINTS[1], 16_875, 32, 4, 256, (4, 5, 6)),
    ],
)
@pytest.mark.parametrize(
    ("variant_name", "parameter_match"),
    [("classical", None), ("classical-expert-match", "expert"), ("classical-head-match", "heads")],
)
def test_classical_yoco_july_launch_matrix(
    point: ExperimentPoint,
    expected_steps: int,
    expected_batch_size: int,
    expected_start_layer: int,
    expert_dim: int,
    head_layers: tuple[int, ...],
    variant_name: str,
    parameter_match: str | None,
):
    step = build_classical_july_step(point, variant_name, parameter_match)
    launch = step.config
    model_config = launch.model.value

    assert launch.resources.value.regions == ("us-central1",)
    assert launch.steps.value == expected_steps
    assert launch.batch_size.value == expected_batch_size
    assert model_config.shared_projected_kv_start_layer == expected_start_layer
    assert model_config.additional_shared_expert_intermediate_dim == (expert_dim if parameter_match == "expert" else 0)
    assert model_config.additional_query_head_layers == (head_layers if parameter_match == "heads" else ())
    assert launch.run_id.endswith(f"-{variant_name}-d{point.hidden_dim}")


def test_classical_yoco_backpropagates_through_shared_projected_kv():
    cfg = model.GrugModelConfig(**_tiny_config_kwargs(num_layers=6), shared_projected_kv_start_layer=3)
    token_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    with jax.set_mesh(_single_device_mesh()):
        transformer = model.Transformer.init(cfg, key=jax.random.key(8))

        def squared_logits(candidate: model.Transformer) -> jax.Array:
            return jnp.mean(jnp.square(candidate.logits(token_ids)))

        loss, grads = eqx.filter_value_and_grad(squared_logits)(transformer)

    assert np.isfinite(np.asarray(loss))
    assert np.linalg.norm(np.asarray(grads.blocks[3].attn.w_k)) > 0
    assert grads.blocks[4].attn.w_k is None
    assert grads.blocks[5].attn.w_v is None
