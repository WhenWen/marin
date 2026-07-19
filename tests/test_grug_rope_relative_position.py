# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from marin.execution.types import VersionedValue
from levanter.grug.attention import AttentionMask

from experiments.grug.moe_rope_relative_position import model
from experiments.grug.moe_rope_relative_position.launch_gate1 import (
    GATE_1_POINTS,
    gate_1_recipe,
    gate_1_steps,
)
from experiments.grug.moe_rope_relative_position.optimizer import RelativeQueryProjectionOptimizer


def _single_device_mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1]).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )


def _dense_reference(q, k, v, relative_queries, relative_embeddings, mask):
    repeat = q.shape[2] // k.shape[2]
    k = jnp.repeat(k, repeat, axis=2)
    v = jnp.repeat(v, repeat, axis=2)
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk",
        q * (q.shape[-1] ** -0.5),
        k,
        preferred_element_type=jnp.float32,
    )
    positions = jnp.arange(q.shape[1], dtype=jnp.int32)
    distances = positions[:, None] - positions[None, :]
    relative_indices = jnp.clip(distances, 0, relative_embeddings.shape[1] - 1)
    relative_vectors = relative_embeddings[:, relative_indices]
    relative_bias = jnp.einsum(
        "bqhr,rqk->bhqk",
        relative_queries,
        relative_vectors,
        preferred_element_type=jnp.float32,
    )
    relative_bias = jnp.where(
        (distances >= 0) & (distances < relative_embeddings.shape[1]),
        relative_bias,
        0.0,
    )
    allowed = mask.materialize_mask(q.shape[1], k.shape[1])
    if allowed.ndim == 2:
        allowed = allowed[None, :, :]
    weights = jax.nn.softmax(jnp.where(allowed[:, None, :, :], scores + relative_bias, -1e9), axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v).astype(v.dtype)


def test_relative_attention_matches_dense_values_and_gradients():
    keys = jax.random.split(jax.random.key(0), 5)
    q = jax.random.normal(keys[0], (2, 5, 4, 3))
    k = jax.random.normal(keys[1], (2, 5, 2, 3))
    v = jax.random.normal(keys[2], (2, 5, 2, 3))
    relative_queries = jax.random.normal(keys[3], (2, 5, 4, 2))
    relative_embeddings = jax.random.normal(keys[4], (2, 3))
    segment_ids = jnp.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=jnp.int32)
    mask = AttentionMask.causal(sliding_window=3).with_segment_ids(segment_ids)

    def actual(*args):
        return model.relative_position_attention(*args, mask, block_size=4)

    def expected(*args):
        return _dense_reference(*args, mask)

    np.testing.assert_allclose(
        np.asarray(actual(q, k, v, relative_queries, relative_embeddings)),
        np.asarray(expected(q, k, v, relative_queries, relative_embeddings)),
        rtol=1e-5,
        atol=1e-5,
    )

    actual_grads = jax.grad(lambda *args: jnp.sum(actual(*args) ** 2), argnums=(0, 1, 2, 3, 4))(
        q, k, v, relative_queries, relative_embeddings
    )
    expected_grads = jax.grad(lambda *args: jnp.sum(expected(*args) ** 2), argnums=(0, 1, 2, 3, 4))(
        q, k, v, relative_queries, relative_embeddings
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual_grad), np.asarray(expected_grad), rtol=1e-5, atol=1e-5)


def test_inkling_parameters_use_independent_untruncated_normal_initialization():
    cfg = model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=8,
        initializer_std=0.03,
        relative_position_dim=4,
        relative_position_extent=8,
        relative_position_initializer_std=0.02,
        attention_qk_normalization=model.AttentionQKNormalization.LEARNED_RMS,
        relative_position_embedding_init=model.RelativePositionEmbeddingInit.NORMAL,
        relative_query_projection_init=model.RelativeQueryProjectionInit.NORMAL,
    )
    key = jax.random.key(0)
    split_keys = jax.random.split(key, 6)

    with jax.set_mesh(_single_device_mesh()):
        attention = model.CausalSelfAttention.init(cfg, key=key)

    expected_w_r = 0.02 * jax.random.normal(split_keys[3], (32, 8))
    expected_table = 0.02 * jax.random.normal(split_keys[4], (4, 8))
    np.testing.assert_array_equal(np.asarray(attention.w_r), np.asarray(expected_w_r))
    np.testing.assert_array_equal(np.asarray(attention.relative_position_embeddings), np.asarray(expected_table))
    np.testing.assert_array_equal(np.asarray(attention.q_norm_weight), np.ones((16,), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(attention.k_norm_weight), np.ones((16,), dtype=np.float32))


@pytest.mark.parametrize("disable_rope", [False, True])
def test_relative_attention_preserves_july_half_rope_policy(monkeypatch, disable_rope):
    cfg = model.GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=4,
        sliding_window=4,
        qk_mult=0.5,
        attention_qk_normalization=model.AttentionQKNormalization.LEARNED_RMS,
    )
    q_gain = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float32)
    k_gain = jnp.array([2.0, 1.5, 1.0, 0.5], dtype=jnp.float32)
    captured = {}

    with jax.set_mesh(_single_device_mesh()):
        attention = model.CausalSelfAttention.init(cfg, key=jax.random.key(0))
        attention = eqx.tree_at(lambda module: module.w_q, attention, attention.w_q.astype(jnp.bfloat16))
        attention = eqx.tree_at(lambda module: module.w_k, attention, attention.w_k.astype(jnp.bfloat16))
        attention = eqx.tree_at(lambda module: module.w_v, attention, attention.w_v.astype(jnp.bfloat16))
        attention = eqx.tree_at(lambda module: module.q_norm_weight, attention, q_gain)
        attention = eqx.tree_at(lambda module: module.k_norm_weight, attention, k_gain)

        def capture_attention(q, k, v, relative_queries, relative_embeddings, mask, *, block_size):
            del relative_queries, relative_embeddings, mask, block_size
            captured["q"] = q
            captured["k"] = k
            return jnp.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]), dtype=v.dtype)

        monkeypatch.setattr(model, "relative_position_attention", capture_attention)
        x = jnp.arange(24, dtype=jnp.bfloat16).reshape(1, 3, 8) / 8
        attention(x, AttentionMask.causal(sliding_window=4), disable_rope=disable_rope)

    projected_q = jnp.einsum("bsh,hd->bsd", x, attention.w_q).reshape(1, 3, 2, 4)
    projected_k = jnp.einsum("bsh,hd->bsd", x, attention.w_k).reshape(1, 3, 1, 4)
    expected_q = (model.rms_norm(projected_q) * q_gain).astype(projected_q.dtype)
    expected_k = (model.rms_norm(projected_k) * k_gain).astype(projected_k.dtype)
    if not disable_rope:
        q_rot, k_rot = model.apply_rotary_embedding(
            expected_q[..., :2],
            expected_k[..., :2],
            seq_len=3,
            head_dim=2,
            rope=cfg.rope,
        )
        expected_q = jnp.concatenate([q_rot, expected_q[..., 2:]], axis=-1)
        expected_k = jnp.concatenate([k_rot, expected_k[..., 2:]], axis=-1)
    expected_q = expected_q * cfg.qk_mult

    assert captured["q"].dtype == jnp.bfloat16
    assert captured["k"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(captured["q"]), np.asarray(expected_q))
    np.testing.assert_array_equal(np.asarray(captured["k"]), np.asarray(expected_k))


@pytest.mark.parametrize(
    ("index", "hidden_dim", "expected_run_id", "batch_size", "num_steps", "initializer_std"),
    (
        (0, 512, "MOE-JULY-ROPE-RPE-INKP2-001-d512", 16, 10_980, 0.022097086912079608),
        (1, 768, "MOE-JULY-ROPE-RPE-INKP2-002-d768", 32, 16_875, 0.018042195912175808),
    ),
)
def test_gate_1_cells_match_real_july_baseline_and_inkling_parameterization(
    index, hidden_dim, expected_run_id, batch_size, num_steps, initializer_std
):
    point = GATE_1_POINTS[index]
    model_cfg, optimizer_cfg = gate_1_recipe(point)
    step = gate_1_steps[index]
    config = step.config
    assert isinstance(config.model, VersionedValue)
    assert isinstance(config.run_id, VersionedValue)

    assert point.hidden_dim == hidden_dim
    assert config.run_id.value == expected_run_id
    assert model_cfg.hidden_dim == hidden_dim
    assert model_cfg.inferred_head_dim == 128
    assert model_cfg.initializer_std == pytest.approx(initializer_std)
    assert model_cfg.disable_long_rope is True
    assert model_cfg.disable_pko is True
    assert model_cfg.qk_mult == pytest.approx(128**-0.5)
    assert model_cfg.attention_qk_normalization is model.AttentionQKNormalization.LEARNED_RMS
    assert model_cfg.relative_position_dim == 16
    assert model_cfg.relative_position_extent == 1024
    assert model_cfg.relative_position_initializer_std == pytest.approx(0.02)
    assert model_cfg.relative_position_embedding_order is model.RelativePositionEmbeddingOrder.DIRECT
    assert model_cfg.relative_position_embedding_init is model.RelativePositionEmbeddingInit.NORMAL
    assert model_cfg.relative_query_projection_init is model.RelativeQueryProjectionInit.NORMAL
    assert optimizer_cfg.relative_query_projection_optimizer is RelativeQueryProjectionOptimizer.ADAM
    assert config.batch_size.value == batch_size
    assert config.steps.value == num_steps
    assert config.tracker.group == "MOE-JULY-ROPE-RPE-INKP2-gate1-issue-7208"
    assert "half-rope" in config.tracker.tags
    assert "learned-qk-rmsnorm" in config.tracker.tags
