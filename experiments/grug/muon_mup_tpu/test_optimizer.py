# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.muon_mup_tpu.optimizer import (
    MuonComparisonConfig,
    MuonHSweepConfig,
    MuonMDConfig,
    MuonWComparisonConfig,
    MuownConfig,
    NorMuonHConfig,
    RowNormalizedMuonHConfig,
    muon_step,
)
from experiments.grug.muon_mup_tpu.row_normalized import normalize_neuron_updates


@pytest.mark.parametrize("shape", [(8, 4), (4, 8), (8, 8), (2, 8, 4), (2, 3, 4, 8)])
def test_muon_direction_scaling_and_decoupled_decay(shape):
    p = jax.random.normal(jax.random.PRNGKey(1), shape)
    g = jax.random.normal(jax.random.PRNGKey(2), shape)
    reference = _grug_scale_with_muon()
    direction, _ = reference.update(g, reference.init(p), p)
    tx = muon_step(
        learning_rate=0.02,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        steps=5,
        eps=1e-8,
        coefficient_type="quintic",
    )
    actual, _ = tx.update(g, tx.init(p), p)
    direction_only = -actual / 0.02 - 0.1 * p
    np.testing.assert_allclose(direction_only, direction, atol=2e-7, rtol=2e-5)
    zero_update, _ = tx.update(jnp.zeros_like(g), tx.init(p), p)
    np.testing.assert_allclose(optax.apply_updates(p, zero_update), 0.998 * p, atol=3e-7, rtol=1e-6)


def test_adam_and_adamh_groups_match_original_across_steps():
    params = {
        "output_proj": jnp.eye(8),
        "token_embed": jnp.ones((16, 8)),
        "attn_res_query_final": jnp.ones(8),
        "layer": {"router": jnp.ones((8, 4)), "attn_gate": jnp.ones((8, 2))},
    }
    baseline = GrugMoeMuonHConfig(learning_rate=0.02, adam_lr=0.005).build(100)
    candidate = MuonComparisonConfig(learning_rate=0.01, head_lr=0.02, adam_lr=0.005).build(100)
    left_state, right_state = baseline.init(params), candidate.init(params)
    for _ in range(4):
        grads = jax.tree.map(lambda p: jnp.sin(p) + 0.1, params)
        left, left_state = baseline.update(grads, left_state, params)
        right, right_state = candidate.update(grads, right_state, params)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
            np.testing.assert_array_equal(a, b)
        params = optax.apply_updates(params, left)


def test_expert_update_preserves_sharding_before_weight_decay():
    run_on_cpu_devices(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, AxisType, PartitionSpec as P
        from haliax.partitioning import set_mesh
        from experiments.grug.muon_mup_tpu.optimizer import muon_step

        mesh = Mesh(np.array(jax.devices()).reshape(1,4,1,1),
                    ('replica_dcn','data','expert','model'), axis_types=(AxisType.Explicit,)*4)
        with set_mesh(mesh):
            p = jax.sharding.reshard(jnp.ones((2,4,8,16)), P(None,'expert','model','data'))
            g = jax.sharding.reshard(jax.random.normal(jax.random.PRNGKey(0), p.shape), p.sharding)
            tx = muon_step(learning_rate=0.01, weight_decay=0.1,
                           momentum=0.95, nesterov=True, steps=5, eps=1e-8, coefficient_type='quintic')
            updates, _ = jax.jit(tx.update)(g, tx.init(p), p)
            assert updates.sharding == p.sharding
            assert bool(jnp.isfinite(updates).all())
            assert float(jnp.max(jnp.abs(updates))) > 0
        """,
        device_count=4,
    )


@pytest.mark.parametrize("multiplier", [0.25, 1.0, 4.0])
def test_muonh_sweep_preserves_reference_updates_and_fixed_adam_groups(multiplier):
    params = {
        "output_proj": jax.random.normal(jax.random.PRNGKey(0), (8, 4)),
        "token_embed": jnp.ones((16, 8)),
        "attn_res_query_final": jnp.ones(8),
        "layer": {
            "router": jnp.ones((8, 4)),
            "matrix": jax.random.normal(jax.random.PRNGKey(1), (2, 8, 4)),
            "experts": jax.random.normal(jax.random.PRNGKey(2), (2, 3, 4, 8)),
        },
        "kda_blocks": {"stacked": {"attn": {"w_beta": jnp.ones((2, 8, 4))}}},
    }
    common = dict(adam_lr=0.005, warmup=0.2, lr_schedule="linear", min_lr_ratio=0.05)
    fixed_config = GrugMoeMuonHConfig(learning_rate=0.02, **common)
    fixed = fixed_config.build(20)
    scaled = GrugMoeMuonHConfig(learning_rate=0.02 * multiplier, **common).build(20)
    candidate = MuonHSweepConfig(learning_rate=0.02 * multiplier, head_lr=0.02, **common).build(20)
    fixed_state, scaled_state, candidate_state = [tx.init(params) for tx in (fixed, scaled, candidate)]
    labels = jax.tree.leaves(fixed_config.create_mask(params))
    for _ in range(8):
        grads = jax.tree.map(lambda p: jnp.sin(p) + 0.1, params)
        fixed_updates, fixed_state = fixed.update(grads, fixed_state, params)
        scaled_updates, scaled_state = scaled.update(grads, scaled_state, params)
        updates, candidate_state = candidate.update(grads, candidate_state, params)
        for group, original, changed, actual in zip(
            labels,
            jax.tree.leaves(fixed_updates),
            jax.tree.leaves(scaled_updates),
            jax.tree.leaves(updates),
            strict=True,
        ):
            expected = changed if group in ("muonh", "kda_beta") else original
            np.testing.assert_array_equal(actual, expected)
        params = optax.apply_updates(params, updates)


@pytest.mark.parametrize("config_type", [MuonWComparisonConfig, MuownConfig, MuonMDConfig])
def test_muonw_decays_logical_matrices_and_keeps_stacked_vectors_undecayed(config_type):
    params = {
        "output_proj": jnp.ones((8, 16)),
        "token_embed": jnp.ones((16, 8)),
        "attn_res_query_final": jnp.ones(8),
        "kda_blocks": {
            "stacked": {
                "rms_attn": {"weight": jnp.ones((2, 8))},
                "attn_res_query_attn": jnp.ones((2, 8)),
                "mlp": {"router": jnp.ones((2, 8, 4)), "router_bias": jnp.ones((2, 4))},
                "attn": {"a_log": jnp.ones((2, 4)), "w_a_down": jnp.ones((2, 8, 4))},
            }
        },
    }
    config = config_type(
        learning_rate=0.01, head_lr=0.02, adam_lr=0.005, warmup=0, min_lr_ratio=1, beta1=0.9, beta2=0.99, epsilon=1e-8
    )
    tx = config.build(100)
    labels = config.create_mask(params)
    # Independent expected decay policy: vectors remain vectors after layer stacking.
    decay = {
        "output_proj": True,
        "token_embed": True,
        "attn_res_query_final": False,
        "kda_blocks": {
            "stacked": {
                "rms_attn": {"weight": False},
                "attn_res_query_attn": False,
                "mlp": {"router": True, "router_bias": False},
                "attn": {"a_log": False, "w_a_down": True},
            }
        },
    }
    references = {
        group: optax.adamw(lr, b1=0.9, b2=0.99, eps=1e-8, weight_decay=0.1, mask=decay)
        for group, lr in [("adamh", 0.02), ("adam", 0.005), ("attn_res_query", 0.0005)]
    }
    state = tx.init(params)
    ref_states = {group: ref.init(params) for group, ref in references.items()}
    for _ in range(3):
        grads = jax.tree.map(lambda p: 0.2 * jnp.cos(p), params)
        actual, state = tx.update(grads, state, params)
        for group, ref in references.items():
            expected, ref_states[group] = ref.update(grads, ref_states[group], params)
            for label, a, e in zip(
                jax.tree.leaves(labels), jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
            ):
                if label == group:
                    np.testing.assert_allclose(a, e, rtol=2e-6, atol=1e-8)
        params = optax.apply_updates(params, actual)


@pytest.mark.parametrize("shape", [(7, 4), (4, 7), (2, 7, 4), (2, 3, 7, 4)])
def test_row_weight_muonh_preserves_each_neuron_radius(shape):
    params = jax.random.normal(jax.random.PRNGKey(30), shape) * jnp.arange(1, shape[-1] + 1)
    radii = np.linalg.norm(np.asarray(params), axis=-2)
    config = RowNormalizedMuonHConfig()
    tx = config.matrix_transform(0.07)
    reference = _grug_scale_with_muon()
    state, reference_state = tx.init(params), reference.init(params)
    for index in range(3):
        grads = jax.random.normal(jax.random.PRNGKey(31 + index), shape)
        direction, reference_state = reference.update(grads, reference_state, params)
        update, state = tx.update(grads, state, params)
        # Independent per-neuron vector geometry, in conventional [out, in] layout.
        p = np.swapaxes(np.asarray(params), -1, -2).reshape(-1, shape[-2])
        u = np.swapaxes(np.asarray(direction), -1, -2).reshape(-1, shape[-2])
        expected = []
        for weight, delta in zip(p, u, strict=True):
            radius = np.linalg.norm(weight)
            vector = weight / radius - 0.07 * delta / np.linalg.norm(delta)
            expected.append(radius * vector / np.linalg.norm(vector))
        actual = optax.apply_updates(params, update)
        np.testing.assert_allclose(
            np.swapaxes(np.asarray(actual), -1, -2).reshape(p.shape), expected, atol=2e-6, rtol=2e-6
        )
        np.testing.assert_allclose(np.linalg.norm(np.asarray(actual), axis=-2), radii, atol=3e-6, rtol=2e-6)
        params = actual


def test_normuon_row_moments_match_official_matrix_reference():
    shape = (2, 3, 7, 4)
    tx = normalize_neuron_updates(0.95)
    state = tx.init(jnp.zeros(shape))
    moments = np.zeros((6, 4, 1), dtype=np.float32)
    for index in range(3):
        updates = jax.random.normal(jax.random.PRNGKey(50 + index), shape)
        actual, state = tx.update(updates, state, jnp.zeros(shape))
        # Official NorMuon normalization on a batch of [out, in] matrices.
        matrices = np.swapaxes(np.asarray(updates), -1, -2).reshape(6, 4, 7)
        moments = 0.95 * moments + 0.05 * (matrices**2).mean(axis=-1, keepdims=True)
        normalized = matrices / (np.sqrt(moments) + 1e-10)
        expected = normalized * (
            np.linalg.norm(matrices, axis=(-2, -1), keepdims=True)
            / (np.linalg.norm(normalized, axis=(-2, -1), keepdims=True) + 1e-10)
        )
        np.testing.assert_allclose(
            np.swapaxes(np.asarray(actual), -1, -2).reshape(6, 4, 7), expected, atol=1e-6, rtol=2e-6
        )


@pytest.mark.parametrize("config_type", [NorMuonHConfig, RowNormalizedMuonHConfig])
def test_neuron_variants_preserve_auxiliary_groups(config_type):
    params = {
        "output_proj": jnp.eye(4),
        "token_embed": jnp.ones((8, 4)),
        "attn_res_query_final": jnp.ones(4),
        "layer": {"router": jnp.ones((4, 2))},
    }
    common = dict(learning_rate=0.02, head_lr=0.03, adam_lr=0.005, warmup=0)
    reference, candidate = MuonHSweepConfig(**common).build(10), config_type(**common).build(10)
    left_state, right_state = reference.init(params), candidate.init(params)
    for _ in range(3):
        grads = jax.tree.map(jnp.cos, params)
        left, left_state = reference.update(grads, left_state, params)
        right, right_state = candidate.update(grads, right_state, params)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
            np.testing.assert_array_equal(a, b)
        params = optax.apply_updates(params, left)


def test_neuron_variants_match_unsharded_updates_and_state():
    run_on_cpu_devices(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, AxisType, PartitionSpec as P
        from haliax.partitioning import set_mesh
        from experiments.grug.muon_mup_tpu.optimizer import NorMuonHConfig, RowNormalizedMuonHConfig
        mesh = Mesh(np.array(jax.devices()).reshape(1,2,2,1),
                    ('replica_dcn','data','expert','model'), axis_types=(AxisType.Explicit,)*4)
        with set_mesh(mesh):
            original = jax.random.normal(jax.random.PRNGKey(0), (2,4,8,16))
            for spec in (P(None,'expert','model','data'), P(None,'expert','data','model')):
                p = jax.sharding.reshard(original, spec)
                replicated = jax.sharding.reshard(original, P())
                for config in (NorMuonHConfig(), RowNormalizedMuonHConfig()):
                    tx = config.matrix_transform(0.03)
                    state, reference_state = tx.init(p), tx.init(replicated)
                    for i in range(3):
                        gradient = jax.random.normal(jax.random.PRNGKey(i+1), p.shape)
                        actual, state = jax.jit(tx.update)(jax.sharding.reshard(gradient, spec), state, p)
                        expected, reference_state = jax.jit(tx.update)(
                            jax.sharding.reshard(gradient, P()), reference_state, replicated)
                        assert actual.sharding == p.sharding
                        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
                        for a,b in zip(jax.tree.leaves(state), jax.tree.leaves(reference_state), strict=True):
                            np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
        """,
        device_count=4,
    )
