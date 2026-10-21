# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.stiefel import semiorthogonal_init, skewon_tangent, stiefel_step


@pytest.mark.parametrize("shape", [(8, 4), (4, 8), (2, 8, 4), (2, 8, 16, 4), (2, 8, 4, 16)])
def test_sharded_manifold_init_and_steps_keep_maximum_effective_rank(shape):
    mesh = Mesh(np.asarray(jax.devices()), ("data",), axis_types=(AxisType.Explicit,))
    spec = P(*([None] * (len(shape) - 2)), "data", None)
    weight = jax.device_put(jax.random.normal(jax.random.key(5), shape) * 0.2, NamedSharding(mesh, spec))
    expected_norm = np.linalg.norm(np.asarray(weight), axis=(-2, -1))
    with jax.set_mesh(mesh):
        value = jax.jit(semiorthogonal_init)(weight)
        for step in range(5):
            grad = jax.device_put(jax.random.normal(jax.random.key(step), shape), NamedSharding(mesh, spec))
            value += jax.jit(stiefel_step)(value, grad, 0.03)
    assert value.sharding.spec == spec
    singular = np.linalg.svd(np.asarray(value), compute_uv=False)
    np.testing.assert_allclose(
        singular,
        np.broadcast_to(expected_norm[..., None] / np.sqrt(min(shape[-2:])), singular.shape),
        rtol=2e-5,
        atol=2e-6,
    )


def test_reduced_skewon_matches_full_tangent_and_is_descent():
    x = semiorthogonal_init(jax.random.normal(jax.random.key(1), (2, 40, 8)))
    x /= jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) / np.sqrt(8)
    m = jax.random.normal(jax.random.key(2), x.shape)
    small = skewon_tangent(x, m)
    full = skewon_tangent(x, m, reduced=False)
    np.testing.assert_allclose(small, full, atol=2e-6, rtol=2e-5)
    tangent_error = x.swapaxes(-1, -2) @ small + small.swapaxes(-1, -2) @ x
    np.testing.assert_allclose(tangent_error, 0, atol=2e-6)
    assert float(jnp.sum(m * small)) < 0


def test_release_preserves_momentum_and_adam_groups_then_allows_rank_change():
    config = GrugMoeMuonHConfig(learning_rate=0.02, adam_lr=0.001, warmup=0, lr_schedule="constant", muon_bimaxwell=True)
    shaped_config = dataclasses.replace(config, stiefel_active_fraction=0.25)
    params = {
        key: jax.random.normal(jax.random.key(i), shape) * 0.1
        for i, (key, shape) in enumerate(
            [("matrix", (8, 4)), ("token_embed", (8, 4)), ("output_proj", (8, 4)), ("norm", (4,))]
        )
    }
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    params = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())), params)
    with jax.set_mesh(mesh):
        original = params
        params = shaped_config.initialize_stiefel_weights(params)
        for key in ("token_embed", "output_proj", "norm"):
            np.testing.assert_array_equal(params[key], original[key])
        baseline, shaped = config.build(8), shaped_config.build(8)
        base_state, shaped_state = baseline.init(params), shaped.init(params)
        for step in range(5):
            grads = jax.tree.map(lambda x, step=step: jnp.sin(x * 7 + step), params)
            base_update, base_state = jax.jit(baseline.update)(grads, base_state, params)
            shaped_update, shaped_state = jax.jit(shaped.update)(grads, shaped_state, params)
            for key in ("token_embed", "output_proj", "norm"):
                np.testing.assert_array_equal(shaped_update[key], base_update[key])
            if step >= 2:
                np.testing.assert_allclose(shaped_update["matrix"], base_update["matrix"], rtol=2e-6, atol=2e-7)
            params = optax.apply_updates(params, shaped_update)
            singular = np.linalg.svd(np.asarray(params["matrix"]), compute_uv=False)
            if step < 2:
                np.testing.assert_allclose(singular, np.full_like(singular, singular.mean()), rtol=2e-5)
            else:
                assert singular.max() - singular.min() > 1e-4
