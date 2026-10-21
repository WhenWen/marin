# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.orca import orthogonal_gradient, sharded_orthogonal_gradient


@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (2, 3, 5), (2, 8, 5, 3)])
def test_orthogonal_gradient_matches_autodiff(shape):
    weight = jax.random.normal(jax.random.key(3), shape) * 0.2

    def penalty(w):
        matrices = w.reshape(-1, *shape[-2:])
        if shape[-2] > shape[-1]:
            matrices = matrices.swapaxes(-1, -2)
        return jnp.sum(jax.vmap(lambda m: jnp.sum((m @ m.T - jnp.eye(m.shape[0])) ** 2))(matrices)) / 4

    np.testing.assert_allclose(orthogonal_gradient(weight), jax.grad(penalty)(weight), atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("shape", [(2, 8, 8, 4), (2, 8, 4, 8), (2, 8, 4), (8, 4)])
def test_sharded_orthogonal_gradient_preserves_values_and_layout(shape):
    count = len(jax.devices())
    mesh = Mesh(np.asarray(jax.devices()), ("data",), axis_types=(AxisType.Explicit,))
    weight = jax.random.normal(jax.random.key(5), shape) * 0.2
    expected = orthogonal_gradient(weight)
    spec = P(*([None] * (len(shape) - 2)), "data", None)
    assert shape[-2] % count == 0
    weight = jax.device_put(weight, NamedSharding(mesh, spec))
    with jax.set_mesh(mesh):
        actual = jax.jit(sharded_orthogonal_gradient)(weight)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    assert actual.sharding.spec == weight.sharding.spec


def test_orca_muonh_correction_and_hard_release_leave_adam_groups_unchanged():
    config = GrugMoeMuonHConfig(learning_rate=0.02, adam_lr=0.001, warmup=0, lr_schedule="constant")
    params = {
        "matrix": jax.random.normal(jax.random.key(1), (4, 8)) * 0.1,
        "token_embed": jnp.ones((4, 8)) * 0.1,
        "output_proj": jnp.ones((4, 8)) * 0.1,
        "norm": jnp.ones((8,)),
    }
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    params = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())), params)
    with jax.set_mesh(mesh):
        grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.01, params)
        baseline = config.build(8)
        shaped = dataclasses.replace(config, orca_strength=1.0, orca_active_fraction=0.25).build(8)
        base_state, shaped_state = baseline.init(params), shaped.init(params)
        for step in range(4):
            # Same supplied parameters isolate the update rule and momentum state
            # across the release boundary, independently of a training trajectory.
            base_update, base_state = jax.jit(baseline.update)(grads, base_state, params)
            shaped_update, shaped_state = jax.jit(shaped.update)(grads, shaped_state, params)
            expected = base_update["matrix"]
            if step < 2:
                expected = expected - config.learning_rate * orthogonal_gradient(params["matrix"])
            np.testing.assert_allclose(shaped_update["matrix"], expected, atol=1e-7, rtol=1e-6)
            for key in ("token_embed", "output_proj", "norm"):
                np.testing.assert_array_equal(shaped_update[key], base_update[key])
