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
from experiments.grug.muon_mup_tpu.optimizer import MuonComparisonConfig, muon_step


@pytest.mark.parametrize("shape", [(8, 4), (4, 8), (8, 8), (2, 8, 4), (2, 3, 4, 8)])
@pytest.mark.parametrize("spectral", [False, True])
def test_muon_direction_scaling_and_decoupled_decay(shape, spectral):
    p = jax.random.normal(jax.random.PRNGKey(1), shape)
    g = jax.random.normal(jax.random.PRNGKey(2), shape)
    reference = _grug_scale_with_muon()
    direction, _ = reference.update(g, reference.init(p), p)
    tx = muon_step(
        learning_rate=0.02,
        weight_decay=0.1,
        spectral=spectral,
        momentum=0.95,
        nesterov=True,
        steps=5,
        eps=1e-8,
        coefficient_type="quintic",
    )
    actual, _ = tx.update(g, tx.init(p), p)
    direction_only = -actual / 0.02 - 0.1 * p
    expected_ratio = np.sqrt(min(1, shape[-1] / shape[-2])) if spectral else 1
    np.testing.assert_allclose(direction_only, direction * expected_ratio, atol=2e-7, rtol=2e-5)
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
            tx = muon_step(learning_rate=0.01, weight_decay=0.1, spectral=True,
                           momentum=0.95, nesterov=True, steps=5, eps=1e-8, coefficient_type='quintic')
            updates, _ = jax.jit(tx.update)(g, tx.init(p), p)
            assert updates.sharding == p.sharding
            assert bool(jnp.isfinite(updates).all())
            assert float(jnp.max(jnp.abs(updates))) > 0
        """,
        device_count=4,
    )
