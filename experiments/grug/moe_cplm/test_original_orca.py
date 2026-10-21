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

from experiments.grug.moe_cplm.original_orca import BelayOrcaConfig, scale_original_orca


@pytest.mark.parametrize(
    "shape,shard_axis", [((8, 4), -2), ((4, 8), -2), ((2, 8, 4), -2), ((2, 8, 8, 4), -2), ((2, 8, 4, 8), -1)]
)
def test_original_orca_decay_shaping_release_and_update_scale(shape, shard_axis):
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(-1, 1, 1), ("data", "model", "expert"), axis_types=(AxisType.Explicit,) * 3
    )
    axes = [None] * len(shape)
    axes[shard_axis] = "data"
    spec = P(*axes)
    w = jax.device_put(jax.random.normal(jax.random.key(3), shape) * 0.2, NamedSharding(mesh, spec))
    g = jax.device_put(jax.random.normal(jax.random.key(7), shape), NamedSharding(mesh, spec))
    lr = jax.device_put(np.array(0.016, np.float32), NamedSharding(mesh, P()))
    with jax.set_mesh(mesh):
        tx = scale_original_orca(lr, strength=1.0, weight_decay=0.1, release_step=2, head_dim=2)
        state = tx.init({"matrix": w})
        update = jax.jit(tx.update)
        for step in range(4):
            delta, state = update({"matrix": g}, state, {"matrix": w})
            wa = np.asarray(w) * (1 - 0.016 * 0.1)
            correction = (wa @ wa.swapaxes(-1, -2)) @ wa - wa if step < 2 else np.zeros_like(wa)
            # Recover the task direction independently from the full parameter delta.
            task = -np.asarray(delta["matrix"]) / 0.016 - 0.1 * np.asarray(w) - correction
            np.testing.assert_allclose(np.sqrt(np.mean(task**2, axis=(-2, -1))), 0.2, atol=3e-7)
            assert np.sum(task * np.asarray(g)) > 0
            assert delta["matrix"].sharding.spec == spec


def test_uniform_release_and_headwise_shaping_with_zero_task_gradient():
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(-1, 1, 1), ("data", "model", "expert"), axis_types=(AxisType.Explicit,) * 3
    )
    p = {"stacked_blocks": {"stacked": {"attn": {"w_q": jnp.arange(128, dtype=jnp.float32).reshape(2, 8, 8) / 500}}}}
    p = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P(None, "data", None))), p)
    with jax.set_mesh(mesh):
        tx = scale_original_orca(
            0.016,
            strength=1.0,
            weight_decay=0.1,
            release_step=2,
            head_dim=4,
        )
        state = tx.init(p)
        update = jax.jit(tx.update)
        for step in range(7):
            delta, state = update(jax.tree.map(jnp.zeros_like, p), state, p)
            w = np.asarray(p["stacked_blocks"]["stacked"]["attn"]["w_q"])
            expected = -0.016 * 0.1 * w
            for layer in range(2):
                if step < 2:
                    for start in (0, 4):
                        wa = w[layer, :, start : start + 4] * (1 - 0.016 * 0.1)
                        expected[layer, :, start : start + 4] -= 0.016 * ((wa @ wa.T) @ wa - wa)
            np.testing.assert_allclose(delta["stacked_blocks"]["stacked"]["attn"]["w_q"], expected, atol=1e-8, rtol=2e-5)


def test_fallback_is_adamw_and_independent_of_matrix_lr():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    p = {"matrix": jnp.ones((4, 8)) * 0.2, "output_proj": jnp.ones((4, 8)) * 0.2, "norm": jnp.ones(8) * 0.2}
    p = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())), p)
    with jax.set_mesh(mesh):
        g = jax.tree.map(lambda x: jnp.ones_like(x) * 0.01, p)
        cfg = BelayOrcaConfig(learning_rate=0.004, warmup=0, decay=0, lr_schedule="constant")
        low, high = cfg.build(10), dataclasses.replace(cfg, learning_rate=0.016).build(10)
        a, _ = jax.jit(low.update)(g, low.init(p), p)
        b, _ = jax.jit(high.update)(g, high.init(p), p)
        ref = optax.adamw(
            cfg.adam_lr, b1=cfg.grouping.beta1, b2=cfg.grouping.beta2, eps=cfg.grouping.epsilon, weight_decay=0.1
        )
        expected, _ = ref.update(g["output_proj"], ref.init(p["output_proj"]), p["output_proj"])
        np.testing.assert_allclose(a["output_proj"], expected, atol=1e-8)
        np.testing.assert_array_equal(a["output_proj"], b["output_proj"])
        np.testing.assert_array_equal(a["norm"], b["norm"])
        assert not np.allclose(a["matrix"], b["matrix"])
