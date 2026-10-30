# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.optim.grugmuon import _grug_scale_with_muon

from experiments.grug.july_muon_tpu.heuristic import MoeHeuristic
from experiments.grug.july_muon_tpu.optimizer import JulyMuonHConfig, JulyMuonWConfig


def params_and_grads():
    params = {
        "token_embed": jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 30 + 0.1,
        "lm_head": jnp.arange(24, dtype=jnp.float32).reshape(4, 6) / 30 + 0.1,
        "block": {
            "router": jnp.ones((4, 3)) / 5,
            "norm": jnp.ones(4),
            "bias": jnp.zeros(4),
            "q": jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 40 + 0.1,
            "experts": jnp.arange(96, dtype=jnp.float32).reshape(3, 4, 8) / 100 + 0.1,
        },
    }
    keys = jax.random.split(jax.random.PRNGKey(31), len(jax.tree.leaves(params)))
    grads = jax.tree.unflatten(
        jax.tree.structure(params),
        [jax.random.normal(k, p.shape) / 10 for k, p in zip(keys, jax.tree.leaves(params), strict=True)],
    )
    return params, grads


def test_muonh_matches_july_reference_through_warmup_and_decay():
    base = MoeHeuristic().build_optimizer_config(16, 10980 * 16 * 8192, 512, seq_len=8192)
    # A short schedule exercises the same scheduler on both sides through all phases.
    configs = [base, JulyMuonHConfig(**asdict(base), head_lr=base.learning_rate)]
    transforms = [config.build(12) for config in configs]
    params, grads = params_and_grads()
    states = [tx.init(params) for tx in transforms]
    for _ in range(12):
        results = [tx.update(grads, state, params) for tx, state in zip(transforms, states, strict=True)]
        states = [r[1] for r in results]
        for a, b in zip(jax.tree.leaves(results[0][0]), jax.tree.leaves(results[1][0]), strict=True):
            np.testing.assert_array_equal(a, b)
        params = optax.apply_updates(params, results[0][0])


def test_muonw_fallback_is_adamw_and_matrix_decay_is_decoupled():
    config = JulyMuonWConfig(
        learning_rate=0.02,
        head_lr=0.03,
        adam_lr=0.004,
        weight_decay=0.1,
        warmup=0,
        min_lr_ratio=1,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
    )
    tx = config.build(20)
    params, grads = params_and_grads()
    labels = config.create_mask(params)
    states = tx.init(params)
    adam_refs = {
        group: optax.adamw(
            lr, b1=0.9, b2=0.99, eps=1e-8, weight_decay=0.1, mask=jax.tree.map(lambda p: p.ndim >= 2, params)
        )
        for group, lr in [("adam", 0.004), ("adamh", 0.03)]
    }
    adam_states = {g: ref.init(params) for g, ref in adam_refs.items()}
    direction = _grug_scale_with_muon(
        momentum=0.95, nesterov=True, steps=5, muon_eps=1e-8, use_kimi_scaling=False, coefficient_type="quintic"
    )
    matrix = {k: params["block"][k] for k in ("q", "experts")}
    direction_state = direction.init(matrix)
    for _ in range(3):
        updates, states = tx.update(grads, states, params)
        for group, ref in adam_refs.items():
            expected, adam_states[group] = ref.update(grads, adam_states[group], params)
            for label, actual, want in zip(
                jax.tree.leaves(labels), jax.tree.leaves(updates), jax.tree.leaves(expected), strict=True
            ):
                if label == group:
                    np.testing.assert_allclose(actual, want, rtol=2e-6, atol=1e-8)
        matrix = {k: params["block"][k] for k in matrix}
        dirs, direction_state = direction.update({k: grads["block"][k] for k in matrix}, direction_state, matrix)
        for k in matrix:
            np.testing.assert_allclose(updates["block"][k], -0.02 * (dirs[k] + 0.1 * matrix[k]), rtol=2e-6, atol=1e-8)
        params = optax.apply_updates(params, updates)


def test_sweep_keeps_fallback_updates_fixed():
    params, grads = params_and_grads()
    for cls in (JulyMuonHConfig, JulyMuonWConfig):
        config = cls(learning_rate=0.01, head_lr=0.01, adam_lr=0.002, warmup=0)
        transforms = [
            replace(config, learning_rate=config.learning_rate * multiplier).build(100) for multiplier in (0.25, 4)
        ]
        updates = [tx.update(grads, tx.init(params), params)[0] for tx in transforms]
        for key in ("token_embed", "lm_head"):
            np.testing.assert_array_equal(updates[0][key], updates[1][key])
        for key in ("router", "norm", "bias"):
            np.testing.assert_array_equal(updates[0]["block"][key], updates[1]["block"][key])
        assert not np.allclose(updates[0]["block"]["q"], updates[1]["block"]["q"])
