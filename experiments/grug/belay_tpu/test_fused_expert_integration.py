# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check capacity masks and routing gradients around the fused expert kernel."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu import local_experts
from experiments.grug.belay_tpu.capacity_layout import factorized_route_counts, interval_route_layout
from experiments.grug.belay_tpu.fused_expert_backward import fused_expert_mlp
from experiments.grug.belay_tpu.streaming_combine import combine_streaming, dispatch_streaming


@pytest.mark.parametrize("capacity", [0, 48, 512])
def test_fused_expert_preserves_capacity_and_routing_gradients(monkeypatch, capacity):
    rng = np.random.default_rng(446)
    selected = jnp.asarray(rng.choice([0, 1, 3], size=(256, 2)), jnp.int32)
    counts = jnp.bincount(selected.reshape(-1), length=4)
    accepted = jnp.minimum(counts, capacity)
    x = jnp.asarray(rng.normal(scale=0.1, size=(256, 128)), jnp.bfloat16)
    up = jnp.asarray(rng.normal(scale=0.05, size=(4, 128, 256)), jnp.bfloat16)
    down = jnp.asarray(rng.normal(scale=0.05, size=(4, 256, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.uniform(0.1, 0.9, size=selected.shape), jnp.float32)
    dy = jnp.asarray(rng.normal(scale=0.1, size=x.shape), jnp.bfloat16)
    monkeypatch.setattr(local_experts, "route_counts", factorized_route_counts)
    monkeypatch.setattr(local_experts, "route_layout", interval_route_layout)
    monkeypatch.setattr(local_experts, "dispatch_local", dispatch_streaming)
    monkeypatch.setattr(
        local_experts, "combine_local", partial(combine_streaming, implementation="scan", scalar_permutation="sort")
    )

    def evaluate(x, up, down, weights):
        return local_experts.local_expert_mlp(x, selected, weights, up, down, accepted_counts=accepted)

    with jax.default_matmul_precision("highest"):
        reference, reference_backward, expected_keep = jax.vjp(evaluate, x, up, down, weights, has_aux=True)
        expected = (reference, *reference_backward(dy))
        monkeypatch.setattr(local_experts, "expert_mlp", partial(fused_expert_mlp, interpret=True, tile_rows=128))
        output, backward, keep = jax.vjp(evaluate, x, up, down, weights, has_aux=True)
        actual = (output, *backward(dy))
    np.testing.assert_array_equal(keep, expected_keep)
    assert int(keep.sum()) == int(accepted.sum())
    for result, wanted in zip(actual, expected, strict=True):
        assert np.isfinite(result).all()
        np.testing.assert_allclose(np.asarray(result, np.float32), np.asarray(wanted, np.float32), atol=2e-2, rtol=2e-2)
    np.testing.assert_array_equal(actual[-1][~keep], 0)
    np.testing.assert_array_equal(actual[2][2], 0)
    np.testing.assert_array_equal(actual[3][2], 0)
