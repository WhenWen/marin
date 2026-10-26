# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check fused expert values and all gradients across tile/group boundaries."""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.belay_tpu.fused_row_relu2 import fused_row_relu2


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("groups", [(63, 0, 271, 178), (0, 512, 0, 0)])
@pytest.mark.parametrize("rejected", [0, 179, 512])
def test_local_fused_relu2_matches_independent_expert_slices(dtype, groups, rejected):
    rng = np.random.default_rng(445)
    x = jnp.asarray(rng.normal(scale=0.1, size=(512, 128)), dtype)
    up = jnp.asarray(rng.normal(scale=0.05, size=(4, 128, 256)), dtype)
    down = jnp.asarray(rng.normal(scale=0.05, size=(4, 256, 128)), dtype)
    gradient = jnp.asarray(rng.normal(scale=0.1, size=x.shape), dtype)
    active = jnp.arange(512)[:, None] < 512 - rejected

    def candidate(x, up, down):
        output = fused_row_relu2(
            jnp.where(active, x, 0), up, down, jnp.asarray(groups, jnp.int32), interpret=True, tile_rows=128
        )
        return jnp.where(active, output, 0)

    def reference(x, up, down):
        x = jnp.where(active, x, 0)
        parts = []
        for expert, (lo, hi) in enumerate(pairwise(np.cumsum([0, *groups]))):
            pre = (x[lo:hi].astype(jnp.float32) @ up[expert].astype(jnp.float32)).astype(dtype)
            hidden = jnp.square(jax.nn.relu(pre))
            parts.append((hidden.astype(jnp.float32) @ down[expert].astype(jnp.float32)).astype(dtype))
        return jnp.where(active, jnp.concatenate(parts), 0)

    with jax.default_matmul_precision("highest"):
        output, backward = jax.vjp(candidate, x, up, down)
        expected, expected_backward = jax.vjp(reference, x, up, down)
        values = (output, *backward(gradient))
        references = (expected, *expected_backward(gradient))
    tolerance = 1e-4 if dtype == jnp.float32 else 2e-2
    for actual, wanted in zip(values, references, strict=True):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(
            np.asarray(actual, np.float32), np.asarray(wanted, np.float32), atol=tolerance, rtol=tolerance
        )
