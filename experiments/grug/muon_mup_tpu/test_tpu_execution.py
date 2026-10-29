# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from levanter.grug.attention._inkling_relpos import dense_rel_bias

from experiments.grug.fast_track.model import _bincount_upper_quantile
from experiments.grug.muon_mup_tpu.inkling_tpu import sliced_dense_rel_bias
from experiments.grug.muon_mup_tpu.tpu_execution import two_stage_quantile


@pytest.mark.parametrize("constant", [False, True])
def test_kma_histogram_thresholds_match_flat_counts(constant):
    # Actual B16/v4-32 shard geometry, including ties and both clipped edges.
    values = np.random.default_rng(42).normal(size=(4096, 384)).astype(np.float32)
    values[:100] = 0
    values[100:110] = -5
    values[110:120] = 5
    if constant:
        values[:] = 0
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("replica_dcn", "data", "expert"))

    def both(s):
        kwargs = dict(num_experts=384, n_bins=10000, lo=jnp.array(-2.0), hi=jnp.array(2.0), target_rank=4096 * 8 / 384)
        return _bincount_upper_quantile(s, **kwargs), two_stage_quantile(s, **kwargs)

    reference, candidate = jax.jit(jax.shard_map(both, mesh=mesh, in_specs=P(), out_specs=(P(), P())))(values)
    np.testing.assert_array_equal(reference, candidate)


@pytest.mark.parametrize("q_len,k_len,extent", [(256, 256, 128), (256, 512, 256), (512, 512, 128)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_sliced_inkling_bias_and_transpose_match_dense_gather(q_len, k_len, extent, dtype):
    band = jax.random.normal(jax.random.PRNGKey(4), (1, 2, q_len, extent + 128)).astype(dtype)
    cotangent = jax.random.normal(jax.random.PRNGKey(5), (1, 2, q_len, k_len)).astype(dtype)

    def evaluate(expand):
        fn = jax.jit(lambda x: expand(x, q_len, k_len))
        result, pullback = jax.vjp(fn, band)
        return result, pullback(cotangent)[0]

    expected, expected_gradient = evaluate(dense_rel_bias)
    actual, gradient = evaluate(sliced_dense_rel_bias)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(gradient, expected_gradient)
