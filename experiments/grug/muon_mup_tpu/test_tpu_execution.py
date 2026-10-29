# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.fast_track.model import _bincount_upper_quantile
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
