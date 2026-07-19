# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_identity_hyperconnection.model import (
    _identity_hc_collapse,
    _identity_hc_residual_update,
)


def test_zero_dynamic_mapping_has_paper_initial_coefficients():
    streams = jnp.arange(8, dtype=jnp.float32).reshape(1, 1, 4, 2)
    projection = jnp.zeros((8, 8), dtype=jnp.float32)
    bias = jnp.zeros((8,), dtype=jnp.float32)
    scale = jnp.full((2,), 0.01, dtype=jnp.float32)

    collapsed, pre, post = _identity_hc_collapse(
        streams,
        projection,
        bias,
        scale,
        num_streams=4,
        norm_eps=1e-6,
    )

    np.testing.assert_allclose(pre, 0.5)
    np.testing.assert_allclose(post, 1.0)
    np.testing.assert_allclose(collapsed, 0.5 * streams.sum(axis=-2))


def test_residual_update_never_mixes_streams():
    streams = jnp.arange(24, dtype=jnp.float32).reshape(1, 3, 4, 2)
    sublayer_output = jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2)
    post = jnp.array([[[0.25, 0.5, 1.0, 2.0]]], dtype=jnp.float32)

    updated = _identity_hc_residual_update(streams, sublayer_output, post)

    expected = streams + post[..., None] * sublayer_output[..., None, :]
    np.testing.assert_array_equal(updated, expected)
