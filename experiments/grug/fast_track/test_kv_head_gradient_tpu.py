# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU interpreter checks; these do not qualify TPU lowering or performance."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from experiments.grug.fast_track.kv_head_gradient_tpu import expand_kv_fp32, sum_head_gradients


def test_head_gradient_preserves_small_residual_after_cancellation():
    # Rounding each operand first loses this representable BF16 residual.
    gradient = np.empty((1, 128, 2, 128), np.float32)
    gradient[:, :, 0] = 1 + 1 / 256
    gradient[:, :, 1] = -1
    with jax.default_device(jax.devices("cpu")[0]):
        actual = sum_head_gradients(jnp.asarray(gradient), 1, interpret=True)
    np.testing.assert_array_equal(np.asarray(actual).astype(np.float32), np.full((1, 128, 1, 128), 1 / 256))


@pytest.mark.parametrize("batch,kv_heads,repeats", [(1, 1, 1), (1, 1, 2), (2, 2, 3), (1, 2, 4)])
def test_kv_expansion_adjoint_preserves_head_groups(batch, kv_heads, repeats):
    rng = np.random.default_rng(0)
    source = rng.normal(size=(batch, 128, kv_heads, 128)).astype(ml_dtypes.bfloat16)
    # Dyadic contributions make every FP32 addition exact, including cancellation.
    cotangent = rng.integers(-1024, 1025, size=(batch, 128, kv_heads * repeats, 128)).astype(np.float32) / 256
    expected = np.empty(source.shape, np.float64)
    for head in range(kv_heads):
        expected[:, :, head] = cotangent[:, :, head * repeats : (head + 1) * repeats].astype(np.float64).sum(axis=2)
    expected = expected.astype(ml_dtypes.bfloat16)
    with jax.default_device(jax.devices("cpu")[0]):
        output, pullback = jax.vjp(lambda x: expand_kv_fp32(x, kv_heads * repeats, interpret=True), jnp.asarray(source))
        actual = pullback(jnp.asarray(cotangent))[0]
    np.testing.assert_array_equal(np.asarray(output), np.repeat(source.astype(np.float32), repeats, axis=2))
    np.testing.assert_array_equal(np.asarray(actual), expected)
