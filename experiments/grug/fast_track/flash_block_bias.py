# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chip-local FlashAttention with block-sliced Inkling bias and VJP precision."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from experiments.grug.fast_track import benchmark_inkling_flash_tpu as probe

PROTOCOL = "https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6177016537"
FP32_SCALE_PROTOCOL = "https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6179473106"
BLOCK = 128


def expand_band(band, width, window):
    """Expand band rows with zero padding, then apply the sliding-window bias."""
    batch, heads, length, actual_width = band.shape
    if width != actual_width or length % BLOCK or width % BLOCK:
        raise ValueError("Bias expansion requires aligned self-attention and matching band width")
    if window is not None and window <= 0:
        raise ValueError("The sliding window must be positive")

    def block(index):
        start = index * BLOCK
        rows = jax.lax.dynamic_slice_in_dim(band, start, BLOCK, axis=2)
        padded = jnp.pad(rows, ((0, 0), (0, 0), (0, 0), (length, length)))
        values = jax.lax.dynamic_slice_in_dim(padded, length + width - BLOCK - start, length, axis=-1)
        if window is not None:
            allowed = jnp.arange(length)[None, :] >= start + jnp.arange(BLOCK)[:, None] - window + 1
            values = jnp.where(allowed, values, jnp.array(-1e9, band.dtype))
        return values

    blocks = jax.lax.map(block, jnp.arange(length // BLOCK))
    return blocks.transpose(1, 2, 0, 3, 4).reshape(batch, heads, length, length)


def project_band(gradient, width, window):
    """Project a dense bias cotangent to its represented, unmasked band entries."""
    batch, heads, length, keys = gradient.shape
    if length != keys or length % BLOCK or width % BLOCK:
        raise ValueError("Bias projection requires aligned self-attention")

    def block(index):
        start = index * BLOCK
        rows = jax.lax.dynamic_slice_in_dim(gradient, start, BLOCK, axis=2)
        if window is not None:
            allowed = jnp.arange(length)[None, :] >= start + jnp.arange(BLOCK)[:, None] - window + 1
            rows = jnp.where(allowed, rows, 0)
        padded = jnp.pad(rows, ((0, 0), (0, 0), (0, 0), (width, width)))
        return jax.lax.dynamic_slice_in_dim(padded, width + start - (width - BLOCK), width, axis=-1)

    blocks = jax.lax.map(block, jnp.arange(length // BLOCK))
    return blocks.transpose(1, 2, 0, 3, 4).reshape(batch, heads, length, width)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def block_bias(band, width, window):
    return expand_band(band, width, window)


def block_bias_fwd(band, width, window):
    return expand_band(band, width, window), None


def block_bias_bwd(width, window, residual, gradient):
    del residual
    return (project_band(gradient, width, window),)


block_bias.defvjp(block_bias_fwd, block_bias_bwd)


def attention(q, k, v, band, segments, *, window, query_scaling="bf16"):
    if q.shape[1] != k.shape[1] or q.shape[-1] != BLOCK:
        raise ValueError("This chip-local probe requires self-attention with head dimension 128")
    scale = 1 / math.sqrt(q.shape[-1])
    if query_scaling == "bf16":
        scaled_q = (q * scale).astype(q.dtype).astype(jnp.float32)
    elif query_scaling == "fp32":
        scaled_q = q.astype(jnp.float32) * scale
    else:
        raise ValueError(f"Unknown query scaling: {query_scaling}")
    output = probe.flash_core(
        scaled_q,
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        block_bias(band.astype(jnp.float32), band.shape[-1], window),
        segments,
    )
    return output.astype(q.dtype)


def values_and_gradients(q, k, v, band, segments, cotangent, *, precision, window, query_scaling="bf16"):
    """Set matrix precision while tracing both the primal and its custom VJP."""
    with jax.default_matmul_precision(precision):
        return probe.values_and_gradients(
            q, k, v, band, segments, cotangent, attention=partial(attention, window=window, query_scaling=query_scaling)
        )


@partial(jax.jit, static_argnums=(2, 3))
def layout_values(band, cotangent, width, window):
    expanded = expand_band(band, width, window)
    projected = project_band(cotangent, width, window)
    automatic = jax.vjp(partial(expand_band, width=width, window=window), band)[1](cotangent)[0]
    custom = jax.vjp(partial(block_bias, width=width, window=window), band)[1](cotangent)[0]
    return expanded, projected, automatic, custom


def layout_checks(device):
    """Yield exact layout/VJP checks and fixtures against coordinate enumeration."""
    dtypes = (
        (ml_dtypes.bfloat16, np.float32, np.float64) if device.platform == "cpu" else (ml_dtypes.bfloat16, np.float32)
    )
    for length in (128, 256, 1024):
        for width in (256, 1152):
            for window in (None, 129):
                for seed in (0, 1, 2):
                    for dtype in dtypes:
                        rng = np.random.default_rng(seed)
                        band = rng.normal(size=(1, 2, length, width)).astype(dtype)
                        cotangent = rng.normal(size=(1, 2, length, length)).astype(dtype)
                        expected = np.zeros(cotangent.shape, dtype=dtype)
                        expected_gradient = np.zeros(band.shape, dtype=dtype)
                        for query in range(length):
                            for key in range(length):
                                column = key - (query // BLOCK) * BLOCK + width - BLOCK
                                if window is not None and key < query - window + 1:
                                    expected[:, :, query, key] = np.array(-1e9, dtype=dtype)
                                elif 0 <= column < width:
                                    expected[:, :, query, key] = band[:, :, query, column]
                                    expected_gradient[:, :, query, column] = cotangent[:, :, query, key]
                        with jax.enable_x64(dtype == np.float64), jax.default_device(device):
                            inputs = jax.device_put(band, device), jax.device_put(cotangent, device)
                            expanded, projected, automatic, custom = layout_values(*inputs, width, window)
                            for actual, oracle in (
                                (expanded, expected),
                                (projected, expected_gradient),
                                (automatic, expected_gradient),
                                (custom, expected_gradient),
                            ):
                                np.testing.assert_array_equal(np.asarray(actual), oracle)
                        yield (
                            {
                                "length": length,
                                "width": width,
                                "window": window,
                                "seed": seed,
                                "dtype": np.dtype(dtype).name,
                                "backend": device.platform,
                                "exact": True,
                            },
                            {"band": band, "cotangent": cotangent},
                        )
