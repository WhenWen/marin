# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental shared-KV gradient reduction with one BF16 output conversion.

Each chip-local program owns 128 rows of one KV head's gradient. Query-head
contributions are read and added in FP32 inside Pallas, then stored in BF16.
The input transpose can add HBM traffic; this module has no throughput claim.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from levanter.grug.attention._core import align_kv_heads

ROWS = 128
HEAD_DIM = 128


def _sum_heads(input_ref, output_ref):
    total = input_ref[0, :, :]
    for head in range(1, input_ref.shape[0]):
        total = total + input_ref[head, :, :]
    output_ref[...] = total.astype(jnp.bfloat16)


def sum_head_gradients(gradient, num_kv_heads, *, interpret=False):
    """Sum contiguous query-head groups, preserving FP32 until the BF16 store.

    Args:
        gradient: Chip-local FP32 cotangent with shape [B, S, Hq, 128].
        num_kv_heads: Number of shared KV heads, dividing Hq.
        interpret: Execute the Pallas interpreter for non-training CPU checks.
    """
    batch, length, query_heads, dim = gradient.shape
    if gradient.dtype != jnp.float32 or dim != HEAD_DIM:
        raise ValueError("Head-gradient reduction requires FP32 inputs and head dimension 128")
    if num_kv_heads <= 0 or query_heads % num_kv_heads:
        raise ValueError("Query heads must be divisible by the positive KV-head count")
    repeats = query_heads // num_kv_heads
    if repeats == 1:
        return gradient.astype(jnp.bfloat16)
    rows = batch * length * num_kv_heads
    if rows % ROWS:
        raise ValueError("The chip-local row count must be divisible by 128")
    inputs = gradient.reshape(rows, repeats, dim).transpose(1, 0, 2)
    output = pl.pallas_call(
        _sum_heads,
        out_shape=jax.ShapeDtypeStruct((rows, dim), jnp.bfloat16),
        grid=(rows // ROWS,),
        in_specs=[pl.BlockSpec((repeats, ROWS, dim), lambda tile: (0, tile, 0))],
        out_specs=pl.BlockSpec((ROWS, dim), lambda tile: (tile, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        cost_estimate=pl.CostEstimate(
            flops=rows * dim * (repeats - 1),
            transcendentals=0,
            bytes_accessed=4 * rows * repeats * dim + 2 * rows * dim,
        ),
        interpret=interpret,
        name="kv_head_gradient_fp32_sum",
    )(inputs)
    return output.reshape(batch, length, num_kv_heads, dim)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _expand(x, num_query_heads, num_kv_heads, interpret):
    del num_kv_heads, interpret
    return align_kv_heads(x.astype(jnp.float32), num_q_heads=num_query_heads)


def _expand_fwd(x, num_query_heads, num_kv_heads, interpret):
    return _expand(x, num_query_heads, num_kv_heads, interpret), None


def _expand_bwd(num_query_heads, num_kv_heads, interpret, residual, gradient):
    del num_query_heads, residual
    return (sum_head_gradients(gradient, num_kv_heads, interpret=interpret),)


_expand.defvjp(_expand_fwd, _expand_bwd)


def expand_kv_fp32(x, num_query_heads, *, interpret=False):
    """Broadcast BF16 KV heads in FP32 with an explicitly accumulated adjoint."""
    if x.dtype != jnp.bfloat16:
        raise ValueError("The experimental KV expansion expects BF16 model inputs")
    return _expand(x, num_query_heads, x.shape[2], interpret)
