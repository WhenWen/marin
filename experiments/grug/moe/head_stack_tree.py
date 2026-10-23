# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inverse-free interval composition and fused TPU preparation for head stacks."""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from experiments.grug.moe.head_stack_kda import Preparation, _coefficients, _intra


class Interval(NamedTuple):
    transport: jax.Array
    writes: jax.Array  # Transposed injection columns: [innovations, depth, key].
    erase: jax.Array
    read: jax.Array
    attention: jax.Array
    values: jax.Array
    inverse: jax.Array


def _rows(rows, transport, *, right):
    # Four scalar-vector products per output slot; no dense 512-square map.
    return jnp.stack(
        [
            sum(
                rows[..., :, j, :] * transport[..., j if right else i, i if right else j, :][..., None, :]
                for j in range(4)
            )
            for i in range(4)
        ],
        axis=-2,
    )


def _compose(right, left):
    return jnp.stack(
        [
            jnp.stack([sum(right[..., i, k, :] * left[..., k, j, :] for k in range(4)) for j in range(4)], -2)
            for i in range(4)
        ],
        -3,
    )


def _block(left, right, cross):
    zero = jnp.zeros((*cross.shape[:-2], left.shape[-2], right.shape[-1]), left.dtype)
    return jnp.concatenate((jnp.concatenate((left, zero), -1), jnp.concatenate((cross, right), -1)), -2)


def _flat(rows):
    return rows.reshape(*rows.shape[:-2], -1)


def _merge(left, right, precision):
    cross = _intra(
        jnp.concatenate((_flat(right.erase), _flat(right.read)), -2),
        jnp.swapaxes(_flat(left.writes), -1, -2),
        precision,
    )
    count = right.erase.shape[-3]
    return Interval(
        _compose(right.transport, left.transport),
        jnp.concatenate((_rows(left.writes, right.transport, right=False), right.writes), -3),
        jnp.concatenate((left.erase, _rows(right.erase, left.transport, right=True)), -3),
        jnp.concatenate((left.read, _rows(right.read, left.transport, right=True)), -3),
        _block(left.attention, right.attention, cross[..., count:, :]),
        jnp.concatenate((left.values, right.values), -2),
        _block(
            left.inverse,
            right.inverse,
            -_intra(_intra(right.inverse, cross[..., :count, :], precision), left.inverse, precision),
        ),
    )


def _leaf(token):
    actions, query, key, value, g, beta, keep = token
    push, pop, active = _coefficients(actions)
    decay = jnp.exp(g)
    eye = jnp.eye(4, dtype=jnp.float32)
    transport = eye[..., None] * (active[..., :, None] * decay)[..., :, None, :]
    transport += push[..., None, None, None] * jnp.eye(4, k=-1)[..., None]
    transport += pop[..., None, None, None] * jnp.eye(4, k=1)[..., None]
    transport *= keep[..., None, None, None]
    erase = key * decay * (active * beta * keep[..., None])[..., None]
    return Interval(
        transport,
        key[..., :, None, :] * eye[..., None],
        erase[..., :, None, :] * eye[..., None],
        query[..., :, None, :] * transport,
        jnp.sum(query * key, -1)[..., :, None] * eye,
        value * (active * beta)[..., None],
        jnp.broadcast_to(eye, (*query.shape[:-2], 4, 4)),
    )


def prepare_tree(inputs, precision="fp32"):
    """Balanced interval tree; inputs have arbitrary leading axes then C tokens."""
    length = inputs[1].shape[-3]
    if length & (length - 1):
        raise ValueError("Tree preparation requires a power-of-two chunk")
    axis = inputs[1].ndim - 3
    nodes = [_leaf(tuple(x[(slice(None),) * axis + (t,)] for x in inputs)) for t in range(length)]
    while len(nodes) > 1:
        nodes = [_merge(nodes[i], nodes[i + 1], precision) for i in range(0, len(nodes), 2)]
    node = nodes[0]
    size = 4 * inputs[1].shape[-1]
    solved = _intra(node.inverse, jnp.concatenate((_flat(node.erase), node.values), -1), precision)
    return Preparation(
        node.transport,
        _flat(node.writes),
        solved[..., :size],
        solved[..., size:],
        _flat(node.read),
        node.attention,
    )


def _kernel(*refs, precision, backward):
    inputs = tuple(ref[...] for ref in refs[:7])
    if backward:
        _, pullback = jax.vjp(lambda xs: prepare_tree(xs, precision), inputs)
        outputs = pullback(Preparation(*(ref[...] for ref in refs[7:13])))[0]
        targets = refs[13:]
    else:
        outputs = prepare_tree(inputs, precision)
        targets = refs[7:]
    for ref, value in zip(targets, outputs, strict=True):
        ref[...] = value


def _call(inputs, precision, interpret, cotangents=None):
    lead = inputs[1].shape[:-3]
    count = math.prod(lead)
    flat = tuple(x.reshape(count, *x.shape[len(lead) :]) for x in inputs)
    sample = tuple(jax.ShapeDtypeStruct(x.shape[1:], x.dtype) for x in flat)
    prepared = jax.eval_shape(lambda xs: prepare_tree(xs, precision), sample)
    backward = cotangents is not None
    if backward:
        operands = (*flat, *(x.reshape(count, *x.shape[len(lead) :]) for x in cotangents))
        output_shapes = sample
    else:
        operands, output_shapes = flat, prepared

    def spec(shape):
        return pl.BlockSpec((None, *shape), lambda index: (index, *((0,) * len(shape))))

    results = pl.pallas_call(
        functools.partial(_kernel, precision=precision, backward=backward),
        out_shape=tuple(jax.ShapeDtypeStruct((count, *x.shape), x.dtype) for x in output_shapes),
        grid=(count,),
        in_specs=[spec(x.shape[1:]) for x in operands],
        out_specs=[spec(x.shape) for x in output_shapes],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="head_stack_tree_backward" if backward else "head_stack_tree_forward",
    )(*operands)
    results = tuple(x.reshape(*lead, *x.shape[1:]) for x in results)
    return results if backward else Preparation(*results)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def fused_prepare(inputs, precision="fp32", interpret=False):
    return _call(inputs, precision, interpret)


def _fwd(inputs, precision, interpret):
    return _call(inputs, precision, interpret), inputs


def _bwd(precision, interpret, inputs, cotangents):
    return (_call(inputs, precision, interpret, cotangents),)


fused_prepare.defvjp(_fwd, _bwd)
