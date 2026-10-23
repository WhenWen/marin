# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused four-slot chunk state stream and its explicit reverse stream."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from experiments.grug.moe.head_stack_kda import Preparation, _cross, _intra


def _transport(f, state, transpose=False):
    rank = state.shape[0] // 4
    return jnp.concatenate(
        [sum(f[j if transpose else i, i if transpose else j, :, None] * state[j*rank:(j+1)*rank]
             for j in range(4)) for i in range(4)], axis=0,
    )


def _chunk(p, state, precision):
    innovation = p.values - _cross(p.erase, state, precision)
    out = _cross(p.read, state, precision) + _intra(p.attention, innovation, precision)
    following = _transport(p.transport, state) + _cross(p.writes.T, innovation, precision)
    return out, following, innovation


def _reverse(p, state, innovation, dout, ds, precision):
    dz = _intra(p.attention.T, dout, precision) + _cross(p.writes, ds, precision)
    before = _transport(p.transport, ds, True)
    before += _cross(p.read.T, dout, precision) - _cross(p.erase.T, dz, precision)
    rank = state.shape[0] // 4
    df = jnp.stack([
        jnp.stack([jnp.sum((state[j*rank:(j+1)*rank] * ds[i*rank:(i+1)*rank]).T, axis=0)
                   for j in range(4)]) for i in range(4)
    ])
    return Preparation(
        df, _cross(innovation, ds.T, precision), -_cross(dz, state.T, precision), dz,
        _cross(dout, state.T, precision), _intra(dout, innovation.T, precision),
    ), before


def _forward(*refs, precision):
    *inputs, outr, hr, zr, finalr, scratch = refs
    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = inputs[-1][...]
    state = scratch[...]
    out, following, innovation = _chunk(Preparation(*(x[...] for x in inputs[:-1])), state, precision)
    outr[...], hr[...], zr[...] = out, state, innovation
    scratch[...], finalr[...] = following, following


def _backward(*refs, precision):
    p = Preparation(*(x[...] for x in refs[:6]))
    hr, zr, dor, dsr = refs[6:10]
    *outputs, scratch = refs[10:]
    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = dsr[...]
    grads, before = _reverse(p, hr[...], zr[...], dor[...], scratch[...], precision)
    for ref, grad in zip(outputs[:-1], grads, strict=True):
        ref[...] = grad
    outputs[-1][...], scratch[...] = before, before


def _call(prep, initial, precision, interpret, residual=None):
    groups, chunks = prep.values.shape[:2]
    rows, width = prep.values.shape[-2:]
    size = initial.shape[-2]
    backward = residual is not None

    def chunk_spec(shape):
        return pl.BlockSpec((None, None, *shape),
                            lambda g, n: (g, chunks-1-n if backward else n, *((0,)*len(shape))))

    state_spec = pl.BlockSpec((None, size, width), lambda g, n: (g, 0, 0))
    if backward:
        states, innovation, dout, dfinal = residual
        inputs = (*prep, states, innovation, dout, dfinal)
        shapes = (*[jax.ShapeDtypeStruct(x.shape, x.dtype) for x in prep],
                  jax.ShapeDtypeStruct(initial.shape, initial.dtype))
        specs = [*[chunk_spec(x.shape[2:]) for x in prep], state_spec]
    else:
        inputs = (*prep, initial)
        shapes = tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in
                       ((groups, chunks, rows, width), (groups, chunks, size, width),
                        (groups, chunks, rows, width), initial.shape))
        specs = [chunk_spec((rows, width)), chunk_spec((size, width)), chunk_spec((rows, width)), state_spec]
    return pl.pallas_call(
        functools.partial(_backward if backward else _forward, precision=precision),
        out_shape=shapes, grid=(groups, chunks),
        in_specs=[*[chunk_spec(x.shape[2:]) for x in inputs[:-1]], state_spec], out_specs=specs,
        scratch_shapes=[pltpu.VMEM((size, width), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret, name="head_stack_state_backward" if backward else "head_stack_state_forward",
    )(*inputs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def state_pass(prep, initial, precision="fp32", interpret=False):
    out, _, _, final = _call(prep, initial, precision, interpret)
    return out, final


def _fwd(prep, initial, precision, interpret):
    out, states, innovation, final = _call(prep, initial, precision, interpret)
    return (out, final), (prep, initial, states, innovation)


def _bwd(precision, interpret, saved, cotangents):
    prep, initial, states, innovation = saved
    grads = _call(prep, initial, precision, interpret, (states, innovation, *cotangents))
    return Preparation(*grads[:-1]), grads[-1]


state_pass.defvjp(_fwd, _bwd)


def chunked(inputs, initial, chunk=16, precision="fp32", *, interpret=False, fused_prep=True):
    from experiments.grug.moe.head_stack_kda import _split
    from experiments.grug.moe.head_stack_tree import fused_prepare, prepare_tree

    lead = initial.shape[:-3]
    groups = math.prod(lead)
    prep = (functools.partial(fused_prepare, interpret=interpret) if fused_prep else prepare_tree)(
        _split(inputs, chunk), precision
    )
    prep = jax.tree.map(lambda x: x.reshape(groups, *x.shape[len(lead):]), prep)
    state = initial.reshape(groups, 4 * initial.shape[-2], initial.shape[-1])
    out, final = state_pass(prep, state, precision, interpret)
    return out.reshape(*inputs[1].shape[:-1], initial.shape[-1]), final.reshape(initial.shape)
