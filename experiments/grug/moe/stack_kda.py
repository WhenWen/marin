# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Depth-four protected KDA with structured chunk preparation and state passes.

Inputs use [..., time, channels]; states use [..., key, depth, value]. The
custom backward retains chunk boundaries and rematerializes one chunk at a
time, avoiding full-sequence token-state or preparation tapes.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from experiments.grug.moe.kda import _unit_lower_triangular_inverse

DEPTH = 4
CHUNK = 32
PRECISION = jax.lax.Precision.HIGHEST


class Preparation(NamedTuple):
    transport: jax.Array
    writes: jax.Array
    erase: jax.Array
    values: jax.Array
    read: jax.Array
    attention: jax.Array


def _transport(state, actions, decay, keep):
    push, pop, stay, replace = (actions[..., i] for i in range(4))
    state = state * keep[..., None, None, None]
    down = jnp.concatenate((jnp.zeros_like(state[..., :1, :]), state[..., :-1, :]), axis=-2)
    up = jnp.concatenate((state[..., 1:, :], jnp.zeros_like(state[..., :1, :])), axis=-2)
    result = (
        push[..., None, None, None] * down
        + pop[..., None, None, None] * up
        + (stay + replace)[..., None, None, None] * state
    )
    top = (push + stay)[..., None, None] * decay[..., :, None] * state[..., 0, :]
    top += pop[..., None, None] * up[..., 0, :]
    return result.at[..., 0, :].set(top)


def _prepare(inputs):
    actions, query, key, value, decay, beta, keep = inputs
    lead, length, rank = query.shape[:-2], query.shape[-2], query.shape[-1]
    size = rank * DEPTH
    transport = jnp.broadcast_to(jnp.eye(DEPTH, dtype=jnp.float32), (*lead, rank, DEPTH, DEPTH))
    writes = jnp.zeros((*lead, rank, DEPTH, length), jnp.float32)
    tokens = tuple(jnp.moveaxis(x, len(lead), 0) for x in inputs)

    def step(carry, token):
        transport, writes = carry
        index, (actions, query, key, value, decay, beta, keep) = token
        active = actions[..., 0] + actions[..., 2]
        erase_key = key * decay * (active * beta * keep)[..., None]
        erase = (erase_key[..., :, None] * transport[..., 0, :]).reshape(*lead, size)
        correction = jnp.einsum("...r,...ri->...i", erase_key, writes[..., 0, :], precision=PRECISION)
        transport = _transport(transport, actions, decay, keep)
        writes = _transport(writes, actions, decay, keep)
        writes = writes.at[..., :, 0, index].add(key)
        read = (query[..., :, None] * transport[..., 0, :]).reshape(*lead, size)
        attention = jnp.einsum("...r,...ri->...i", query, writes[..., 0, :], precision=PRECISION)
        new_value = value * (beta * (active + actions[..., 3]))[..., None]
        return (transport, writes), (erase, correction, read, attention, new_value)

    (transport, writes), rows = jax.lax.scan(step, (transport, writes), (jnp.arange(length), tokens))
    erase, correction, read, attention, values = (jnp.moveaxis(row, 0, -2) for row in rows)
    inverse = _unit_lower_triangular_inverse(-correction, length)
    solved = jnp.matmul(inverse, jnp.concatenate((erase, values), axis=-1), precision=PRECISION)
    return Preparation(
        transport,
        writes.reshape(*lead, size, length),
        solved[..., :size],
        solved[..., size:],
        read,
        attention,
    )


def _advance(prep, state):
    flat = state.reshape(*state.shape[:-3], -1, state.shape[-1])
    innovation = prep.values - jnp.matmul(prep.erase, flat, precision=PRECISION)
    moved = jnp.einsum("...rij,...rjv->...riv", prep.transport, state, precision=PRECISION)
    updated = moved + jnp.matmul(prep.writes, innovation, precision=PRECISION).reshape(state.shape)
    return updated, innovation


def _emit(prep, state, innovation):
    flat = state.reshape(*state.shape[:-3], -1, state.shape[-1])
    return jnp.matmul(prep.read, flat, precision=PRECISION) + jnp.matmul(
        prep.attention, innovation, precision=PRECISION
    )


def _single_chunk(inputs, initial):
    prep = _prepare(inputs)
    final, innovation = _advance(prep, initial)
    return _emit(prep, initial, innovation), final


def _split(inputs, chunk):
    lead, length = inputs[1].shape[:-2], inputs[1].shape[-2]
    if length % chunk:
        raise ValueError("Stack KDA sequence length must divide into complete chunks")
    return tuple(x.reshape(*lead, length // chunk, chunk, *x.shape[len(lead) + 1 :]) for x in inputs)


def _forward(inputs, initial, chunk):
    xs = _split(inputs, chunk)
    prep = _prepare(xs)
    axis = initial.ndim - 3
    blocks = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0), prep)

    def step(state, block):
        final, _ = _advance(block, state)
        return final, state

    final, incoming = jax.lax.scan(step, initial, blocks)
    incoming = jnp.moveaxis(incoming, 0, axis)
    _, innovation = _advance(prep, incoming)
    output = _emit(prep, incoming, innovation)
    output = output.reshape(*initial.shape[:-3], inputs[1].shape[-2], initial.shape[-1])
    return (output, final), incoming


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def chunked(inputs, initial, chunk=CHUNK):
    """Return top-frame reads and final state for pre-normalized controls.

    Args:
        inputs: Action probabilities, query, key, value, decay, beta and keep.
            Each has [..., time, features], except scalar beta/keep [..., time].
        initial: FP32 state [..., key, 4, value].
        chunk: Number of tokens per local block.
    """
    return _forward(inputs, initial, chunk)[0]


def _fwd(inputs, initial, chunk):
    result, incoming = _forward(inputs, initial, chunk)
    return result, (inputs, incoming)


def _bwd(chunk, saved, cotangents):
    inputs, incoming = saved
    output_grad, final_grad = cotangents
    xs = _split(inputs, chunk)
    prep = _prepare(xs)
    axis = incoming.ndim - 4
    dy = output_grad.reshape(*incoming.shape[:-3], chunk, incoming.shape[-1])
    write_grad = jnp.matmul(jnp.swapaxes(prep.attention, -1, -2), dy, precision=PRECISION)
    source = jnp.matmul(jnp.swapaxes(prep.read, -1, -2), dy, precision=PRECISION)
    source -= jnp.matmul(jnp.swapaxes(prep.erase, -1, -2), write_grad, precision=PRECISION)
    source = source.reshape(incoming.shape)
    blocks = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0), (prep, source))

    def adjoint(carry, block):
        prep, source = block
        flat = carry.reshape(*carry.shape[:-3], -1, carry.shape[-1])
        propagated = jnp.einsum("...rji,...rjv->...riv", prep.transport, carry, precision=PRECISION)
        correction = jnp.matmul(jnp.swapaxes(prep.writes, -1, -2), flat, precision=PRECISION)
        correction = jnp.matmul(jnp.swapaxes(prep.erase, -1, -2), correction, precision=PRECISION)
        return propagated - correction.reshape(carry.shape) + source, carry

    initial_grad, outgoing = jax.lax.scan(adjoint, final_grad, blocks, reverse=True)
    outgoing = jnp.moveaxis(outgoing, 0, axis)
    leading = incoming.shape[:-3]

    def flatten(x):
        return x.reshape(-1, *x.shape[len(leading) :])

    # A bounded local VJP keeps the prepare/replay tape independent of T.
    def local(values):
        tokens, state, dy, terminal = values
        _, pullback = jax.vjp(_single_chunk, tokens, state)
        return pullback((dy, terminal))[0]

    local_inputs = jax.tree.map(flatten, (xs, incoming, dy, outgoing))
    gradients = jax.lax.map(local, local_inputs)
    gradients = tuple(x.reshape(original.shape) for x, original in zip(gradients, inputs, strict=True))
    return gradients, initial_grad


chunked.defvjp(_fwd, _bwd)


def native_kernel(q, k, v, log_decay, beta, actions, segment_ids=None, *, chunk=CHUNK):
    """Normalize native [B,T,H,R] inputs and return [B,T,H,V] reads."""
    q, k, v, log_decay, actions = (jnp.swapaxes(x.astype(jnp.float32), 1, 2) for x in (q, k, v, log_decay, actions))
    q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6) / q.shape[-1] ** 0.5
    k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)
    beta = jnp.swapaxes(beta.astype(jnp.float32), 1, 2)
    keep = jnp.ones(beta.shape, jnp.float32)
    if segment_ids is not None:
        starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
        keep = keep * (~starts[:, None, :]).astype(jnp.float32)
    inputs = (actions, q, k, v, jnp.exp(log_decay), beta, keep)
    initial = jnp.zeros((*q.shape[:-2], q.shape[-1], DEPTH, v.shape[-1]), jnp.float32)
    output, _ = chunked(inputs, initial, chunk)
    return jnp.swapaxes(output, 1, 2)
