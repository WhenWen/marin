# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four native heads as one stack, with no additional recurrent state.

Vector inputs have shape [..., T, 4, channels], shared actions [..., T, 4],
beta [..., T, 4], keep [..., T], and state [..., 4, R, V].
"""

from functools import partial

import jax
import jax.numpy as jnp

from experiments.grug.moe.kda import _unit_lower_triangular_inverse
from experiments.grug.moe.stack_kda import Preparation, _advance, _emit, _move_state

DEPTH = 4
CHUNK = 16
BACKWARD_BATCH = 16
PRECISION = jax.lax.Precision.HIGHEST


def _coefficients(actions):
    push, pop, stay, replace = (actions[..., i] for i in range(DEPTH))
    active = jnp.broadcast_to(stay[..., None], actions.shape).at[..., 0].add(push)
    write = active.at[..., 0].add(replace)
    return push, pop, replace, active, write


def _transport(state, actions, decay, keep):
    # The key axis is last; state has shape [..., destination, columns, R].
    push, pop, replace, active, _ = _coefficients(actions)
    diagonal = active[..., :, None] * decay
    diagonal += replace[..., None, None] * (jnp.arange(DEPTH) > 0)[:, None]
    down = jnp.concatenate((jnp.zeros_like(state[..., :1, :, :]), state[..., :-1, :, :]), axis=-3)
    up = jnp.concatenate((state[..., 1:, :, :], jnp.zeros_like(state[..., :1, :, :])), axis=-3)
    return keep[..., None, None, None] * (
        diagonal[..., :, None, :] * state + push[..., None, None, None] * down + pop[..., None, None, None] * up
    )


def _prepare(inputs):
    query = inputs[1]
    lead, length, heads, rank = query.shape[:-3], *query.shape[-3:]
    size, count = heads * rank, length * heads
    transport = jnp.broadcast_to(jnp.eye(heads, dtype=jnp.float32)[..., None], (*lead, heads, heads, rank))
    writes = jnp.zeros((*lead, heads, count, rank), jnp.float32)
    tokens = tuple(jnp.moveaxis(x, len(lead), 0) for x in inputs)

    def step(carry, token):
        transport, writes = carry
        index, (actions, query, key, value, decay, beta, keep) = token
        _, _, _, active, write = _coefficients(actions)
        erase_key = key * decay * (active * beta * keep[..., None])[..., None]
        erase = (erase_key[..., :, None, :] * transport).reshape(*lead, heads, size)
        correction = jnp.sum(erase_key[..., :, None, :] * writes, axis=-1)
        transport = _transport(transport, actions, decay, keep)
        writes = _transport(writes, actions, decay, keep)
        writes = writes.at[..., jnp.arange(heads), index * heads + jnp.arange(heads), :].add(key)
        read = (query[..., :, None, :] * transport).reshape(*lead, heads, size)
        attention = jnp.sum(query[..., :, None, :] * writes, axis=-1)
        new_value = value * (beta * write)[..., None]
        return (transport, writes), (erase, correction, read, attention, new_value)

    (transport, writes), rows = jax.lax.scan(step, (transport, writes), (jnp.arange(length), tokens))
    erase, correction, read, attention, values = (
        jnp.moveaxis(row, 0, len(lead)).reshape(*lead, count, row.shape[-1]) for row in rows
    )
    with jax.default_matmul_precision("highest"):
        inverse = _unit_lower_triangular_inverse(-correction, count)
    solved = jnp.matmul(inverse, jnp.concatenate((erase, values), axis=-1), precision=PRECISION)
    return Preparation(
        transport,
        jnp.swapaxes(writes, -3, -2).reshape(*lead, count, size),
        solved[..., :size],
        solved[..., size:],
        read,
        attention,
    )


def reference(inputs, initial):
    """Literal four-branch recurrence used as the independent numerical oracle."""
    axis = initial.ndim - 3
    tokens = tuple(jnp.moveaxis(x, axis, 0) for x in inputs)

    def step(state, token):
        actions, query, key, value, decay, beta, keep = token
        state = keep[..., None, None, None] * state
        decayed = decay[..., :, :, None] * state
        prediction = jnp.einsum("...hr,...hrv->...hv", key, decayed, precision=PRECISION)
        write = beta[..., :, None, None] * key[..., :, :, None] * value[..., :, None, :]
        updated = decayed + beta[..., :, None, None] * key[..., :, :, None] * (value - prediction)[..., :, None, :]
        push = jnp.concatenate((updated[..., :1, :, :], state[..., :-1, :, :]), axis=-3)
        pop = jnp.concatenate((state[..., 1:, :, :], jnp.zeros_like(state[..., :1, :, :])), axis=-3)
        replace = jnp.concatenate((write[..., :1, :, :], state[..., 1:, :, :]), axis=-3)
        branches = jnp.stack((push, pop, updated, replace), axis=-4)
        state = jnp.sum(branches * actions[..., :, None, None, None], axis=-4)
        output = jnp.einsum("...hr,...hrv->...hv", query, state, precision=PRECISION)
        return state, output

    final, output = jax.lax.scan(step, initial, tokens)
    return jnp.moveaxis(output, 0, axis), final


def _split(inputs, chunk):
    lead, length = inputs[1].shape[:-3], inputs[1].shape[-3]
    if length % chunk:
        raise ValueError("Head-stack sequence length must divide into complete chunks")
    return tuple(x.reshape(*lead, length // chunk, chunk, *x.shape[len(lead) + 1 :]) for x in inputs)


def _forward(inputs, initial, chunk):
    prep = _prepare(_split(inputs, chunk))
    axis = initial.ndim - 3
    blocks = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0), prep)

    def step(state, block):
        following, _ = _advance(block, state)
        return following, state

    final, incoming = jax.lax.scan(step, initial, blocks)
    incoming = jnp.moveaxis(incoming, 0, axis)
    _, innovation = _advance(prep, incoming)
    output = _emit(prep, incoming, innovation)
    return (output.reshape(*inputs[1].shape[:-1], initial.shape[-1]), final), incoming


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def chunked(inputs, initial, chunk=CHUNK):
    """Compute all four slot reads and the final state with chunk boundaries."""
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
    dy = output_grad.reshape(*incoming.shape[:-3], chunk * DEPTH, incoming.shape[-1])
    write_grad = jnp.matmul(jnp.swapaxes(prep.attention, -1, -2), dy, precision=PRECISION)
    source = jnp.matmul(jnp.swapaxes(prep.read, -1, -2), dy, precision=PRECISION)
    source -= jnp.matmul(jnp.swapaxes(prep.erase, -1, -2), write_grad, precision=PRECISION)
    source = source.reshape(incoming.shape)
    blocks = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0), (prep, source))

    def adjoint(carry, block):
        prep, source = block
        flat = carry.reshape(*carry.shape[:-3], -1, carry.shape[-1])
        propagated = _move_state(jnp.swapaxes(prep.transport, -3, -2), carry)
        correction = jnp.matmul(prep.writes, flat, precision=PRECISION)
        correction = jnp.matmul(jnp.swapaxes(prep.erase, -1, -2), correction, precision=PRECISION)
        return propagated - correction.reshape(carry.shape) + source, carry

    initial_grad, outgoing = jax.lax.scan(adjoint, final_grad, blocks, reverse=True)
    outgoing = jnp.moveaxis(outgoing, 0, axis)
    leading = incoming.shape[:-3]

    def flatten(x):
        return x.reshape(-1, *x.shape[len(leading) :])

    def local(values):
        tokens, state, dy, terminal = values
        _, pullback = jax.vjp(reference, tokens, state)
        return pullback((dy.reshape(chunk, DEPTH, state.shape[-1]), terminal))[0]

    local_inputs = jax.tree.map(flatten, (xs, incoming, dy, outgoing))
    gradients = jax.lax.map(local, local_inputs, batch_size=BACKWARD_BATCH)
    gradients = tuple(x.reshape(original.shape) for x, original in zip(gradients, inputs, strict=True))
    return gradients, initial_grad


chunked.defvjp(_fwd, _bwd)


def native_kernel(q, k, v, log_decay, beta, actions, segment_ids=None, *, chunk=CHUNK):
    """Normalize native [B,T,H,R] projections and read the four existing slots."""
    q, k, v, log_decay, actions = (x.astype(jnp.float32) for x in (q, k, v, log_decay, actions))
    q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6) / q.shape[-1] ** 0.5
    k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)
    keep = jnp.ones(q.shape[:2], jnp.float32)
    if segment_ids is not None:
        starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
        keep *= (~starts).astype(jnp.float32)
    initial = jnp.zeros((q.shape[0], DEPTH, q.shape[-1], v.shape[-1]), jnp.float32)
    output, _ = chunked((actions, q, k, v, jnp.exp(log_decay), beta.astype(jnp.float32), keep), initial, chunk)
    return output
