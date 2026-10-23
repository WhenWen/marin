# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four native heads as one stack, with no additional recurrent state.

Vector inputs have shape [..., T, 4, channels], shared actions [..., T, 3],
beta [..., T, 4], keep [..., T], and state [..., 4, R, V].
"""

import jax
import jax.numpy as jnp

from experiments.grug.moe.kda import _unit_lower_triangular_inverse
from experiments.grug.moe.stack_kda import Preparation, _move_state

DEPTH = 4
CHUNK = 16
PRECISION = jax.lax.Precision.HIGHEST


def _coefficients(actions):
    push, pop, stay = (actions[..., i] for i in range(3))
    active = jnp.broadcast_to(stay[..., None], (*actions.shape[:-1], DEPTH)).at[..., 0].add(push)
    return push, pop, active


def _transport(state, actions, decay, keep):
    # The key axis is last; state has shape [..., destination, columns, R].
    push, pop, active = _coefficients(actions)
    diagonal = active[..., :, None] * decay
    down = jnp.concatenate((jnp.zeros_like(state[..., :1, :, :]), state[..., :-1, :, :]), axis=-3)
    up = jnp.concatenate((state[..., 1:, :, :], jnp.zeros_like(state[..., :1, :, :])), axis=-3)
    return keep[..., None, None, None] * (
        diagonal[..., :, None, :] * state + push[..., None, None, None] * down + pop[..., None, None, None] * up
    )


def _intra(left, right, precision):
    if precision == "mixed":
        return (left.astype(jnp.bfloat16) @ right.astype(jnp.bfloat16)).astype(jnp.float32)
    return jnp.matmul(left, right, precision=PRECISION)


def _cross(left, right, precision):
    return jnp.matmul(left, right, precision=jax.lax.Precision.HIGH if precision == "mixed" else PRECISION)


def _prepare(inputs, precision):
    query = inputs[1]
    lead, length, heads, rank = query.shape[:-3], *query.shape[-3:]
    size, count = heads * rank, length * heads
    transport = jnp.broadcast_to(jnp.eye(heads, dtype=jnp.float32)[..., None], (*lead, heads, heads, rank))
    writes = jnp.zeros((*lead, heads, count, rank), jnp.float32)
    tokens = tuple(jnp.moveaxis(x, len(lead), 0) for x in inputs)

    def step(carry, token):
        transport, writes = carry
        index, (actions, query, key, value, decay, beta, keep) = token
        _, _, active = _coefficients(actions)
        erase_key = key * decay * (active * beta * keep[..., None])[..., None]
        erase = (erase_key[..., :, None, :] * transport).reshape(*lead, heads, size)
        correction = jnp.sum(erase_key[..., :, None, :] * writes, axis=-1)
        transport = _transport(transport, actions, decay, keep)
        writes = _transport(writes, actions, decay, keep)
        writes = writes.at[..., jnp.arange(heads), index * heads + jnp.arange(heads), :].add(key)
        read = (query[..., :, None, :] * transport).reshape(*lead, heads, size)
        attention = jnp.sum(query[..., :, None, :] * writes, axis=-1)
        new_value = value * (beta * active)[..., None]
        return (transport, writes), (erase, correction, read, attention, new_value)

    (transport, writes), rows = jax.lax.scan(step, (transport, writes), (jnp.arange(length), tokens))
    erase, correction, read, attention, values = (
        jnp.moveaxis(row, 0, len(lead)).reshape(*lead, count, row.shape[-1]) for row in rows
    )
    with jax.default_matmul_precision("highest"):
        inverse = _unit_lower_triangular_inverse(
            -correction, count, matmul_dtype=jnp.bfloat16 if precision == "mixed" else None
        )
    solved = _intra(inverse, jnp.concatenate((erase, values), axis=-1), precision)
    return Preparation(
        transport,
        jnp.swapaxes(writes, -3, -2).reshape(*lead, count, size),
        solved[..., :size],
        solved[..., size:],
        read,
        attention,
    )


def _advance(prep, state, precision):
    flat = state.reshape(*state.shape[:-3], -1, state.shape[-1])
    innovation = prep.values - _cross(prep.erase, flat, precision)
    updated = _move_state(prep.transport, state)
    updated += _cross(jnp.swapaxes(prep.writes, -1, -2), innovation, precision).reshape(state.shape)
    return updated, innovation


def _emit(prep, state, innovation, precision):
    flat = state.reshape(*state.shape[:-3], -1, state.shape[-1])
    return _cross(prep.read, flat, precision) + _intra(prep.attention, innovation, precision)


def reference(inputs, initial):
    """Literal three-action recurrence used as the independent numerical oracle."""
    axis = initial.ndim - 3
    tokens = tuple(jnp.moveaxis(x, axis, 0) for x in inputs)

    def step(state, token):
        actions, query, key, value, decay, beta, keep = token
        state = keep[..., None, None, None] * state
        decayed = decay[..., :, :, None] * state
        prediction = jnp.einsum("...hr,...hrv->...hv", key, decayed, precision=PRECISION)
        updated = decayed + beta[..., :, None, None] * key[..., :, :, None] * (value - prediction)[..., :, None, :]
        push = jnp.concatenate((updated[..., :1, :, :], state[..., :-1, :, :]), axis=-3)
        pop = jnp.concatenate((state[..., 1:, :, :], jnp.zeros_like(state[..., :1, :, :])), axis=-3)
        branches = jnp.stack((push, pop, updated), axis=-4)
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


def _forward(inputs, initial, chunk, precision):
    prep = jax.checkpoint(lambda x: _prepare(x, precision))(_split(inputs, chunk))
    axis = initial.ndim - 3
    blocks = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0), prep)

    def step(state, block):
        following, _ = _advance(block, state, precision)
        return following, state

    final, incoming = jax.lax.scan(step, initial, blocks)
    incoming = jnp.moveaxis(incoming, 0, axis)
    _, innovation = _advance(prep, incoming, precision)
    output = _emit(prep, incoming, innovation, precision)
    return (output.reshape(*inputs[1].shape[:-1], initial.shape[-1]), final), incoming


def chunked(inputs, initial, chunk=CHUNK, precision="fp32"):
    """Read four slots; differentiate the executed graph, including BF16 casts.

    Both precision modes use the same rematerialized preparation and automatic
    reverse pass. FP32 remains the strict equivalence reference.
    """
    if precision not in ("fp32", "mixed"):
        raise ValueError("Head-stack precision must be fp32 or mixed")
    return _forward(inputs, initial, chunk, precision)[0]


def native_kernel(q, k, v, log_decay, beta, actions, segment_ids=None, *, chunk=CHUNK, precision="mixed"):
    """Normalize native [B,T,H,R] projections and read the four existing slots."""
    q, k, v, log_decay, actions = (x.astype(jnp.float32) for x in (q, k, v, log_decay, actions))
    q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6) / q.shape[-1] ** 0.5
    k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)
    keep = jnp.ones(q.shape[:2], jnp.float32)
    if segment_ids is not None:
        starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
        keep *= (~starts).astype(jnp.float32)
    initial = jnp.zeros((q.shape[0], DEPTH, q.shape[-1], v.shape[-1]), jnp.float32)
    output, _ = chunked(
        (actions, q, k, v, jnp.exp(log_decay), beta.astype(jnp.float32), keep), initial, chunk, precision
    )
    return output
