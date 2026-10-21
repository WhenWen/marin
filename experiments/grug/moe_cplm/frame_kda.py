# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chunk-parallel KDA memories with protected parent frames.

The token operator is sparse in depth and diagonal-plus-rank-one in the top
frame. Only chunk boundaries use dense affine composition. Controllers depend
on the input stream, never on the recurrent state. All recurrence math is FP32.
"""

from functools import partial

import jax
import jax.numpy as jnp

PRECISION = jax.lax.Precision.HIGHEST


def transition(state, actions, key, decay, beta, keep):
    """Apply the homogeneous operator to [..., depth, rank, columns]."""
    state = state * keep[..., None, None, None]
    push, pop, stay, replace = (actions[..., i] for i in range(4))
    down = jnp.concatenate((state[..., :1, :, :], state[..., :-1, :, :]), axis=-3)
    up = jnp.concatenate((state[..., 1:, :, :], jnp.zeros_like(state[..., :1, :, :])), axis=-3)
    result = (
        push[..., None, None, None] * down
        + pop[..., None, None, None] * up
        + (stay + replace)[..., None, None, None] * state
    )
    top = state[..., 0, :, :] * decay[..., :, None]
    prediction = jnp.einsum("...r,...rv->...v", key, top, precision=PRECISION)
    top = top - beta[..., None, None] * key[..., :, None] * prediction[..., None, :]
    # Pop restores the parent without applying the child's decay or delta rule.
    parent = up[..., 0, :, :]
    top = (push + stay)[..., None, None] * top + pop[..., None, None] * parent
    return result.at[..., 0, :, :].set(top)


def transition_transpose(cotangent, actions, key, decay, beta, keep):
    """Apply the transpose without constructing a dense token matrix."""
    push, pop, stay, replace = (actions[..., i] for i in range(4))
    top = cotangent[..., 0, :, :]
    result = (stay + replace)[..., None, None, None] * cotangent
    result = result.at[..., 0, :, :].set(0)
    result = result.at[..., :-1, :, :].add(push[..., None, None, None] * cotangent[..., 1:, :, :])
    result = result.at[..., 1:, :, :].add(pop[..., None, None, None] * cotangent[..., :-1, :, :])
    projection = jnp.einsum("...r,...rv->...v", key, top, precision=PRECISION)
    top = (top - beta[..., None, None] * key[..., :, None] * projection[..., None, :]) * decay[..., :, None]
    result = result.at[..., 0, :, :].add((push + stay)[..., None, None] * top)
    return result * keep[..., None, None, None]


def step(state, inputs):
    """Read after the soft push/pop/stay/replace and top-frame delta update."""
    actions, query, key, value, decay, beta, address, keep = inputs
    state = transition(state, actions, key, decay, beta, keep)
    write = (actions[..., 0] + actions[..., 2] + actions[..., 3]) * beta
    state = state.at[..., 0, :, :].add(write[..., None, None] * key[..., :, None] * value[..., None, :])
    output = jnp.einsum("...d,...r,...drv->...v", address, query, state, precision=PRECISION)
    return state, output


def recurrent(inputs, initial):
    """Literal token recurrence; inputs have time as their first dimension."""
    return jax.lax.scan(step, initial, inputs)


def _compose(earlier, later):
    a1, b1 = earlier
    a2, b2 = later
    return (
        jnp.einsum("...ij,...jk->...ik", a2, a1, precision=PRECISION),
        jnp.einsum("...ij,...jv->...iv", a2, b1, precision=PRECISION) + b2,
    )


def _chunks(inputs, chunk):
    # [B,T,H,...] -> [C,B,n,H,...], so lax.scan runs C steps over all chunks.
    batch, length = inputs[0].shape[:2]
    if length % chunk:
        raise ValueError("The sequence length must be a multiple of frame_kda_chunk")
    return tuple(jnp.moveaxis(x.reshape(batch, length // chunk, chunk, *x.shape[2:]), 2, 0) for x in inputs)


def _summaries(xs, initial):
    batch, heads, depth, rank, width = initial.shape
    chunks = xs[0].shape[2]
    size = depth * rank
    eye = jnp.broadcast_to(jnp.eye(size, dtype=jnp.float32), (batch, chunks, heads, size, size))
    zero = jnp.zeros((batch, chunks, heads, size, width), jnp.float32)
    augmented = jnp.concatenate((eye, zero), axis=-1).reshape(batch, chunks, heads, depth, rank, size + width)

    def summarize(state, token):
        actions, _, key, value, decay, beta, _, keep = token
        state = transition(state, actions, key, decay, beta, keep)
        write = (actions[..., 0] + actions[..., 2] + actions[..., 3]) * beta
        state = state.at[..., 0, :, size:].add(
            write[..., None, None] * key[..., :, None] * value[..., None, :]
        )
        return state, None

    augmented, _ = jax.lax.scan(summarize, augmented, xs)
    augmented = augmented.reshape(batch, chunks, heads, size, size + width)
    return augmented[..., :size], augmented[..., size:]


def _forward(inputs, initial, chunk):
    xs = _chunks(inputs, chunk)
    matrices, offsets = _summaries(xs, initial)
    prefix_a, prefix_b = jax.lax.associative_scan(_compose, (matrices, offsets), axis=1)
    batch, heads, depth, rank, width = initial.shape
    init = initial.reshape(batch, heads, depth * rank, width)
    ends = jnp.einsum("bnhij,bhjv->bnhiv", prefix_a, init, precision=PRECISION) + prefix_b
    incoming = jnp.concatenate((init[:, None], ends[:, :-1]), axis=1)
    incoming = incoming.reshape(batch, ends.shape[1], heads, depth, rank, width)
    _, output = recurrent(xs, incoming)
    output = jnp.moveaxis(output, 0, 2).reshape(batch, -1, heads, width)
    final = ends[:, -1].reshape(initial.shape)
    return (output, final), (xs, incoming, matrices)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def chunked(inputs, initial, chunk):
    """Return outputs and final state; scan only chunk summaries globally.

    inputs are (actions, query, key, value, decay, beta, depth_address, keep),
    shaped [B,T,H,...]. initial is [B,H,D,R,V]. keep is float, zero at resets.
    The VJP scans boundary adjoints then differentiates independent local replays;
    it never differentiates the dense summary-building recurrence.
    """
    return _forward(inputs, initial, chunk)[0]


def _fwd(inputs, initial, chunk):
    return _forward(inputs, initial, chunk)


def _bwd(chunk, saved, cotangents):
    xs, incoming, matrices = saved
    doutput, dfinal = cotangents
    batch, blocks, heads, depth, rank, width = incoming.shape
    dy = jnp.moveaxis(doutput.reshape(batch, blocks, chunk, heads, width), 2, 0)

    def adjoint(carry, token):
        values, dy_token = token
        actions, query, key, _, decay, beta, address, keep = values
        carry = carry + jnp.einsum("...d,...r,...v->...drv", address, query, dy_token, precision=PRECISION)
        carry = transition_transpose(carry, actions, key, decay, beta, keep)
        return carry, None

    sources, _ = jax.lax.scan(adjoint, jnp.zeros_like(incoming), (xs, dy), reverse=True)
    sources = sources.reshape(batch, blocks, heads, depth * rank, width)
    transposed = jnp.swapaxes(matrices, -1, -2)
    # Reverse chronological order for ordinary affine composition.
    rev_a, rev_b = jax.lax.associative_scan(_compose, (transposed[:, ::-1], sources[:, ::-1]), axis=1)
    terminal = dfinal.reshape(batch, heads, depth * rank, width)
    starts = jnp.einsum("bnhij,bhjv->bnhiv", rev_a, terminal, precision=PRECISION) + rev_b
    starts = starts[:, ::-1]
    outgoing = jnp.concatenate((starts[:, 1:], terminal[:, None]), axis=1).reshape(incoming.shape)
    # Rematerialize only each local replay. The long sequence has no serial VJP.
    _, pullback = jax.vjp(jax.checkpoint(recurrent), xs, incoming)
    dxs, _ = pullback((outgoing, dy))
    dinputs = tuple(jnp.moveaxis(x, 0, 2).reshape(batch, blocks * chunk, *x.shape[3:]) for x in dxs)
    dinitial = starts[:, 0].reshape(batch, heads, depth, rank, width)
    return dinputs, dinitial


chunked.defvjp(_fwd, _bwd)
