# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native four-head KDA with fused soft stack routing at chunk boundaries."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from experiments.grug.moe import kda
from experiments.grug.moe import kda_state_tpu as native

CHUNK = 16


def branches(state):
    zero = jnp.zeros_like(state[:1])
    return jnp.concatenate((zero, state[:-1]), 0), jnp.concatenate((state[1:], zero), 0), state


def mix(state, actions):
    return sum(branch * actions[i][None, None, :] for i, branch in enumerate(branches(state)))


def mix_backward(state, actions, ds):
    push, pop, stay = branches(ds)
    before = actions[0][None, None, :] * pop + actions[1][None, None, :] * push
    before += actions[2][None, None, :] * stay
    # Each action is broadcast over value channels by the wrapper. Its VJP
    # reduces these per-value partials outside the kernel, avoiding scalar IO.
    da = jnp.stack([jnp.sum(branch * ds, axis=(0, 1)) for branch in branches(state)])
    return before, da


def head_forward(q, a, k, w, v, decay, state, mixed):
    if mixed:
        return native._chunk(q, a, k, w, v, decay, state, precision=jax.lax.Precision.HIGH)
    dot = functools.partial(native._dot, precision=jax.lax.Precision.HIGHEST)
    z = v - dot(w, state)
    return dot(q, state) + dot(a, z), decay.T * state + dot(k.T, z), z


def head_backward(q, a, k, w, decay, state, z, dy, ds, mixed):
    if mixed:
        return native._chunk_backward(q, a, k, w, decay, state, z, dy, ds, precision=jax.lax.Precision.HIGH)
    dot = functools.partial(native._dot, precision=jax.lax.Precision.HIGHEST)
    dz = dot(a.T, dy) + dot(k, ds)
    return (
        dot(dy, state.T),
        dot(dy, z.T),
        dot(z, ds.T),
        -dot(dz, state.T),
        dz,
        jnp.sum((state * ds).T, axis=0)[None, :],
        decay.T * ds - dot(w.T, dz) + dot(q.T, dy),
    )


def forward_chunk(q, a, k, w, v, decay, actions, state, *, mixed):
    values = [head_forward(q[h], a[h], k[h], w[h], v[h], decay[h], state[h], mixed) for h in range(4)]
    out, updated, innovation = (jnp.stack([value[i] for value in values]) for i in range(3))
    following = mix(updated, actions)
    return out, state, updated, innovation, following


def backward_chunk(q, a, k, w, _v, decay, actions, state, updated, innovation, dy, ds, *, mixed):
    du, da = mix_backward(updated, actions, ds)
    values = [
        head_backward(q[h], a[h], k[h], w[h], decay[h], state[h], innovation[h], dy[h], du[h], mixed) for h in range(4)
    ]
    grads = [jnp.stack([value[i] for value in values]) for i in range(7)]
    return *grads[:6], da, grads[6]


def _forward(*refs, mixed):
    qr, ar, kr, wr, vr, dr, actr, s0r, outr, hr, ur, zr, finalr, scratch = refs

    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = s0r[...]

    out, state, updated, innovation, following = forward_chunk(
        qr[...], ar[...], kr[...], wr[...], vr[...], dr[...], actr[...], scratch[...], mixed=mixed
    )
    outr[...], hr[...], ur[...], zr[...] = out, state, updated, innovation
    scratch[...], finalr[...] = following, following


def _backward(*refs, mixed):
    *inputs, dsr = refs[:12]
    *outputs, scratch = refs[12:]

    @pl.when(pl.program_id(1) == 0)
    def initialize():
        scratch[...] = dsr[...]

    grads = backward_chunk(*(ref[...] for ref in inputs), scratch[...], mixed=mixed)
    for ref, value in zip(outputs, grads, strict=True):
        ref[...] = value
    scratch[...] = grads[-1]


def _call(prepared, initial, mixed, interpret, residual=None):
    groups, chunks = prepared[0].shape[:2]
    reverse = residual is not None

    def chunk_spec(shape):
        return pl.BlockSpec(
            (None, None, *shape), lambda g, n: (g, chunks - 1 - n if reverse else n, *((0,) * len(shape)))
        )

    state_spec = pl.BlockSpec((None, *initial.shape[1:]), lambda g, n: (g, 0, 0, 0))
    if reverse:
        states, updated, innovation, dy, ds = residual
        inputs = (*prepared, states, updated, innovation, dy, ds)
        shapes = (
            *[jax.ShapeDtypeStruct(x.shape, x.dtype) for x in prepared],
            jax.ShapeDtypeStruct(initial.shape, initial.dtype),
        )
        specs = [*[chunk_spec(x.shape[2:]) for x in prepared], state_spec]
    else:
        inputs = (*prepared, initial)
        dims = (
            prepared[4].shape,
            (groups, chunks, *initial.shape[1:]),
            (groups, chunks, *initial.shape[1:]),
            prepared[4].shape,
            initial.shape,
        )
        shapes = tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in dims)
        specs = [*[chunk_spec(shape[2:]) for shape in dims[:-1]], state_spec]
    return pl.pallas_call(
        functools.partial(_backward if reverse else _forward, mixed=mixed),
        out_shape=shapes,
        grid=(groups, chunks),
        in_specs=[*[chunk_spec(x.shape[2:]) for x in inputs[:-1]], state_spec],
        out_specs=specs,
        scratch_shapes=[pltpu.VMEM(initial.shape[1:], jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        cost_estimate=native._cost(
            functools.partial(backward_chunk if reverse else forward_chunk, mixed=mixed),
            [
                jax.ShapeDtypeStruct(x.shape[2:] if i < len(inputs) - 1 else x.shape[1:], x.dtype)
                for i, x in enumerate(inputs)
            ],
            groups * chunks,
            inputs,
            shapes,
        ),
        interpret=interpret,
        name="boundary_stack_backward" if reverse else "boundary_stack_forward",
    )(*inputs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def state_pass(prepared, initial, mixed=True, interpret=False):
    out, _, _, _, final = _call(prepared, initial, mixed, interpret)
    return out, final


def _fwd(prepared, initial, mixed, interpret):
    out, states, updated, innovation, final = _call(prepared, initial, mixed, interpret)
    return (out, final), (prepared, initial, states, updated, innovation)


def _bwd(mixed, interpret, saved, cotangents):
    prepared, initial, states, updated, innovation = saved
    grads = _call(prepared, initial, mixed, interpret, (states, updated, innovation, *cotangents))
    return tuple(grads[:-1]), grads[-1]


state_pass.defvjp(_fwd, _bwd)


def scan_state(prepared, initial, mixed):
    def sequence(xs, state):
        def step(state, p):
            values = [head_forward(*(x[h] for x in p[:6]), state[h], mixed) for h in range(4)]
            out, updated = (jnp.stack([x[i] for x in values]) for i in range(2))
            return mix(updated, p[6]), out

        final, out = jax.lax.scan(step, state, xs)
        return out, final

    return jax.vmap(sequence)(prepared, initial)


def chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    *,
    actions,
    chunk_size=CHUNK,
    initial_state=None,
    use_qk_l2norm=True,
    matmul_dtype=jnp.bfloat16,
    segment_ids=None,
    interpret=False,
    reference=False,
):
    """Native head-major KDA inputs; actions are [..., T/C, push/pop/stay]."""
    lead, heads, length, rank = q.shape[:-3], *q.shape[-3:]
    if heads != 4 or length % chunk_size:
        raise ValueError("Boundary stack requires four heads and complete chunks")
    mixed = matmul_dtype == jnp.bfloat16
    if matmul_dtype not in (None, jnp.float32, jnp.bfloat16):
        raise ValueError("Boundary stack supports FP32 and BF16/FP32")
    q, k = kda._prepare_qk(q, k, use_qk_l2norm)
    groups, chunks, width = math.prod(lead), length // chunk_size, v.shape[-1]
    if actions.shape != (*lead, chunks, 3):
        raise ValueError("Exactly one push/pop/stay vector is required per chunk")
    qc, kc, vc, gc = (x.astype(jnp.float32).reshape(*lead, 4, chunks, chunk_size, x.shape[-1]) for x in (q, k, v, g))
    bc = beta.astype(jnp.float32).reshape(*lead, 4, chunks, chunk_size)
    masks = None
    if segment_ids is not None:
        if segment_ids.ndim == beta.ndim - 1:
            segment_ids = segment_ids[..., None, :]
        starts = kda.doc_starts(jnp.broadcast_to(segment_ids, beta.shape))
        masks = kda._segment_masks(starts.reshape(*lead, 4, chunks, chunk_size))

    def mm(spec, a, b):
        if mixed:
            return jnp.einsum(spec, a.astype(jnp.bfloat16), b.astype(jnp.bfloat16)).astype(jnp.float32)
        return jnp.einsum(spec, a, b, precision=jax.lax.Precision.HIGHEST)

    with jax.default_matmul_precision("highest"):
        prep = kda._chunk_prep_xla(qc, kc, kc, vc, gc, bc, mm, jnp.bfloat16 if mixed else None, chunk_size, masks)
    raw = (prep.q_inflate, prep.attn, prep.kw, prep.k_cumdecay, prep.v_pseudo, prep.decay[..., None, :])
    packed = tuple(jnp.swapaxes(x, -4, -3).reshape(groups, chunks, 4, *x.shape[-2:]) for x in raw)
    act = jnp.broadcast_to(actions[..., None], (*lead, chunks, 3, width)).reshape(groups, chunks, 3, width)
    if initial_state is None:
        initial_state = jnp.zeros((*lead, 4, rank, width), jnp.float32)
    state = initial_state.reshape(groups, 4, rank, width)
    if reference:
        out, final = scan_state((*packed, act), state, mixed)
    else:
        out, final = state_pass((*packed, act), state, mixed, interpret)
    out = jnp.swapaxes(out.reshape(*lead, chunks, 4, chunk_size, width), -4, -3)
    return out.reshape(*lead, 4, length, width), final.reshape(initial_state.shape)


def native_kernel(q, k, v, g, beta, actions, segment_ids=None, *, chunk=CHUNK, precision="mixed"):
    q, k, v, g = (jnp.swapaxes(x, 1, 2) for x in (q, k, v, g))
    out, _ = chunk_kda(
        q,
        k,
        v,
        g,
        jnp.swapaxes(beta, 1, 2),
        actions=actions,
        chunk_size=chunk,
        segment_ids=segment_ids,
        matmul_dtype=jnp.bfloat16 if precision == "mixed" else jnp.float32,
    )
    return jnp.swapaxes(out, 1, 2)
