# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Skewon (arXiv 2608.06218): Muon on the Stiefel manifold, for matrices kept at a scaled semi-orthogonal point.

A leaf ``X = c * Xn`` (stacked ``[L, n, p]``, transposed when it is wide) keeps ``Xn^T Xn = I``: all singular
values equal ``c``, which is fixed at init. Per step, with Nesterov momentum ``M``:

- ``N = skew(M Xn^T)`` (``n x n``) and ``B = -msign(N) Xn``: the exact solution of Muon's subproblem
  ``min <M, B>`` over the tangent space with ``||B||_2 <= 1`` (the paper's Proposition 1).
- ``Xn <- polar(Xn + eta B)``, with ``||eta B||_F = lr * ||Xn||_F``: the same relative Frobenius step as MuonH's
  hyperball, so the MuonH learning rate carries over. ``msign`` uses quintic Newton-Schulz; the polar retraction
  uses cubic Newton-Schulz, which converges in a few iterations because the step leaves ``Xn`` nearly orthogonal.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS

from experiments.grug.fast_track.grugmuon_stacked import _intra_rack_axes, _target_named_sharding

MSIGN_STEPS = 5
RETRACTION_ITERS = 4


class StiefelState(NamedTuple):
    momentum: optax.Updates


def _msign(x: jax.Array, steps: int = MSIGN_STEPS) -> jax.Array:
    """Approximate matrix sign (polar factor) of square ``[..., n, n]`` matrices by quintic Newton-Schulz."""
    coeffs = NEWTON_SCHULZ_COEFFICIENTS["quintic"]
    x = x / (jnp.sqrt(jnp.sum(jnp.square(x), axis=(-2, -1), keepdims=True)) + 1e-7)
    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        gram = x @ jnp.swapaxes(x, -1, -2)
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x


def polar_retract(z: jax.Array, iters: int = RETRACTION_ITERS) -> jax.Array:
    """Polar factor of near-orthonormal-column ``[..., n, p]`` matrices by cubic Newton-Schulz."""
    eye = jnp.eye(z.shape[-1], dtype=z.dtype)
    for _ in range(iters):
        z = z @ (1.5 * eye - 0.5 * jnp.swapaxes(z, -1, -2) @ z)
    return z


def _whole_matrices(fn, *values, replicated_args=()):
    """Run a local matrix operation with whole matrices and sharded expert banks."""
    weight = values[0]
    mesh = jax.sharding.get_abstract_mesh()
    target = _target_named_sharding(weight)
    if mesh.empty or target is None or target.mesh.empty:
        return fn(*values, *replicated_args)
    axes = tuple(name for name, _ in _intra_rack_axes(mesh))
    shards = math.prod(mesh.shape[name] for name in axes)
    if weight.ndim == 4 and axes and weight.shape[1] % shards == 0:
        whole = P(None, axes, None, None)
    else:
        whole = P(*tuple(target.spec)[: weight.ndim - 2], None, None)
    values = tuple(reshard(value, whole) for value in values)
    replicated_args = tuple(reshard(jnp.asarray(value), P()) for value in replicated_args)
    result = jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=(whole,) * len(values) + (P(),) * len(replicated_args),
        out_specs=whole,
        check_vma=False,
    )(*values, *replicated_args)
    return reshard(result, target)


def _semiorthogonal_init(weight):
    """QR initialization with the original Frobenius norm for each matrix."""
    value = weight.astype(jnp.float32)
    wide = value.shape[-2] < value.shape[-1]
    if wide:
        value = jnp.swapaxes(value, -1, -2)
    q, r = jnp.linalg.qr(value, mode="reduced")
    signs = jnp.where(jnp.diagonal(r, axis1=-2, axis2=-1) < 0, -1.0, 1.0)
    scale = jnp.sqrt(jnp.sum(value**2, axis=(-2, -1), keepdims=True) / value.shape[-1])
    value = q * signs[..., None, :] * scale
    if wide:
        value = jnp.swapaxes(value, -1, -2)
    return value.astype(weight.dtype)


def semiorthogonal_init(weight):
    return _whole_matrices(_semiorthogonal_init, weight)


def _skew(value):
    return 0.5 * (value - jnp.swapaxes(value, -1, -2))


def skewon_tangent(xn, momentum, *, reduced=True):
    """Algorithms 1/2 of arXiv:2608.06218, including the final skew projection."""
    n, p = xn.shape[-2:]
    if reduced and 2 * p <= n:
        # [-X, M] = QR; R J R^T / 2 is only [2p, 2p].
        q, r = jnp.linalg.qr(jnp.concatenate((-xn, momentum), axis=-1), mode="reduced")
        a, b = r[..., :p], r[..., p:]
        small = _skew(a @ jnp.swapaxes(b, -1, -2))
        y = -_skew(_msign(small))
        return q @ (y @ (jnp.swapaxes(q, -1, -2) @ xn))
    nmat = _skew(momentum @ jnp.swapaxes(xn, -1, -2))
    return -_skew(_msign(nmat)) @ xn


def _stiefel_step(x, direction, learning_rate):
    x32, m32 = x.astype(jnp.float32), direction.astype(jnp.float32)
    wide = x.shape[-2] < x.shape[-1]
    if wide:
        x32, m32 = jnp.swapaxes(x32, -1, -2), jnp.swapaxes(m32, -1, -2)
    p = x32.shape[-1]
    scale = jnp.sqrt(jnp.sum(jnp.square(x32), axis=(-2, -1), keepdims=True) / p)
    xn = x32 / jnp.maximum(scale, 1e-12)
    tangent = skewon_tangent(xn, m32)
    tangent_norm = jnp.sqrt(jnp.sum(jnp.square(tangent), axis=(-2, -1), keepdims=True))
    eta = learning_rate * jnp.sqrt(float(p)) / jnp.maximum(tangent_norm, 1e-12)
    delta = scale * polar_retract(xn + eta * tangent) - x32
    if wide:
        delta = jnp.swapaxes(delta, -1, -2)
    return delta.astype(x.dtype)


@jax.jit
def stiefel_step(x: jax.Array, direction: jax.Array, learning_rate) -> jax.Array:
    """Scaled Skewon update, independently per matrix, restoring parameter sharding."""
    with jax.default_matmul_precision("highest"):
        return _whole_matrices(_stiefel_step, x, direction, replicated_args=(learning_rate,))


def scale_with_stiefel_muon(*, momentum: float, nesterov: bool, learning_rate) -> optax.GradientTransformation:
    """Skewon for every leaf of its group (see the module docstring). Emits the full parameter update, so no
    ``scale(-lr)`` follows it."""

    def init_fn(params):
        return StiefelState(momentum=jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params))

    def update_fn(updates, state, params):
        buffers = jax.tree.map(lambda m, g: momentum * m + g.astype(jnp.float32), state.momentum, updates)
        directions = (
            jax.tree.map(lambda g, m: g.astype(jnp.float32) + momentum * m, updates, buffers) if nesterov else buffers
        )
        new_updates = jax.tree.map(lambda x, d: stiefel_step(x, d, learning_rate), params, directions)
        return new_updates, StiefelState(momentum=buffers)

    return optax.GradientTransformation(init_fn, update_fn)
