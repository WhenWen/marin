# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ORCA's transient orthogonality correction, composed with Belay's MuonH."""

import math

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P

from experiments.grug.fast_track.grugmuon_stacked import _intra_rack_axes, _target_named_sharding


def orthogonal_gradient(weight: jax.Array) -> jax.Array:
    """Gradient of one quarter squared Gram error, independently per matrix."""
    value = weight.astype(jnp.float32)
    transpose = jnp.swapaxes(value, -1, -2)
    if weight.shape[-2] <= weight.shape[-1]:
        gram = jnp.matmul(value, transpose, precision=jax.lax.Precision.HIGHEST)
        gradient = jnp.matmul(gram, value, precision=jax.lax.Precision.HIGHEST) - value
    else:
        gram = jnp.matmul(transpose, value, precision=jax.lax.Precision.HIGHEST)
        gradient = jnp.matmul(value, gram, precision=jax.lax.Precision.HIGHEST) - value
    return gradient.astype(weight.dtype)


def sharded_orthogonal_gradient(weight: jax.Array) -> jax.Array:
    """Distribute whole expert matrices, then restore the parameter layout."""
    mesh = jax.sharding.get_abstract_mesh()
    target = _target_named_sharding(weight)
    if mesh.empty or target is None:
        return orthogonal_gradient(weight)
    axes = tuple(name for name, _ in _intra_rack_axes(mesh))
    shards = math.prod(mesh.shape[name] for name in axes)
    if weight.ndim == 4 and axes and weight.shape[1] % shards == 0:
        # Exchange the FSDP matrix shards for independent expert shards. No
        # device materializes the full expert bank.
        whole = P(None, axes, None, None)
    else:
        whole = P(*tuple(target.spec)[: weight.ndim - 2], None, None)
    distributed = jax.sharding.reshard(weight, whole)
    result = jax.shard_map(orthogonal_gradient, mesh=mesh, in_specs=whole, out_specs=whole, check_vma=False)(distributed)
    return jax.sharding.reshard(result, target)


def add_early_orthogonal_correction(learning_rate, strength: float, release_step: int):
    """Add -lr * strength * G_orth(W_t) to an already scaled update until release.

    The correction is outside MuonH's retraction and uses pre-update weights,
    as in arXiv:2610.06116. Weight norms may change during the active phase.
    The underlying momentum state and all non-matrix parameter groups are untouched.
    """

    def init_fn(params):
        del params
        return jnp.zeros([], jnp.int32)

    def update_fn(updates, count, params=None):
        if params is None:
            raise ValueError("ORCA requires pre-update parameters")

        def active(deltas):
            return jax.tree.map(
                lambda delta, weight: delta - learning_rate * strength * sharded_orthogonal_gradient(weight),
                deltas,
                params,
            )

        updates = jax.lax.cond(count < release_step, active, lambda deltas: deltas, updates)
        return updates, optax.safe_int32_increment(count)

    return optax.GradientTransformation(init_fn, update_fn)
