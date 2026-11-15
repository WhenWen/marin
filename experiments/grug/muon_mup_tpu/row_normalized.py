# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neuron-wise normalization of Muon directions or MuonH weight constraints."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from levanter.optim.util import CoefficientType

from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon
from experiments.grug.fast_track.optimizer import (
    _match_named_sharding_to_params,
    _pin_sharding,
    _scale_invariant_hyperball_updates,
)


class RowMomentState(NamedTuple):
    second_moment: object


def normalize_neuron_updates(beta2: float, eps: float = 1e-10) -> optax.GradientTransformation:
    """Normalize output-neuron updates, preserving each logical matrix's norm.

    KMA stores matrices as [..., input, output], transposed from the NorMuon
    paper. Reduce axis -2, independently for every layer and expert. The input
    already carries Muon's shape scale; remove it from the moment statistic.
    beta2=0 gives instantaneous row RMS normalization without temporal averaging.
    """

    def init_fn(params):
        moments = jax.tree.map(lambda p: jnp.zeros_like(jnp.mean(p.astype(jnp.float32), axis=-2, keepdims=True)), params)
        return RowMomentState(moments)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("Neuron normalization requires parameter sharding")
        updates = _match_named_sharding_to_params(updates, params)

        def moment(u, v):
            shape_scale_squared = max(1.0, u.shape[-1] / u.shape[-2])
            statistic = jnp.mean(jnp.square(u.astype(jnp.float32)), axis=-2, keepdims=True) / shape_scale_squared
            return beta2 * v + (1 - beta2) * statistic

        moments = jax.tree.map(moment, updates, state.second_moment)

        def normalize(u, v):
            original = u.astype(jnp.float32)
            normalized = original / (jnp.sqrt(v) + eps)
            norm = jnp.sqrt(jnp.sum(jnp.square(original), axis=(-2, -1), keepdims=True))
            new_norm = jnp.sqrt(jnp.sum(jnp.square(normalized), axis=(-2, -1), keepdims=True))
            return (normalized * norm / (new_norm + eps)).astype(u.dtype)

        normalized = jax.tree.map(normalize, updates, moments)
        return _match_named_sharding_to_params(normalized, params), RowMomentState(moments)

    return optax.GradientTransformation(init_fn, update_fn)


def normuonh(
    *,
    learning_rate: float,
    beta2: float,
    momentum: float,
    nesterov: bool,
    steps: int,
    muon_eps: float,
    coefficient_type: CoefficientType,
) -> optax.GradientTransformation:
    """Apply neuron-normalized directions through the unchanged MuonH hyperball."""
    direction = optax.chain(
        _grug_scale_with_muon(momentum, nesterov, steps, muon_eps, coefficient_type),
        normalize_neuron_updates(beta2),
    )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("MuonH requires parameters for its norm-preserving projection")
        updates, next_state = direction.update(updates, state, params)
        return _scale_invariant_hyperball_updates(params, updates, learning_rate), next_state

    return optax.GradientTransformation(direction.init, update_fn)


def row_weight_muonh(
    *, learning_rate, momentum, nesterov, steps, muon_eps, coefficient_type
) -> optax.GradientTransformation:
    """MuonH on output-neuron spheres, preserving each row's initial weight norm.

    Apply the original H relative step and retraction independently to each
    output neuron. KMA's [..., input, output] layout makes the row axis -2;
    layer and expert dimensions remain independent. No update-moment EMA is used.
    """
    direction = _grug_scale_with_muon(momentum, nesterov, steps, muon_eps, coefficient_type)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("Row-weight MuonH requires parameters for its projection")
        updates, next_state = direction.update(updates, state, params)
        updates = _match_named_sharding_to_params(updates, params)

        def project(p, u):
            p32, u32 = p.astype(jnp.float32), u.astype(jnp.float32)
            radius = jnp.sqrt(jnp.sum(p32**2, axis=-2, keepdims=True))
            direction_norm = jnp.sqrt(jnp.sum(u32**2, axis=-2, keepdims=True))
            candidate = _pin_sharding(p32 - learning_rate * u32 * radius / jnp.maximum(direction_norm, 1e-10), p)
            candidate_norm = jnp.sqrt(jnp.sum(candidate**2, axis=-2, keepdims=True))
            return (candidate * radius / jnp.maximum(candidate_norm, 1e-10) - p32).astype(p.dtype)

        return _match_named_sharding_to_params(jax.tree.map(project, params, updates), params), next_state

    return optax.GradientTransformation(direction.init, update_fn)
