# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused-weight Muown and magnitude-direction updates for KMA matrices.

KMA stores logical matrices as [..., input, output], the transpose of the
papers' convention. Leading layer/expert axes are independent matrices.
Sources: Muown 2605.10797, MD 2606.25971, Algorithm 2. The shared Newton-Schulz
kernel supplies the original Muon shape factor, which is removed below.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon
from experiments.grug.fast_track.optimizer import _match_named_sharding_to_params, _pin_sharding


class MuownState(NamedTuple):
    magnitude: object
    direction_norm: object
    direction: object
    adam: object


class MDState(NamedTuple):
    row_raw: object
    col_raw: object
    radius: object
    direction: object
    adam: object


def _norm(x, axis):
    return jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=axis, keepdims=True))


def _unscale(u):
    fan_in, fan_out = u.shape[-2:]
    return u / math.sqrt(max(1, fan_out / fan_in))


def muown_step(
    *,
    learning_rate,
    weight_decay,
    momentum=0.95,
    nesterov=True,
    steps=5,
    eps=1e-7,
    coefficient_type="simple",
    beta1=0.9,
    beta2=0.95,
    adam_eps=1e-8,
):
    """Muown with a shared direction/magnitude LR and fused-weight decay.

    Follows kcc-lion/muown at 3bd0c054. KMA has separate Q/K/V matrices;
    the author's shape-based fused-QKV heuristic is therefore not used.
    """
    muon = _grug_scale_with_muon(momentum, nesterov, steps, eps, coefficient_type)
    adam = optax.scale_by_adam(beta1, beta2, adam_eps)

    def init(params):
        magnitude = jax.tree.map(lambda p: _norm(p, -2), params)
        return MuownState(magnitude, magnitude, muon.init(params), adam.init(magnitude))

    def update(grads, state, params):
        unit = jax.tree.map(lambda p, g: p / g, params, state.magnitude)
        v = jax.tree.map(lambda u, r: u * r, unit, state.direction_norm)
        gg = jax.tree.map(lambda g, u: jnp.sum(g * u, axis=-2, keepdims=True), grads, unit)
        gv = jax.tree.map(
            lambda g, u, gg, mag, r: (mag / r) * (g - u * gg), grads, unit, gg, state.magnitude, state.direction_norm
        )
        directions, direction_state = muon.update(gv, state.direction, params)
        directions = _match_named_sharding_to_params(directions, params)
        v_new = jax.tree.map(
            lambda v, d, p: _pin_sharding(v - learning_rate * 0.2 * math.sqrt(max(d.shape[-2:])) * _unscale(d), p),
            v,
            directions,
            params,
        )
        r_new = jax.tree.map(lambda v: _norm(v, -2), v_new)
        dg, adam_state = adam.update(gg, state.adam, state.magnitude)
        g_new = jax.tree.map(lambda g, d: g - learning_rate * d, state.magnitude, dg)
        new = jax.tree.map(
            lambda v, r, g, p: _pin_sharding(g * (v / r) - learning_rate * weight_decay * p, p),
            v_new,
            r_new,
            g_new,
            params,
        )
        # Decay acts on the fused old weight; resynchronize the decomposition.
        if weight_decay != 0.0:
            g_new = jax.tree.map(lambda p: _norm(p, -2), new)
        updates = jax.tree.map(lambda n, p: n - p, new, params)
        return updates, MuownState(g_new, r_new, direction_state, adam_state)

    return optax.GradientTransformation(init, update)


def muon_md_step(
    *,
    learning_rate,
    gain_learning_rate,
    momentum=0.95,
    nesterov=True,
    steps=5,
    eps=1e-7,
    coefficient_type="quintic",
    beta1=0.9,
    beta2=0.99,
    adam_eps=1e-8,
):
    """MD with softplus row/column gains and the initialization Frobenius sphere.

    The Muon scale is radius / sqrt(min(input, output)), generalizing the
    paper's shape-up factor to KMA's different initialization. No matrix decay.
    """
    muon = _grug_scale_with_muon(momentum, nesterov, steps, eps, coefficient_type)
    adam = optax.scale_by_adam(beta1, beta2, adam_eps)

    def init(params):
        raw_one = math.log(math.expm1(1.0))
        row = jax.tree.map(lambda p: jnp.full_like(jnp.sum(p, axis=-2, keepdims=True), raw_one), params)
        col = jax.tree.map(lambda p: jnp.full_like(jnp.sum(p, axis=-1, keepdims=True), raw_one), params)
        radius = jax.tree.map(lambda p: _norm(p, (-2, -1)), params)
        return MDState(row, col, radius, muon.init(params), adam.init((row, col)))

    def update(grads, state, params):
        row = jax.tree.map(jax.nn.softplus, state.row_raw)
        col = jax.tree.map(jax.nn.softplus, state.col_raw)
        bare = jax.tree.map(lambda p, r, c: p / (r * c), params, row, col)
        gr = jax.tree.map(
            lambda w, g, c, raw: jnp.sum(w * g * c, axis=-2, keepdims=True) * jax.nn.sigmoid(raw),
            bare,
            grads,
            col,
            state.row_raw,
        )
        gc = jax.tree.map(
            lambda w, g, r, raw: jnp.sum(w * g * r, axis=-1, keepdims=True) * jax.nn.sigmoid(raw),
            bare,
            grads,
            row,
            state.col_raw,
        )
        gw = jax.tree.map(lambda g, r, c: g * r * c, grads, row, col)
        directions, direction_state = muon.update(gw, state.direction, params)
        directions = _match_named_sharding_to_params(directions, params)
        stepped = jax.tree.map(
            lambda w, d, radius, p: _pin_sharding(
                w - learning_rate * _unscale(d) * radius / math.sqrt(min(d.shape[-2:])), p
            ),
            bare,
            directions,
            state.radius,
            params,
        )
        normalized = jax.tree.map(lambda w, r: w * r / jnp.maximum(_norm(w, (-2, -1)), 1e-30), stepped, state.radius)
        (dr, dc), adam_state = adam.update((gr, gc), state.adam, (state.row_raw, state.col_raw))
        raw_r = jax.tree.map(lambda r, d: r - gain_learning_rate * d, state.row_raw, dr)
        raw_c = jax.tree.map(lambda c, d: c - gain_learning_rate * d, state.col_raw, dc)
        new = jax.tree.map(lambda w, r, c: w * jax.nn.softplus(r) * jax.nn.softplus(c), normalized, raw_r, raw_c)
        updates = _match_named_sharding_to_params(jax.tree.map(lambda n, p: n - p, new, params), params)
        return updates, MDState(raw_r, raw_c, state.radius, direction_state, adam_state)

    return optax.GradientTransformation(init, update)
