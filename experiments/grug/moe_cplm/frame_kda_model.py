# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A protected-frame KDA residual before the unchanged CPLM Plus head."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from experiments.grug.fast_track import model
from experiments.grug.moe_cplm import frame_kda, plus


def local_read(hidden, segments, weights, gain, bias, *, heads, depth, rank, width, chunk):
    batch, length, _ = hidden.shape
    x = hidden.astype(jnp.float32)
    wq, wk, wv, wg, wb, wa, wd, wo = weights

    def project(weight, size):
        return jnp.einsum("btd,df->btf", x, weight.astype(jnp.float32)).reshape(batch, length, heads, size)

    def unit(value):
        return value * jax.lax.rsqrt(jnp.sum(value * value, -1, keepdims=True) + 1e-6)

    query, key = unit(project(wq, rank)), unit(project(wk, rank))
    value = project(wv, width)
    value = value * jax.lax.rsqrt(jnp.mean(value * value, -1, keepdims=True) + 1e-6)
    decay = jnp.exp(-jax.nn.softplus(project(wg, rank) - 4.0))
    beta = jax.nn.sigmoid(project(wb, 1)[..., 0])
    actions = jax.nn.softmax(project(wa, 4) + bias.reshape(heads, 4), -1)
    address = jax.nn.softmax(project(wd, depth), -1)
    starts = jnp.pad(segments[:, 1:] != segments[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    keep = jnp.broadcast_to((~starts)[..., None], (batch, length, heads)).astype(jnp.float32)
    initial = jnp.zeros((batch, heads, depth, rank, width), jnp.float32)
    read, _ = frame_kda.chunked((actions, query, key, value, decay, beta, address, keep), initial, chunk)
    delta = jnp.einsum("btf,fd->btd", read.reshape(batch, length, heads * width), wo.astype(jnp.float32))
    return (delta * gain).astype(hidden.dtype)


class FrameKDAPlus(plus.BelayCPLMPlus):
    frame_weights: tuple[jax.Array, ...]
    frame_gain: jax.Array
    frame_action_bias: jax.Array

    @staticmethod
    def init(cfg, *, key):
        base = plus.BelayCPLMPlus.init(cfg, key=key)
        heads, rank, width = cfg.frame_kda_heads, cfg.frame_kda_rank, cfg.frame_kda_width
        d, depth = cfg.hidden_dim, cfg.frame_kda_depth
        shapes = [(d, heads * size) for size in (rank, rank, width, rank, 1, 4, depth)] + [(heads * width, d)]
        keys = jax.random.split(jax.random.fold_in(key, 261021301), len(shapes))
        weights = tuple(jax.sharding.reshard(0.02 * jax.random.normal(k, shape), P()) for k, shape in zip(keys, shapes, strict=True))
        return FrameKDAPlus(
            **{field.name: getattr(base, field.name) for field in dataclasses.fields(base)},
            frame_weights=weights,
            frame_gain=jax.sharding.reshard(jnp.zeros(d), P()),
            frame_action_bias=jax.sharding.reshard(jnp.tile(jnp.array([0.0, 0.0, 1.0, 0.0]), heads), P()),
        )

    def copy_head(self, hidden, token_ids, labels, segments):
        cfg = self.config
        hs, ts = P(model._BATCH_AXES, None, None), P(model._BATCH_AXES, None)
        specs = (hs, ts, tuple(P() for _ in self.frame_weights), P(), P())
        args = (hidden, segments, self.frame_weights, self.frame_gain, self.frame_action_bias)
        args = jax.tree.map(jax.sharding.reshard, args, specs)

        def read(*values):
            return local_read(*values, heads=cfg.frame_kda_heads, depth=cfg.frame_kda_depth,
                              rank=cfg.frame_kda_rank, width=cfg.frame_kda_width, chunk=cfg.frame_kda_chunk)

        delta = jax.shard_map(read, in_specs=specs, out_specs=hs, check_vma=False)(*args)
        return plus.BelayCPLMPlus.copy_head(self, jax.sharding.reshard(hidden, hs) + delta, token_ids, labels, segments)
