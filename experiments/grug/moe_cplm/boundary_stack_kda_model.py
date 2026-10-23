# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Belay KDA with one soft stack-routing action per chunk boundary."""

import dataclasses
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

from experiments.grug.fast_track import model
from experiments.grug.moe import boundary_stack_kda

NativeKimiDeltaAttention = model.KimiDeltaAttention


class BoundaryStackKimiDeltaAttention(NativeKimiDeltaAttention):
    precision: str = eqx.field(static=True, default="mixed")
    w_stack_actions: jax.Array | None = None
    stack_action_bias: jax.Array | None = None

    @staticmethod
    def init(cfg, *, key, precision="mixed", stay_bias=6.0):
        base = NativeKimiDeltaAttention.init(cfg, key=key)
        if any(
            value is not None
            for value in (
                base.w_write,
                base.w_erase,
                base.w_beta_down,
                base.beta_scale,
                base.sconv_a,
                base.vres_lambda,
                base.push_decay,
                base.w_rot_down,
                base.comba_d,
                base.value_embed,
                base.bias_qkv,
            )
        ) or any(
            (
                cfg.kda_beta_negative,
                cfg.kda_decay_per_head,
                cfg.kda_gate_per_head,
                cfg.kda_head_pairing != model.KdaHeadPairing.NONE,
            )
        ):
            raise ValueError("Stack KDA uses the frozen native Belay gate/projection configuration")
        if cfg.num_heads != 4:
            raise ValueError("Head stack uses exactly four native KDA heads")
        d = cfg.hidden_dim
        return BoundaryStackKimiDeltaAttention(
            **{field.name: getattr(base, field.name) for field in dataclasses.fields(NativeKimiDeltaAttention)},
            precision=precision,
            w_stack_actions=reshard(jnp.zeros((d, 3)), P(None, None)),
            stack_action_bias=reshard(jnp.array([0.0, 0.0, stay_bias]), P(None)),
        )

    def __call__(
        self,
        x,
        segment_ids=None,
        proj_inputs=None,
        no_decay=False,
        no_beta=False,
        kv_share=None,
        value_residual=False,
        kv_input=None,
        token_ids=None,
    ):
        cfg = self.cfg
        if get_abstract_mesh().shape["model"] != 1:
            raise ValueError("Head stack currently requires all four slots on one device")
        batch, length, _ = x.shape
        heads, rank = cfg.num_heads, cfg.inferred_head_dim
        proj_inputs = proj_inputs or {}

        def project(weight, conv, name):
            source = kv_input if kv_input is not None and name in ("k", "v") else proj_inputs.get(name, x)
            value = jax.nn.silu(conv(model._proj(cfg, source, weight), segment_ids))
            return value.reshape(batch, length, heads, rank)

        q = project(self.w_q, self.sconv_q, "q")
        k = project(self.w_k, self.sconv_k, "k")
        v = project(self.w_v, self.sconv_v, "v")
        low = jnp.einsum("bsd,dr->bsr", x, self.w_a_down)
        decay_logits = jnp.einsum("bsr,re->bse", low, self.w_a_up).reshape(batch, length, heads, rank)
        scale = jnp.exp(self.a_log.astype(jnp.float32))[:, None]
        g = -model.KDA_MIN_LOG_DECAY * jax.nn.sigmoid(
            scale * (decay_logits.astype(jnp.float32) + self.dt_bias.astype(jnp.float32))
        )
        beta = jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.w_beta).astype(jnp.float32))
        if no_decay:
            g = jnp.zeros_like(g)
        if no_beta:
            beta = jnp.ones_like(beta)
        boundary = x[:, boundary_stack_kda.CHUNK - 1 :: boundary_stack_kda.CHUNK, :]
        logits = model._proj(cfg, boundary, self.w_stack_actions).astype(jnp.float32)
        actions = jax.nn.softmax(logits + self.stack_action_bias, axis=-1)
        spec4, spec3 = P(model._BATCH_AXES, None, "model", None), P(model._BATCH_AXES, None, "model")
        q, k, v, g = (reshard(t, spec4) for t in (q, k, v, g))
        action_spec = P(model._BATCH_AXES, None, None)
        actions = reshard(actions, action_spec)
        beta = reshard(beta, spec3)
        segments = jnp.zeros((batch, length), jnp.int32) if segment_ids is None else segment_ids
        segments = reshard(jnp.broadcast_to(segments, (batch, length)), P(model._BATCH_AXES, None))
        kernel = jax.shard_map(
            partial(boundary_stack_kda.native_kernel, precision=self.precision),
            mesh=get_abstract_mesh(),
            in_specs=(spec4, spec4, spec4, spec4, spec3, action_spec, P(model._BATCH_AXES, None)),
            out_specs=spec4,
            check_vma=False,
        )
        output = kernel(q, k, v, g, beta, actions, segments)
        output = self.o_norm(output.astype(x.dtype))
        output = jnp.reshape(output, (batch, length, heads * rank), out_sharding=P(model._BATCH_AXES, None, "model"))
        output *= jax.nn.sigmoid(model._proj(cfg, x, self.w_g))
        stats = {
            f"{model._KDA_STAT_PREFIX}stack_{action}": jax.lax.stop_gradient(jnp.mean(actions[..., index]))
            for index, action in enumerate(("push", "pop", "stay"))
        }
        return model._proj(cfg, output, self.w_o, out_sharding=model._batch_spec()), stats
