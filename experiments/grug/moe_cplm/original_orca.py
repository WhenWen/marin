# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ORCA's standard-Muon update and uniform release schedule for the Belay experiment."""

import re
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon, _merge_heads, _split_heads
from experiments.grug.fast_track.optimizer import (
    GrugMoeMuonHConfig,
    _match_named_sharding_to_params,
    _match_named_update_sharding,
)
from experiments.grug.fast_track.orca import sharded_orthogonal_gradient

QK_PROJECTIONS = re.compile(r"(?:stacked_blocks|kda_blocks)(?:_tail)?\.stacked\.attn\.(w_q|w_k|w_uk|w_q2|w_uk2)$")
MATRIX_GROUPS = frozenset({"muonh", "kda_beta", "muonh_attn", "muonh_routed", "muonh_router", "upper_qk", "muonh_qk"})


class OrcaState(NamedTuple):
    count: jax.Array
    muon: optax.OptState


def path_string(path):
    return ".".join(path) if isinstance(path, (list, tuple)) else str(path)


def scale_original_orca(learning_rate, *, strength, weight_decay, release_step, head_dim):
    """Apply task-only Muon, decay, then shaping on the decayed weights.

    Attention Q/K projections are separate matrices per head. Other matrices,
    including experts and copy Q/K, remain whole. Matrix update RMS is 0.2.
    """
    muon = _grug_scale_with_muon(momentum=0.95, nesterov=True, steps=8, coefficient_type="polar_express")

    def split(value, path):
        return _split_heads(value, 2, head_dim) if QK_PROJECTIONS.fullmatch(path_string(path)) else value

    def views(tree):
        return jax.tree.map(split, tree, leaf_key_paths(tree))

    def init_fn(params):
        return OrcaState(jnp.zeros([], jnp.int32), muon.init(views(params)))

    def update_fn(grads, state, params=None):
        paths = leaf_key_paths(params)
        weights = views(params)
        directions, muon_state = muon.update(views(grads), state.muon, weights)
        directions = _match_named_sharding_to_params(directions, weights)

        def update(weight, direction, original, path):
            name = path_string(path)
            rms = jnp.sqrt(jnp.mean(jnp.square(direction.astype(jnp.float32)), axis=(-2, -1), keepdims=True))
            direction = 0.2 * direction / jnp.maximum(rms, 1e-12)
            decayed = (1 - learning_rate * weight_decay) * weight
            correction = jax.lax.cond(
                state.count < release_step,
                sharded_orthogonal_gradient,
                jnp.zeros_like,
                decayed,
            )
            delta = -learning_rate * (weight_decay * weight + strength * correction + direction)
            if QK_PROJECTIONS.fullmatch(name):
                return _merge_heads(delta, original, 2)
            return delta.astype(original.dtype)

        updates = jax.tree.map(update, weights, directions, params, paths)
        return updates, OrcaState(optax.safe_int32_increment(state.count), muon_state)

    return optax.chain(optax.GradientTransformation(init_fn, update_fn), _match_named_update_sharding())


@OptimizerConfig.register_subclass("belay_original_orca")
@dataclass(frozen=True)
class BelayOrcaConfig(OptimizerConfig):
    """Paper ORCA matrix rule with Belay's parameter grouping and Adam LR multipliers."""

    grouping: GrugMoeMuonHConfig = field(default_factory=GrugMoeMuonHConfig)
    adam_lr: float = 0.0024965273848782323
    strength: float = 1.0
    active_fraction: float = 0.25
    head_dim: int = 128
    backend_steps: int = 8
    warmup: int = 1000
    decay: float = 0.15
    lr_schedule: str = "linear"
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1

    def create_mask(self, params):
        return jax.tree.map(
            lambda label: "matrix" if label in MATRIX_GROUPS else label, self.grouping.create_mask(params)
        )

    def build(self, num_train_steps):
        base = self.grouping
        # Preserve architecture-specific fallback LR multipliers while removing
        # the output head's hyperball constraint and using ordinary AdamW.
        multipliers = {
            "adam": 1.0,
            "adamh": 1.0,
            "embed2": base.embed2_lr_mult,
            "attn_res_query": base.attn_res_query_lr_scale,
            "attn_res_token_query": base.attn_res_query_lr_scale * base.attn_res_token_query_lr_mult,
            "router": base.router_lr_mult,
            "kda_decay": base.kda_decay_lr_mult,
            "kda_beta_adam": base.kda_beta_adam_lr_mult,
            "mix_bias": 1.0,
        }

        def optimizer(learning_rate, adam_lr):
            transforms = {
                name: optax.adamw(
                    adam_lr * multiplier,
                    b1=base.beta1,
                    b2=base.beta2,
                    eps=base.epsilon,
                    weight_decay=self.weight_decay,
                    mask=lambda tree: jax.tree.map(
                        lambda p, path: p.ndim >= 2
                        and not path_string(path).endswith(".weight")
                        and "attn_res_query" not in path_string(path),
                        tree,
                        leaf_key_paths(tree),
                    ),
                )
                for name, multiplier in multipliers.items()
            }
            transforms["frozen"] = optax.set_to_zero()
            transforms["matrix"] = scale_original_orca(
                learning_rate,
                strength=self.strength,
                weight_decay=self.weight_decay,
                release_step=int(self.active_fraction * num_train_steps),
                head_dim=self.head_dim,
            )
            return optax.chain(optax.multi_transform(transforms, self.create_mask), _match_named_update_sharding())

        return optax.inject_hyperparams(optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps),
            adam_lr=self.lr_scheduler(num_train_steps, override_lr=self.adam_lr),
        )
