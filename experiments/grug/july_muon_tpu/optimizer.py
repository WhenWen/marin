# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""July optimizer ablations with independently fixed fallback learning rates."""

from dataclasses import dataclass

import jax
import optax
from levanter.optim.config import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon

from experiments.grug.july_muon_tpu.adamh import scale_by_adamh
from experiments.grug.july_muon_tpu.reference_optimizer import (
    GrugMoeMuonHConfig,
    _match_named_sharding_to_params,
    _match_named_update_sharding,
    scale_with_grug_muonh,
)


def matrix_decay_mask(params):
    return jax.tree.map(lambda p: p.ndim >= 2, params)


@OptimizerConfig.register_subclass("july_muonh_sweep")
@dataclass(frozen=True)
class JulyMuonHConfig(GrugMoeMuonHConfig):
    head_lr: float = 0.009804100203726365

    def matrix_transform(self, lr):
        return scale_with_grug_muonh(
            momentum=self.momentum,
            nesterov=self.nesterov,
            steps=self.backend_steps,
            muon_eps=self.muon_epsilon,
            learning_rate=lr,
            coefficient_type=self.coefficient_type,
        )

    def head_transform(self, lr):
        return scale_by_adamh(self.beta1, self.beta2, self.epsilon, lr)

    def fallback_transform(self, lr):
        return optax.chain(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon), optax.scale(-lr))

    def build(self, num_train_steps):
        def optimizer(learning_rate, adam_lr, head_lr):
            def clipped(*transforms):
                if self.max_grad_norm:
                    return optax.chain(optax.clip_by_global_norm(self.max_grad_norm), *transforms)
                return optax.chain(*transforms)

            return optax.multi_transform(
                {
                    "muonh": clipped(self.matrix_transform(learning_rate), _match_named_update_sharding()),
                    "adamh": clipped(self.head_transform(head_lr)),
                    "adam": clipped(self.fallback_transform(adam_lr)),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps),
            adam_lr=self.lr_scheduler(num_train_steps, override_lr=self.adam_lr),
            head_lr=self.lr_scheduler(num_train_steps, override_lr=self.head_lr),
        )


@OptimizerConfig.register_subclass("july_muonw_sweep")
@dataclass(frozen=True)
class JulyMuonWConfig(JulyMuonHConfig):
    """Original Muon scaling and decoupled decay; all fallback groups use AdamW."""

    def matrix_transform(self, lr):
        direction = _grug_scale_with_muon(
            momentum=self.momentum,
            nesterov=self.nesterov,
            steps=self.backend_steps,
            muon_eps=self.muon_epsilon,
            use_kimi_scaling=False,
            coefficient_type=self.coefficient_type,
        )

        def update(grads, state, params):
            updates, state = direction.update(grads, state, params)
            updates = _match_named_sharding_to_params(updates, params)
            updates = jax.tree.map(lambda u, p: -lr * (u + self.weight_decay * p), updates, params)
            return updates, state

        return optax.GradientTransformation(direction.init, update)

    def head_transform(self, lr):
        return self.fallback_transform(lr)

    def fallback_transform(self, lr):
        return optax.adamw(
            lr,
            b1=self.beta1,
            b2=self.beta2,
            eps=self.epsilon,
            weight_decay=self.weight_decay,
            mask=matrix_decay_mask,
        )
