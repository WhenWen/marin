# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-width Muon spectral scaling ablation, preserving the reference Adam groups."""

import math
from dataclasses import dataclass

import jax
import optax
from levanter.optim.config import OptimizerConfig

from experiments.grug.fast_track.adamh import scale_by_adamh
from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon
from experiments.grug.fast_track.optimizer import (
    GrugMoeMuonHConfig,
    _match_named_sharding_to_params,
    _match_named_update_sharding,
    _scale_by_adam_gate_router_decay,
)


def muon_step(*, learning_rate, weight_decay, spectral, momentum, nesterov, steps, eps, coefficient_type):
    direction = _grug_scale_with_muon(momentum, nesterov, steps, eps, coefficient_type)

    def update(grads, state, params):
        updates, state = direction.update(grads, state, params)
        updates = _match_named_sharding_to_params(updates, params)

        def step(u, p):
            if u is None:
                return None
            # The shared transform already applies sqrt(max(1, fan_out/fan_in)).
            # Correct only the direction; decoupled decay uses the group LR.
            fan_in, fan_out = p.shape[-2:]
            correction = math.sqrt(min(1.0, fan_out / fan_in)) if spectral else 1.0
            return -learning_rate * (correction * u + weight_decay * p)

        return jax.tree.map(step, updates, params, is_leaf=lambda x: x is None), state

    return optax.GradientTransformation(direction.init, update)


@OptimizerConfig.register_subclass("muon_mup_tpu")
@dataclass(frozen=True)
class MuonComparisonConfig(GrugMoeMuonHConfig):
    """Muon matrices with either original or RMS-to-RMS spectral scaling.

    This ablation does not reparameterize the network's readout or initializer.
    AdamH head and all Adam groups retain their reference learning rates.
    """

    spectral: bool = True
    matrix_weight_decay: float = 0.1
    head_lr: float = 0.02242598651082779

    def build(self, num_train_steps):
        def optimizer(learning_rate, adam_lr, head_lr):
            def matrix(lr):
                return optax.chain(
                    muon_step(
                        learning_rate=lr,
                        weight_decay=self.matrix_weight_decay,
                        spectral=self.spectral,
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        eps=self.muon_epsilon,
                        coefficient_type=self.coefficient_type,
                    ),
                    _match_named_update_sharding(),
                )

            return optax.multi_transform(
                {
                    "muonh": matrix(learning_rate),
                    "kda_beta": matrix(learning_rate * self.kda_beta_lr_mult),
                    "adamh": scale_by_adamh(self.beta1, self.beta2, self.epsilon, head_lr),
                    "adam": optax.chain(
                        _scale_by_adam_gate_router_decay(
                            self.beta1, self.beta2, self.epsilon, self.gate_router_weight_decay, num_train_steps
                        ),
                        optax.scale(-adam_lr),
                    ),
                    "attn_res_query": optax.chain(
                        optax.scale_by_adam(self.beta1, self.beta2, self.epsilon),
                        optax.scale(-adam_lr * self.attn_res_query_lr_scale),
                    ),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps),
            adam_lr=self.lr_scheduler(num_train_steps, override_lr=self.adam_lr),
            head_lr=self.lr_scheduler(num_train_steps, override_lr=self.head_lr),
        )
