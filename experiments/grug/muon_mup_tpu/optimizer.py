# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original-scaling Muon, preserving the supplied reference Adam groups."""

from dataclasses import dataclass

import jax
import optax
from levanter.optim.config import OptimizerConfig
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.fast_track.adamh import scale_by_adamh
from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon
from experiments.grug.fast_track.optimizer import (
    GrugMoeMuonHConfig,
    _match_named_sharding_to_params,
    _match_named_update_sharding,
    _scale_by_adam_gate_router_decay,
    scale_with_grug_muonh,
)
from experiments.grug.muon_mup_tpu.decoupled import muon_md_step, muown_step
from experiments.grug.muon_mup_tpu.row_normalized import normuonh, row_weight_muonh


def muon_step(*, learning_rate, weight_decay, momentum, nesterov, steps, eps, coefficient_type):
    direction = _grug_scale_with_muon(momentum, nesterov, steps, eps, coefficient_type)

    def update(grads, state, params):
        updates, state = direction.update(grads, state, params)
        updates = _match_named_sharding_to_params(updates, params)

        def step(u, p):
            if u is None:
                return None
            # The shared transform already applies sqrt(max(1, fan_out/fan_in)).
            # Decoupled decay uses the group LR, without the shape multiplier.
            return -learning_rate * (u + weight_decay * p)

        return jax.tree.map(step, updates, params, is_leaf=lambda x: x is None), state

    return optax.GradientTransformation(direction.init, update)


@OptimizerConfig.register_subclass("muon_mup_tpu")
@dataclass(frozen=True)
class MuonComparisonConfig(GrugMoeMuonHConfig):
    """Original Muon matrices with sqrt(max(1, fan_out/fan_in)) scaling.

    This ablation does not reparameterize the network's readout or initializer.
    AdamH head and all Adam groups retain their reference learning rates.
    """

    matrix_weight_decay: float = 0.1
    head_lr: float = 0.02242598651082779

    def head_transform(self, lr):
        return scale_by_adamh(self.beta1, self.beta2, self.epsilon, lr)

    def fallback_transform(self, lr, num_train_steps):
        return optax.chain(
            _scale_by_adam_gate_router_decay(
                self.beta1, self.beta2, self.epsilon, self.gate_router_weight_decay, num_train_steps
            ),
            optax.scale(-lr),
        )

    def query_transform(self, lr):
        return optax.chain(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon), optax.scale(-lr))

    def matrix_transform(self, lr):
        return muon_step(
            learning_rate=lr,
            weight_decay=self.matrix_weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
            steps=self.backend_steps,
            eps=self.muon_epsilon,
            coefficient_type=self.coefficient_type,
        )

    def build(self, num_train_steps):
        def optimizer(learning_rate, adam_lr, head_lr):
            def matrix(lr):
                return optax.chain(self.matrix_transform(lr), _match_named_update_sharding())

            return optax.multi_transform(
                {
                    "muonh": matrix(learning_rate),
                    "kda_beta": matrix(learning_rate * self.kda_beta_lr_mult),
                    "adamh": self.head_transform(head_lr),
                    "adam": self.fallback_transform(adam_lr, num_train_steps),
                    "attn_res_query": self.query_transform(adam_lr * self.attn_res_query_lr_scale),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps),
            adam_lr=self.lr_scheduler(num_train_steps, override_lr=self.adam_lr),
            head_lr=self.lr_scheduler(num_train_steps, override_lr=self.head_lr),
        )


@OptimizerConfig.register_subclass("muonh_mup_tpu_sweep")
@dataclass(frozen=True)
class MuonHSweepConfig(GrugMoeMuonHConfig):
    """Reference MuonH with a separately scheduled AdamH head learning rate."""

    head_lr: float = 0.02242598651082779

    def matrix_transform(self, lr):
        return scale_with_grug_muonh(
            momentum=self.momentum,
            nesterov=self.nesterov,
            steps=self.backend_steps,
            muon_eps=self.muon_epsilon,
            learning_rate=lr,
            coefficient_type=self.coefficient_type,
        )

    def build(self, num_train_steps):
        def optimizer(learning_rate, adam_lr, head_lr):
            def clipped(*transforms):
                if self.max_grad_norm:
                    return optax.chain(optax.clip_by_global_norm(self.max_grad_norm), *transforms)
                return optax.chain(*transforms)

            def matrix(lr):
                return clipped(
                    self.matrix_transform(lr),
                    _match_named_update_sharding(),
                )

            adam = (
                _scale_by_adam_gate_router_decay(
                    self.beta1, self.beta2, self.epsilon, self.gate_router_weight_decay, num_train_steps
                )
                if self.gate_router_weight_decay > 0.0
                else optax.scale_by_adam(self.beta1, self.beta2, self.epsilon)
            )
            return optax.multi_transform(
                {
                    "muonh": matrix(learning_rate),
                    "adamh": clipped(scale_by_adamh(self.beta1, self.beta2, self.epsilon, head_lr)),
                    "adam": clipped(adam, optax.scale(-adam_lr)),
                    "attn_res_query": clipped(
                        optax.scale_by_adam(self.beta1, self.beta2, self.epsilon),
                        optax.scale(-adam_lr * self.attn_res_query_lr_scale),
                    ),
                    "kda_beta": matrix(learning_rate * self.kda_beta_lr_mult),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps),
            adam_lr=self.lr_scheduler(num_train_steps, override_lr=self.adam_lr),
            head_lr=self.lr_scheduler(num_train_steps, override_lr=self.head_lr),
        )


@OptimizerConfig.register_subclass("normuonh_kma")
@dataclass(frozen=True)
class NorMuonHConfig(MuonHSweepConfig):
    """NorMuon row moments with the reference MuonH projection and auxiliary groups."""

    row_beta2: float = 0.95

    def matrix_transform(self, lr):
        return normuonh(
            learning_rate=lr,
            beta2=self.row_beta2,
            momentum=self.momentum,
            nesterov=self.nesterov,
            steps=self.backend_steps,
            muon_eps=self.muon_epsilon,
            coefficient_type=self.coefficient_type,
        )


@OptimizerConfig.register_subclass("row_muonh_kma")
@dataclass(frozen=True)
class RowNormalizedMuonHConfig(MuonHSweepConfig):
    """Reference MuonH with independent output-neuron weight spheres."""

    def matrix_transform(self, lr):
        return row_weight_muonh(
            learning_rate=lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            steps=self.backend_steps,
            muon_eps=self.muon_epsilon,
            coefficient_type=self.coefficient_type,
        )


def logical_matrix_decay_mask(params):
    """Exclude vector parameters even when ArrayStacked adds a layer dimension."""

    def decay(param, path):
        path = ".".join(path) if isinstance(path, (tuple, list)) else str(path)
        logical_rank = param.ndim - int(".stacked." in path)
        return logical_rank >= 2

    return jax.tree.map(decay, params, leaf_key_paths(params))


@OptimizerConfig.register_subclass("muonw_mup_tpu")
@dataclass(frozen=True)
class MuonWComparisonConfig(MuonComparisonConfig):
    """Original-scaling Muon with AdamW on every fallback group."""

    def head_transform(self, lr):
        return optax.adamw(
            lr,
            b1=self.beta1,
            b2=self.beta2,
            eps=self.epsilon,
            weight_decay=self.matrix_weight_decay,
            mask=logical_matrix_decay_mask,
        )

    def fallback_transform(self, lr, num_train_steps):
        return self.head_transform(lr)

    def query_transform(self, lr):
        return self.head_transform(lr)


@OptimizerConfig.register_subclass("muown_kma")
@dataclass(frozen=True)
class MuownConfig(MuonWComparisonConfig):
    """Muown hidden matrices; the reference AdamW auxiliary groups are unchanged."""

    coefficient_type: CoefficientType = "simple"
    muon_epsilon: float = 1e-7
    gain_beta1: float = 0.9
    gain_beta2: float = 0.95
    gain_epsilon: float = 1e-8

    def matrix_transform(self, lr):
        return muown_step(
            learning_rate=lr,
            weight_decay=self.matrix_weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
            steps=self.backend_steps,
            coefficient_type=self.coefficient_type,
            eps=self.muon_epsilon,
            beta1=self.gain_beta1,
            beta2=self.gain_beta2,
            adam_eps=self.gain_epsilon,
        )


@OptimizerConfig.register_subclass("muon_md_kma")
@dataclass(frozen=True)
class MuonMDConfig(MuonWComparisonConfig):
    """MD hidden matrices with scheduled gains and unchanged AdamW auxiliary groups."""

    matrix_weight_decay: float = 0.0
    auxiliary_weight_decay: float = 0.1
    gain_lr: float = 0.001
    muon_epsilon: float = 1e-7
    gain_beta1: float = 0.9
    gain_beta2: float = 0.99
    gain_epsilon: float = 1e-8

    def head_transform(self, lr):
        return optax.adamw(
            lr,
            b1=self.beta1,
            b2=self.beta2,
            eps=self.epsilon,
            weight_decay=self.auxiliary_weight_decay,
            mask=logical_matrix_decay_mask,
        )

    def matrix_transform(self, lr):
        return muon_md_step(
            learning_rate=lr,
            gain_learning_rate=lr * self.gain_lr / self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
            steps=self.backend_steps,
            coefficient_type=self.coefficient_type,
            eps=self.muon_epsilon,
            beta1=self.gain_beta1,
            beta2=self.gain_beta2,
            adam_eps=self.gain_epsilon,
        )
