# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon spectral width scaling and explicit depth-dependent parameter groups."""

import dataclasses

import haliax as hax
import jax
import optax
from levanter.optim.adamh import scale_by_adamh
from levanter.optim.config import OptimizerConfig
from levanter.optim.muon import scale_with_muon
from levanter.optim.util import label_linear_like_module
from levanter.utils.jax_utils import leaf_key_paths


def masked_linear_transform(transform):
    """Keep masked-out Linear modules away from matrix-only native transforms."""

    def clean(tree):
        return jax.tree.map(
            lambda p: None if isinstance(p, optax.MaskedNode) else p,
            tree,
            is_leaf=lambda p: isinstance(p, optax.MaskedNode),
        )

    def init(params):
        return transform.init(clean(params))

    def update(grads, state, params):
        changes, state = transform.update(clean(grads), state, clean(params))
        changes = jax.tree.map(
            lambda before, after: before if isinstance(before, optax.MaskedNode) else after,
            grads,
            changes,
            is_leaf=lambda p: p is None or isinstance(p, optax.MaskedNode),
        )
        return changes, state

    return optax.GradientTransformation(init, update)


def projected_muon(learning_rate, momentum):
    direction = scale_with_muon(
        momentum=momentum, steps=5, muon_eps=1e-5, use_kimi_scaling=False, coefficient_type="simple"
    )

    def update(grads, state, params):
        directions, state = direction.update(grads, state, params)

        def project(param, change):
            if not isinstance(param, hax.nn.Linear) or not isinstance(param.weight, hax.NamedArray):
                return change
            axes = hax.concat_axis_specs(param.Out, param.In)
            radius = hax.sqrt(hax.sum(param.weight**2, axis=axes))
            norm = hax.sqrt(hax.sum(change.weight**2, axis=axes))
            proposal = param.weight - learning_rate * change.weight * radius / hax.maximum(norm, 1e-10)
            proposal_norm = hax.sqrt(hax.sum(proposal**2, axis=axes))
            delta = proposal * radius / hax.maximum(proposal_norm, 1e-10) - param.weight
            return dataclasses.replace(change, weight=delta)

        return jax.tree.map(project, params, directions, is_leaf=lambda p: isinstance(p, hax.nn.Linear)), state

    return optax.GradientTransformation(direction.init, update)


@OptimizerConfig.register_subclass("hybrid_transfer_muon")
@dataclasses.dataclass(frozen=True)
class TransferOptimizer(OptimizerConfig):
    variant: str = "muonh"
    adam_lr: float = 0.008
    width_multiplier: float = 1.0
    depth_multiplier: float = 1.0
    depth_alpha: float = 0.0
    hidden_lr_multiplier: float = 1.0
    momentum: float = 0.95
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-15

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def label(param, path):
            name = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if isinstance(param, hax.nn.Linear):
                group = "readout" if "lm_head" in name else "matrix"
                return label_linear_like_module(param, weight_label=group, bias_label="hidden_vector")
            if "residual_attention" in name and "query" in name:
                return "depth_query"
            if "layers" in name or "residual_attention" in name:
                return "hidden_vector"
            return "outer_vector"

        return hax.tree_util.tree_map(label, params, paths, is_leaf=lambda p: isinstance(p, hax.nn.Linear))

    def build(self, num_train_steps):
        muon_schedule = self.lr_scheduler(num_train_steps)
        adam_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        eps_outer = self.epsilon / self.width_multiplier
        eps_hidden = eps_outer * self.depth_multiplier ** (-self.depth_alpha)

        def make(learning_rate, adam_lr):
            hidden_lr = learning_rate * self.hidden_lr_multiplier

            def adam(lr, eps, decay=0.0):
                return optax.chain(
                    optax.scale_by_adam(self.beta1, self.beta2, eps),
                    optax.add_decayed_weights(decay),
                    optax.scale(-lr),
                )

            if self.variant == "muonh":
                matrix = projected_muon(hidden_lr, self.momentum)
                readout = scale_by_adamh(self.beta1, self.beta2, eps_outer, learning_rate)
            elif self.variant == "muonw":
                # Original Muon's sqrt(max(1, out/in)) spectral scaling already
                # yields width-independent operator updates. Do not multiply
                # this LR by Adam's 1/width rule (or use Kimi's sqrt(width) rule).
                matrix = optax.chain(
                    scale_with_muon(
                        momentum=self.momentum, steps=5, muon_eps=1e-5, use_kimi_scaling=False, coefficient_type="simple"
                    ),
                    optax.add_decayed_weights(self.weight_decay),
                    optax.scale(-hidden_lr),
                )
                readout = adam(adam_lr, eps_outer, self.weight_decay)
            else:
                raise ValueError(f"Unknown optimizer: {self.variant}")
            groups = {
                "matrix": masked_linear_transform(matrix),
                "readout": readout,
                "hidden_vector": adam(adam_lr * self.hidden_lr_multiplier, eps_hidden),
                "outer_vector": adam(adam_lr, eps_outer),
                # Full AttnRes pseudo-query is an unnormalized dot product
                # over width. Its zero init and 1/m_N LR keep learned scores
                # on a constant scale; this extension needs coordinate checks.
                "depth_query": adam(adam_lr / self.width_multiplier, eps_outer),
            }
            return optax.chain(optax.clip_by_global_norm(1.0), optax.multi_transform(groups, self.create_mask))

        return optax.inject_hyperparams(make)(learning_rate=muon_schedule, adam_lr=adam_schedule)
