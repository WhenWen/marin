# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from levanter.optim import OptimizerConfig
from levanter.optim.config import LrSchedule, LrScheduleContext
from levanter.optim.grugmuon import _grug_scale_with_curvature_muon, _grug_scale_with_muon
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe.adamh import scale_by_adamh
from experiments.grug.moe.amuse import amuse


@LrSchedule.register_subclass("power_law_token")
@dataclass(frozen=True)
class PowerLawTokenDecay(LrSchedule):
    """Power-law LR decay anchored at peak: lr(token) = peak * (token/token_anchor)^(-exponent).

    Within the decay window (local step d, after warmup), lr = peak*(1 + d/warmup_steps)^(-p),
    i.e. lr = peak at the start of decay (token == end-of-warmup anchor) and decaying as a
    power law in total tokens thereafter. Clamped at min_lr.
    """

    exponent: float = 0.3

    def build(self, ctx: LrScheduleContext):
        w = max(float(ctx.warmup_steps), 1.0)
        p = float(self.exponent)
        peak = ctx.learning_rate
        min_lr = ctx.min_lr

        def schedule(local_decay_step):
            # join_schedules evaluates every segment at (step-boundary), negative before
            # this segment starts; clamp to >=0 so the base stays >=1 (value is discarded
            # by the join's select for pre-segment steps).
            d = jnp.maximum(jnp.asarray(local_decay_step, jnp.float32), 0.0)
            lr = peak * (1.0 + d / w) ** (-p)
            return jnp.maximum(lr, min_lr)

        return schedule


def _uses_adamh_baseline_adam_group(path_lower: str) -> bool:
    return (
        "token_embed" in path_lower
        or "router_bias" in path_lower
        or "attn_gate" in path_lower
        or ".router" in path_lower
    )


def _target_named_sharding(array) -> jax.sharding.NamedSharding | None:
    if array is None or not hasattr(array, "shape"):
        return None
    sharding = getattr(array, "sharding", None)
    if sharding is None:
        aval = jax.typeof(array)
        sharding = getattr(aval, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding
    return None


def _match_named_update_sharding() -> optax.GradientTransformation:
    """Restore named mesh sharding without touching single-device arrays."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        def match_sharding(update, param):
            if update is None:
                return None
            target_sharding = _target_named_sharding(param)
            if target_sharding is None:
                return update
            return jax.sharding.reshard(update, target_sharding)

        updates = jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _match_named_sharding_to_params(updates, params):
    def match_sharding(update, param):
        if update is None:
            return None
        target_sharding = _target_named_sharding(param)
        if target_sharding is None:
            return update
        return jax.sharding.reshard(update, target_sharding)

    return jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)


def _scale_invariant_hyperball_updates(params, direction_updates, learning_rate: float):
    direction_updates = _match_named_sharding_to_params(direction_updates, params)

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        if param.ndim == 2:
            param_norm = jnp.linalg.norm(param)
            update_norm = jnp.linalg.norm(update)
            new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.linalg.norm(new_param)
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    return jax.tree.map(
        scale_invariant_update,
        params,
        direction_updates,
        is_leaf=lambda x: x is None,
    )


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
    curvature_lambda: float = 0.0,
    curvature_beta: float = 0.95,
    inner_steps: int = 10,
    riemannian_maxbt: int = 10,
    two_sided: bool = True,
    constraint: str = "stiefel",
    curv_power: str = "sqrt",
    power_iters: int = 8,
    lambda_tracks_lr: bool = False,
    peak_lr: float = 0.0,
) -> optax.GradientTransformation:
    """MuonH transform for raw Grug arrays with matrix-shaped trailing dims.

    curvature_lambda > 0 swaps plain msign(N) for the curvature-corrected Riemannian inner-solve
    (two-sided P^{1/4} by default); 0 ⟹ plain MuonH. Hyperball applied below either way.

    lambda_tracks_lr: scale the curvature coefficient by lr_t/peak_lr each step (curvature strongest at peak
    LR, fading during warmup / linear decay). peak_lr is the schedule's peak (= config.learning_rate).
    """
    if curvature_lambda and curvature_lambda > 0.0:
        muon_transform = _grug_scale_with_curvature_muon(
            momentum=momentum,
            nesterov=nesterov,
            steps=steps,
            muon_eps=muon_eps,
            coefficient_type=coefficient_type,
            curvature_beta=curvature_beta,
            curvature_lambda=curvature_lambda,
            inner_steps=inner_steps,
            riemannian_maxbt=riemannian_maxbt,
            two_sided=two_sided,
            constraint=constraint,
            curv_power=curv_power,
            power_iters=power_iters,
        )
    else:
        muon_transform = _grug_scale_with_muon(
            momentum=momentum,
            nesterov=nesterov,
            steps=steps,
            muon_eps=muon_eps,
            use_kimi_scaling=False,
            coefficient_type=coefficient_type,
        )

    def init_fn(params):
        return muon_transform.init(params)

    def update_fn(updates, state, params=None, *, lr_scale=1.0, **_extra):
        # ``lr_scale`` (default 1.0) multiplies the effective LR *inside* the
        # hyperball so the norm-preserving (angular) step is recomputed at the
        # true scaled LR. The hyperball renormalizes onto the ‖param‖ sphere, so
        # the step is NONLINEAR in the LR — scaling the output delta would be a
        # straight chord off the sphere, not the true MuonH step at lr·lr_scale.
        # ``_extra`` (e.g. value=loss) is dropped.
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")

        # Force fp32 matmuls in the optimizer: TPU runs fp32 inputs through bf16 matmuls by default, which
        # wrecks the iterative curvature solve (Newton-Schulz, P^{1/4}, Riemannian line search) and makes it
        # non-deterministic. The optimizer must always run in true fp32.
        # λ schedule: lam_scale = lr_t/peak_lr (traced) when lambda_tracks_lr, else 1.0. Note this uses the
        # SCHEDULED lr (not lr·lr_scale) so λ follows warmup/decay, matching the qwen3 lambda_tracks_lr.
        # Only the curvature transform accepts lam_scale; the plain-Muon (λ=0) update_fn does not.
        lam_scale = (learning_rate / peak_lr) if (lambda_tracks_lr and peak_lr > 0.0) else 1.0
        curv_kwargs = {"lam_scale": lam_scale} if (curvature_lambda and curvature_lambda > 0.0) else {}
        with jax.default_matmul_precision("highest"):
            muon_updates, next_state = muon_transform.update(updates, state, params, **curv_kwargs)
            muonh_updates = _scale_invariant_hyperball_updates(params, muon_updates, learning_rate * lr_scale)
        return muonh_updates, next_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _scale_by_lr_scale() -> optax.GradientTransformationExtraArgs:
    """Multiply updates by the runtime ``lr_scale`` extra-arg (default 1.0).

    For LR-*linear* base groups (plain AdamW, ``optax.scale(-lr)`` Muon) an
    output scale by ``lr_scale`` is exactly equivalent to running at lr·lr_scale.
    The hyperball groups (MuonH / AdamH) are nonlinear and instead consume
    ``lr_scale`` inside their own update. Lets the AMUSE per-batch line search
    set a uniform effective LR across all parameter groups.
    """

    def update_fn(updates, state, params=None, *, lr_scale=1.0, **_extra):
        return jax.tree.map(lambda u: lr_scale * u, updates), state

    return optax.GradientTransformationExtraArgs(lambda _params: optax.EmptyState(), update_fn)


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.expert_mlp.w_gate_up,
      mlp.expert_mlp.w_down, shared.w_*)
    - adam: norms, biases, router, embeddings, attention gates (1D / small params)
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    adam_lr: float = 6e-4
    expert_lr: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        expert_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=expert_lr_val)

        def optimizer(learning_rate, adam_lr, expert_lr):
            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                return optax.chain(*components)

            def adamh_expert_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, expert_lr))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "adamh": adamh_transform(),
                    "adamh_expert": adamh_expert_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            expert_lr=expert_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower:
                return "adam"
            if "router_bias" in path_lower or "attn_gate" in path_lower or ".router" in path_lower:
                return "adam"
            if ".mlp.expert_mlp.w_" in path_lower or ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """May Recipe MuonH optimizer: 3 LR groups (muonh / adamh / adam).

    Three LR groups:
    - ``muonh``: matrices (attn, MoE MLP, shared) **and** all GatedNorms.
      Newton-Schulz orthogonalisation + Frobenius hyperball scale-invariant step.
    - ``adamh``: ``lm_head`` / ``output_proj``.
    - ``adam``: ``token_embed`` / ``router`` / ``router_bias`` / ``attn_gate``
      / 1-D norm weights.

    ``max_grad_norm`` defaults to ``None`` here (no clipping) for the 1pct-noclip
    schedule used by the May Recipe baseline.
    """

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    coefficient_type: CoefficientType = "quintic"
    # Curvature-corrected Muon (0 = plain MuonH = #6153 baseline). >0 swaps msign(N) for the
    # Riemannian curvature inner-solve, keeping the standard linear-decay schedule.
    curvature_lambda: float = 0.0
    curvature_beta: float = 0.95
    curvature_inner_steps: int = 10
    curvature_maxbt: int = 10
    curvature_two_sided: bool = True
    curvature_constraint: str = "stiefel"
    curv_power: str = "sqrt"
    # If True, the curvature strength tracks the LR schedule: λ_t = curvature_lambda · lr_t/peak_lr.
    curvature_lambda_tracks_lr: bool = False

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        coefficient_type=self.coefficient_type,
                        curvature_lambda=self.curvature_lambda,
                        curvature_beta=self.curvature_beta,
                        inner_steps=self.curvature_inner_steps,
                        riemannian_maxbt=self.curvature_maxbt,
                        two_sided=self.curvature_two_sided,
                        constraint=self.curvature_constraint,
                        curv_power=self.curv_power,
                        lambda_tracks_lr=self.curvature_lambda_tracks_lr,
                        peak_lr=self.learning_rate,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, lr))
                return optax.chain(*components)

            def adam_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            transforms = {
                "muonh": muonh_transform(),
                "adamh": adamh_transform_at(learning_rate),
                "adam": adam_transform_at(adam_lr),
            }
            return optax.multi_transform(transforms, self.create_mask)

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if (
                "token_embed" in path_lower
                or "router_bias" in path_lower
                or path_lower.endswith(".attn_gate")
                or ".router" in path_lower
            ):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            # GatedNorms route to muonh (NS + Frobenius hyperball), same as matrices.
            if "gated_norm" in path_lower:
                return "muonh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_amuse_v1")
@dataclass(frozen=True)
class GrugMoeAmuseConfig(OptimizerConfig):
    """AMUSE for the May-arch Grug MoE (arxiv 2605.22432).

    Time-varying Schedule-Free wrapper over per-group base optimizers:
      - matrix params (ndim in (2, 3)) and GatedNorms outside the baseline-adam
        group → AMUSE(MuonH base)
      - lm head / output proj → AMUSE(AdamH base) at matrix LR
      - 1-D / embedding / router params → AMUSE(AdamW-no-β1 base) at adam_lr

    Per the AMUSE algorithm, the Adam path drops β_1 momentum (SF replaces it
    with the time-varying β_t interpolation). Muon keeps its own μ momentum
    (operates pre-orthogonalization, distinct from SF). The LR schedule is
    warmup + constant (AMUSE removes the need for a decay phase).

    Defaults: beta_1=0.6, rho=0.8, T_0=2000 from the paper's d512 LLM recipe.
    The grug trainer mirrors ``AmuseState.x`` (the averaged X sequence) into
    ``ema_params`` so the X-model eval reports AMUSE's headline metric.
    """

    adam_lr: float = 6e-4
    muon_momentum: float = 0.95
    # AMUSE pseudocode specifies plain heavy-ball (no Nesterov). MuonH-only
    # config defaults to True; expose so AMUSE-as-MuonH controls can match.
    muon_nesterov: bool = False
    backend_steps: int = 5
    muon_epsilon: float = 1e-8
    coefficient_type: CoefficientType = "quintic"
    # Curvature-corrected Muon (0 ⟹ plain MuonH). Riemannian inner-solve on the Muon direction.
    curvature_lambda: float = 0.0
    curvature_beta: float = 0.95
    curvature_inner_steps: int = 10
    curvature_maxbt: int = 10
    curvature_two_sided: bool = True
    curvature_constraint: str = "stiefel"
    curv_power: str = "sqrt"
    # AMUSE hyperparameters (SF interpolation)
    amuse_beta1: float = 0.6
    amuse_rho: float = 0.8
    amuse_warmup_steps: int = 2000
    amuse_weight_lr_power: float = 2.0
    # ScheduleFree+ extensions (arxiv 2605.19095, Defazio FAIR/Meta SI).
    # - ``amuse_beta_schedule``: "amuse" (default, arxiv 2605.22432 ramp) /
    #   "sfplus" (log-linear interpolation in (1-β) from β_initial to β_final
    #   over the run) / "constant" (vanilla SF, β_t = β_1 always).
    # - ``amuse_beta_final``: β at end of training when schedule="sfplus".
    # - ``amuse_c_warmup_steps``: hold c_t = 1 for the first N steps so the
    #   averaged X tracks Z and only starts averaging after weight norms
    #   stabilize. SF+ heuristic: ~2x the LR warmup length.
    # - ``amuse_r_weighting``: exponent r in the per-step weight
    #   w_t = max_lr^p · t^r. SF+ recommends r=1 for long runs.
    amuse_beta_schedule: str = "amuse"
    amuse_beta_final: float = 0.965
    amuse_c_warmup_steps: int = 0
    amuse_r_weighting: float = 0.0
    # Normalization knobs (Z-hyperball is the original AMUSE behavior; Y-hyperball
    # rescales Y_{t+1} per-matrix to match params Frobenius norm). Default = Z-only.
    #   "A" current = z_normalize=True,  y_normalize=False
    #   "B" Y-only  = z_normalize=False, y_normalize=True
    #   "C" both    = z_normalize=True,  y_normalize=True
    #   "D" neither = z_normalize=False, y_normalize=False
    amuse_z_normalize: bool = True
    amuse_y_normalize: bool = False
    # AdamH / Adam hyperparameters. We deviate from the AMUSE paper here: the
    # paper's Adam path drops β_1 (so SF's β_t interpolation is the only
    # smoothing). We keep β_1 instead so the LM head uses the standard SF-AdamH
    # (b1=0.9) the grug pipeline has been tuned for; SF's β_t stacks on top.
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0

    def build(self, num_train_steps):
        # AMUSE removes the LR decay phase; we still respect the user's
        # warmup/lr_schedule from OptimizerConfig (default = warmup + cosine).
        # Recommend setting ``lr_schedule="constant"`` in the launcher.
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            # Per-group base optimizers (no SF averaging — the outer AMUSE
            # wraps these uniformly so there is a single AmuseState with X).
            if self.amuse_z_normalize:
                # Default: Muon + hyperball (Z-norm preserved per matrix).
                muonh_base = scale_with_grug_muonh(
                    momentum=self.muon_momentum,
                    nesterov=self.muon_nesterov,
                    steps=self.backend_steps,
                    muon_eps=self.muon_epsilon,
                    learning_rate=learning_rate,
                    coefficient_type=self.coefficient_type,
                    curvature_lambda=self.curvature_lambda,
                    curvature_beta=self.curvature_beta,
                    inner_steps=self.curvature_inner_steps,
                    riemannian_maxbt=self.curvature_maxbt,
                    two_sided=self.curvature_two_sided,
                    constraint=self.curvature_constraint,
                    curv_power=self.curv_power,
                )
            else:
                # B/D variants: plain Muon (no Z hyperball). LR scales the
                # orthogonalized direction directly. Y-norm (if enabled) is
                # applied AFTER the AMUSE combine.
                # NS5 reshards to P(None, None); without hyperball to restore
                # param sharding, AMUSE's z+du add throws ShardingTypeError on
                # MoE 3-D params. Match update sharding back to params here.
                _muon_kernel = _grug_scale_with_muon(
                    momentum=self.muon_momentum,
                    nesterov=self.muon_nesterov,
                    steps=self.backend_steps,
                    muon_eps=self.muon_epsilon,
                    use_kimi_scaling=False,
                    coefficient_type=self.coefficient_type,
                )
                muonh_base = optax.chain(
                    _muon_kernel,
                    optax.scale(-learning_rate),
                    _scale_by_lr_scale(),
                    _match_named_update_sharding(),
                )

            adamh_base = scale_by_adamh(
                b1=self.beta1,
                b2=self.beta2,
                eps=self.epsilon,
                learning_rate=learning_rate,
            )

            # AdamW is LR-linear, so a uniform output scale by ``lr_scale``
            # reproduces running at adam_lr·lr_scale (keeps the line-searched LR
            # consistent across the matrix and embedding/router/norm groups).
            adamw_base = optax.chain(
                optax.adamw(
                    learning_rate=adam_lr,
                    b1=self.beta1,
                    b2=self.beta2,
                    eps=self.epsilon,
                    weight_decay=self.weight_decay,
                ),
                _scale_by_lr_scale(),
            )

            base = optax.multi_transform(
                {
                    "amuse_muon": muonh_base,
                    "amuse_adamh": adamh_base,
                    "amuse_adamw": adamw_base,
                },
                self.create_mask,
            )

            components = []
            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(
                amuse(
                    base_optimizer=base,
                    learning_rate=learning_rate,
                    beta1=self.amuse_beta1,
                    rho=self.amuse_rho,
                    warmup_steps=self.amuse_warmup_steps,
                    weight_lr_power=self.amuse_weight_lr_power,
                    beta_schedule=self.amuse_beta_schedule,
                    beta_final=self.amuse_beta_final,
                    total_steps=num_train_steps,
                    c_warmup_steps=self.amuse_c_warmup_steps,
                    r_weighting=self.amuse_r_weighting,
                    y_normalize=self.amuse_y_normalize,
                )
            )
            components.append(_match_named_update_sharding())
            return optax.chain(*components)

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            # 1-D / embedding / router / attn_gate → AdamW (decoupled WD) at adam_lr
            if _uses_adamh_baseline_adam_group(path_lower):
                return "amuse_adamw"
            # lm head / output projection → AdamH (hyperball-normalized) at matrix LR
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "amuse_adamh"
            # GatedNorms route to MuonH (NS + Frobenius hyperball), same as the
            # May-arch MuonH baseline (GrugMoeMuonHConfig).
            if "gated_norm" in path_lower:
                return "amuse_muon"
            # Matrix params (incl. ndim-3 expert MLP weights) → MuonH at matrix LR
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "amuse_muon"
            return "amuse_adamw"

        return jax.tree.map(mask_fn, params, paths)


__all__ = [
    "GrugMoeAdamHConfig",
    "GrugMoeAmuseConfig",
    "GrugMoeMuonHConfig",
    "PowerLawTokenDecay",
    "scale_with_grug_muonh",
]
