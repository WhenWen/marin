# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Ported verbatim from the kitchen-sink checkout. The docstrings/comments use
# mathematical Unicode (β, ρ, ×, −, ‖·‖) that is load-bearing for readability;
# silence ruff's ambiguous-character checks for the whole module rather than
# ASCII-fying the math notation.
# ruff: noqa: RUF002, RUF003

"""AMUSE: Anytime Muon with Stable Gradient Evaluation (arxiv 2605.22432).

AMUSE is a time-varying Schedule-Free (SF) wrapper around any base optimizer
(Muon for matrix-valued parameters, AdamH/AdamW for the rest). It keeps
three sequences:

    Y_t = (1 - β_t) Z_t + β_t X_t          # gradient-evaluation point
    Z_{t+1} = Z_t + Δz                     # base sequence (Δz from base opt)
    X_{t+1} = (1 - c_{t+1}) X_t + c_{t+1} Z_{t+1}   # averaged sequence (eval)

Three β_t schedules supported:

  * ``amuse``   (arxiv 2605.22432, default): β_t = β_1 for t ≤ T_0; afterwards
                β_t = 1 - ((T_0 - 1)/(t - 1))^ρ (1 - β_1).
  * ``sfplus``  (arxiv 2605.19095): log-linear interpolation between
                β_initial and β_final over the full ``total_steps`` horizon:
                β_t = 1 - exp((1 - t/T) log(1 - β_initial) + (t/T) log(1 - β_final)).
  * ``constant``: β_t fixed at β_1 (vanilla SF without annealing).

C-warmup and r-weighting (arxiv 2605.19095, Defazio):

  * ``c_warmup_steps``: hold c_t=1 for the first N steps so the averaged X
    tracks Z and only starts averaging after weight norms stabilize. The SF+
    heuristic is c_warmup ≈ 2 × LR warmup.
  * ``r_weighting``: r in the per-step weight w_t = max_lr^p · t^r. Default
    r=0 reproduces optax.contrib.schedule_free. r=1 is recommended for long
    runs.

Convention: the model's stored parameters are kept at Y (the gradient
evaluation point); Z and X both live in optimizer state. The trainer mirrors
``AmuseState.x`` into ``ema_params`` so the existing eval_ema flow uses X.
"""

import math
from collections.abc import Callable
from typing import Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from levanter.optim.grugmuon import (
    _batch_sharded_stack_target_pspec,
    _target_sharding,
    _zeropower_via_newtonschulz_batched_stack_sharded,
    _zeropower_via_newtonschulz_replicated,
)

BetaSchedule = Literal["amuse", "sfplus", "constant"]
MuonDenomType = Literal["fnorm", "nuclear", "hyperball"]

# Adam-natural Polyak denominator constant: E[g²]/E[|g|] = σ·√(π/2) for
# Gaussian g, so Σ_i g²_i/√v_i ≈ √(π/2)·||g||_1 (SF+ arxiv 2605.19095 Eq. 1).
# NOTE: computed via the ``math`` module — do NOT use ``jnp`` at module level
# because that triggers the XLA backend before ``jax.distributed.initialize``.
_ADAM_POLYAK_CONST: float = math.sqrt(math.pi / 2.0)


def _norm_scale_factor(new_leaf: jax.Array, ref_leaf: jax.Array) -> jax.Array:
    """Per-matrix scale factor to make ``new_leaf`` have the same Frobenius norm as ``ref_leaf``.

    Returns a scalar (for ndim<2) or a broadcastable array (for ndim≥2) that
    when multiplied with ``new_leaf`` yields a tensor whose Frobenius norm
    (per-matrix for 2-D, per-expert for 3-D+) matches ``ref_leaf``'s.

    Used by the AMUSE Y-normalize path: when rescaling Y, the SAME factor is
    applied to Z and X so the affine relationship Y = (1−β)Z + βX is preserved
    exactly (otherwise the trainer's stored Y drifts from AMUSE's internal Z/X
    over time and training degenerates).
    """
    if not hasattr(new_leaf, "ndim"):
        return jnp.ones((), dtype=jnp.float32)
    if new_leaf.ndim < 2:
        return jnp.ones((), dtype=jnp.float32)
    if new_leaf.ndim == 2:
        ref_norm = jnp.linalg.norm(ref_leaf)
        cur_norm = jnp.linalg.norm(new_leaf)
        return ref_norm / jnp.maximum(cur_norm, 1e-10)
    # ndim >= 3
    axes = tuple(range(1, new_leaf.ndim))
    ref_norm = jnp.sqrt(jnp.sum(jnp.square(ref_leaf), axis=axes, keepdims=True))
    cur_norm = jnp.sqrt(jnp.sum(jnp.square(new_leaf), axis=axes, keepdims=True))
    return ref_norm / jnp.maximum(cur_norm, 1e-10)


def _apply_norm_scale(leaf: jax.Array, scale: jax.Array) -> jax.Array:
    if not hasattr(leaf, "ndim") or leaf.ndim < 2:
        return leaf
    return leaf * scale


def _muon_polyak_denom_for_leaf_nuclear(g: jax.Array) -> chex.Array:
    """Muon-natural Polyak denominator via the nuclear norm ``||g||_*``.

    Computes ``trace(g^T · UV^T)`` where ``UV^T`` is the NS5-orthogonalized
    gradient. This is the exact "natural inner product with the descent
    direction" for Muon, matching the SF+ paper's choice for Adam
    (``√(π/2)·||g||_1``).

    Sharding plumbing (the reason this had to be retried after the F-norm
    fallback): NS5 reshards arrays to its preferred layout, producing U with
    a sharding incompatible with the MoE 3-D grad layout
    ``('expert', 'data', 'model')``. We mirror Muon's own update path: after
    orthogonalize, reshard U back to ``g.sharding`` via
    ``jax.sharding.reshard`` so the elementwise product broadcasts cleanly.

    For 3-D (MoE) matrices, we use the batch-sharded NS variant when a
    suitable target pspec is available (same as ``_grug_scale_with_muon``);
    fallback to ``vmap`` of replicated NS for non-MoE 3-D cases.
    """
    if g.ndim == 2:
        u = _zeropower_via_newtonschulz_replicated(g, steps=5, coefficient_type="quintic")
    elif g.ndim == 3:
        stack_target_pspec = _batch_sharded_stack_target_pspec(g)
        if stack_target_pspec is not None:
            u = _zeropower_via_newtonschulz_batched_stack_sharded(
                g,
                steps=5,
                coefficient_type="quintic",
                target_pspec=stack_target_pspec,
            )
        else:
            u = jax.vmap(lambda m: _zeropower_via_newtonschulz_replicated(m, steps=5, coefficient_type="quintic"))(g)
    else:
        raise ValueError(f"unexpected ndim={g.ndim} for Muon Polyak denom (expected 2 or 3)")

    # Reshard U back to g's sharding so ``g * u`` broadcasts correctly.
    # Mirrors what ``_match_update_sharding`` does for Muon's own updates.
    # Skip the reshard when no abstract mesh is active (CPU / single-device
    # smoke tests) — ``jax.sharding.reshard`` only accepts NamedSharding /
    # PartitionSpec, not SingleDeviceSharding.
    if not jax.sharding.get_abstract_mesh().empty:
        target_sharding = _target_sharding(g)
        if isinstance(target_sharding, jax.sharding.NamedSharding):
            u = jax.sharding.reshard(u, target_sharding)
    return jnp.sum(g * u)


def _muon_polyak_denom_for_leaf(g: jax.Array) -> chex.Array:
    """Sharding-safe Muon-natural Polyak denominator.

    For a 2-D matrix m×n (m≤n) under Marchenko-Pastur (iid Gaussian entries):
        ||g||_*  ≈  k(c) · √(min(m,n)) · ||g||_F
    with the prefactor k(c) ∈ [0.849, 1.0] depending on aspect ratio
    c = min/max. We approximate as ``√(min) · ||g||_F`` — the k(c) prefactor
    is a global scalar that polyak_warmup absorbs.

    Why this and not ``trace(g^T · UV^T)``? Computing UV^T via NS5 forces a
    reshard pass that's incompatible with MoE 3-D shardings like
    ``('expert', 'data', 'model')``. F-norm is local and sharding-safe.

    For 3-D MoE matrices (E, m, n), each expert slice is treated as an
    independent matrix; the per-expert denominators sum exactly because
    ``Σ_E ||g_E||_*  ≈  Σ_E √min · ||g_E||_F = √min · Σ_E ||g_E||_F``.
    """
    if g.ndim == 2:
        m, n = g.shape
        sqrt_min = math.sqrt(float(min(m, n)))
        # ||g||_F = √Σ(g²); use einsum / sum which keeps sharding sensible.
        return sqrt_min * jnp.sqrt(jnp.sum(jnp.square(g)))
    if g.ndim == 3:
        _e, m, n = g.shape
        sqrt_min = math.sqrt(float(min(m, n)))
        # F-norm per expert, then sum (matches Σ_E ||g_E||_*).
        per_expert_fnorm = jnp.sqrt(jnp.sum(jnp.square(g), axis=(-2, -1)))
        return sqrt_min * jnp.sum(per_expert_fnorm)
    raise ValueError(f"unexpected ndim={g.ndim} for Muon Polyak denom (expected 2 or 3)")


def _hyperball_polyak_rate_for_leaf(x: jax.Array, g: jax.Array) -> chex.Array:
    """Per-matrix hyperball Polyak descent rate ``Σ_i ‖X_i‖_F · ‖G_i‖_F``.

    Derivation (see LOG / docstring of ``amuse_polyak``): a hyperball update steps
    a fixed magnitude ``γ·‖X‖`` along the UNIT direction ``û = u/‖u‖`` (it
    decouples the step norm from the update norm ‖u‖ and renormalizes onto the
    ‖X‖-sphere). The first-order predicted loss change is
    ``Δf ≈ −γ·‖X‖·⟨G, û⟩ = −γ·‖X‖·⟨G,U⟩/‖U‖``, so the Polyak denominator (descent
    rate per unit γ) is ``D = ‖X‖·⟨G,U⟩/‖U‖``. For Muon ``U=polar(G)``: ``⟨G,U⟩=‖G‖_*``,
    ``‖U‖=√min(m,n)`` ⇒ ``D = ‖X‖·‖G‖_*/√min ≈[Marchenko–Pastur ‖G‖_*≈√min·‖G‖_F]
    ‖X‖_F·‖G‖_F``. Computed PER MATRIX (the hyperball preserves norm per 2-D slice /
    per expert), summed over the leading axis for 3-D MoE leaves.
    """
    if not hasattr(x, "ndim") or x.ndim < 2:
        # 1-D (norms/biases) are not on a hyperball path; fall back to the rate
        # of an un-normalized step (Adam-natural handled by the caller).
        return jnp.sqrt(jnp.sum(jnp.square(x))) * jnp.sqrt(jnp.sum(jnp.square(g)))
    axes = tuple(range(1, x.ndim)) if x.ndim >= 3 else None
    if axes is None:  # 2-D: single matrix
        return jnp.sqrt(jnp.sum(jnp.square(x))) * jnp.sqrt(jnp.sum(jnp.square(g)))
    # 3-D MoE (E, m, n): per-expert ‖X_e‖_F · ‖G_e‖_F, summed over experts.
    x_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axes))
    g_norm = jnp.sqrt(jnp.sum(jnp.square(g), axis=axes))
    return jnp.sum(x_norm * g_norm)


def make_polyak_denom_fn(
    mask_fn: Callable[[optax.Params], object],
    *,
    muon_label: str,
    muon_denom_type: MuonDenomType = "nuclear",
    adam_const: float = _ADAM_POLYAK_CONST,
    hyperball_labels: tuple[str, ...] = (),
    # Kept for API compatibility.
    muon_ns_steps: int = 5,
    muon_coefficient_type: str = "quintic",
) -> Callable[[optax.Updates, optax.Params], chex.Array]:
    """Build the multi-group Polyak denominator function.

    For each leaf:
      * ``mask == muon_label`` (matrix Muon path):
          - ``"nuclear"`` (default): ``trace(g · UV^T)`` via NS5, with U
            resharded to ``g.sharding`` to keep MoE shardings happy.
          - ``"fnorm"``: ``√(min(m,n)) · ||g||_F`` approximation (Marchenko-
            Pastur). Sharding-safe, no NS5 reshard, ~15% lower fidelity.
      * other leaves (Adam path): use ``√(π/2)·||g||_1`` (paper Eq. 1).

    Returns a callable ``denom_fn(grads, params) -> scalar`` summing the
    per-leaf denominators into a single Polyak denominator.
    """
    del muon_ns_steps, muon_coefficient_type  # implicit in nuclear-norm path.

    hyperball = muon_denom_type == "hyperball"
    if muon_denom_type == "nuclear":
        muon_leaf_fn = _muon_polyak_denom_for_leaf_nuclear
    elif muon_denom_type == "fnorm":
        muon_leaf_fn = _muon_polyak_denom_for_leaf
    elif hyperball:
        muon_leaf_fn = None  # handled per-leaf with params below
    else:
        raise ValueError(f"unknown muon_denom_type: {muon_denom_type!r}")
    # In "hyperball" mode the norm-preserving groups (Muon + AdamH) use the
    # ‖X‖·‖G‖ descent-rate denominator (derived above); only the un-normalized
    # AdamW group keeps the Adam-natural √(π/2)·‖g‖₁. Default to {muon_label}.
    hb_set = set(hyperball_labels) if hyperball_labels else {muon_label}

    def denom_fn(grads: optax.Updates, params: optax.Params) -> chex.Array:
        mask = mask_fn(params)
        g_leaves = jax.tree.leaves(grads)
        m_leaves = jax.tree.leaves(mask)
        x_leaves = jax.tree.leaves(params)
        if len(g_leaves) != len(m_leaves):
            raise ValueError(f"polyak denom: grad/mask leaf count mismatch ({len(g_leaves)} vs {len(m_leaves)})")
        contributions = []
        for g, m, x in zip(g_leaves, m_leaves, x_leaves, strict=False):
            if hyperball and m in hb_set:
                contributions.append(_hyperball_polyak_rate_for_leaf(x, g))
            elif (not hyperball) and m == muon_label:
                contributions.append(muon_leaf_fn(g))
            else:
                contributions.append(adam_const * jnp.sum(jnp.abs(g)))
        # sum() on an empty list returns int 0; cast for safety.
        return jnp.asarray(sum(contributions, jnp.float32(0.0)), dtype=jnp.float32)

    return denom_fn


class AmuseState(NamedTuple):
    """State for the AMUSE wrapper.

    z: base sequence (same pytree structure as params).
    x: averaged sequence (same pytree structure as params). Used as the eval
       model — populated each step so the trainer can read it directly
       without needing to walk opt_state and recompute.
    base_state: state of the wrapped base optimizer (e.g., Muon momentum, Adam V).
    weight_sum: running Σ η_i^weight_lr_power, for the c_t averaging coefficient.
    max_lr: max learning rate seen so far (matches optax.contrib.schedule_free).
    step: step counter (1-indexed, advanced before computing β_t / c_t).
    """

    z: optax.Params
    x: optax.Params
    base_state: optax.OptState
    weight_sum: chex.Array
    max_lr: chex.Array
    step: chex.Array


def _beta_t(
    step: chex.Array,
    *,
    schedule: BetaSchedule,
    beta1: float,
    rho: float,
    warmup_steps: int,
    beta_final: float,
    total_steps: int,
) -> chex.Array:
    """β_t schedule.

    ``amuse``: β_t = β_1 for t ≤ T_0; else 1 - ((T_0-1)/(t-1))^ρ (1 - β_1).
    ``sfplus``: log-linear interp of (1-β) from β_initial=β_1 to β_final
        over total_steps. Equivalent to β_t = 1 - (1-β_1)^(1-u)(1-β_f)^u
        with u = t / total_steps.
    ``constant``: β_t = β_1 for all t.
    """
    t = step.astype(jnp.float32)
    b1 = jnp.asarray(float(beta1), dtype=jnp.float32)
    if schedule == "constant":
        return b1
    if schedule == "amuse":
        t0 = jnp.asarray(float(warmup_steps), dtype=jnp.float32)
        denom = jnp.maximum(t - 1.0, 1.0)
        ratio = (t0 - 1.0) / denom
        decay = jnp.power(ratio, jnp.asarray(float(rho), dtype=jnp.float32))
        beta_post = 1.0 - decay * (1.0 - b1)
        return jnp.where(t <= t0, b1, beta_post)
    if schedule == "sfplus":
        if total_steps <= 0:
            raise ValueError("sfplus schedule needs total_steps > 0")
        T = jnp.asarray(float(total_steps), dtype=jnp.float32)
        u = jnp.clip(t / T, 0.0, 1.0)
        log1mbi = jnp.log1p(-float(beta1))
        log1mbf = jnp.log1p(-float(beta_final))
        log1mb = (1.0 - u) * log1mbi + u * log1mbf
        return 1.0 - jnp.exp(log1mb)
    raise ValueError(f"unknown beta schedule: {schedule}")


def amuse(
    base_optimizer: optax.GradientTransformation,
    learning_rate: optax.ScalarOrSchedule,
    beta1: float = 0.6,
    rho: float = 0.8,
    warmup_steps: int = 2000,
    weight_lr_power: float = 2.0,
    *,
    beta_schedule: BetaSchedule = "amuse",
    beta_final: float = 0.965,
    total_steps: int = 0,
    c_warmup_steps: int = 0,
    r_weighting: float = 0.0,
    y_normalize: bool = False,
) -> optax.GradientTransformation:
    """AMUSE wrapper.

    The model's parameters represent Y_t (the gradient-evaluation point); grads
    received by ``update_fn`` are ∇L(Y_t). The base_optimizer is expected to
    return updates Δz that include the ``-η_t`` scaling (and any weight decay
    on Z if desired). AMUSE itself does only the Y/Z/X averaging — weight
    decay, gradient clipping, and base-optimizer specifics live in the wrapped
    base.

    Args:
        base_optimizer: Any optax transform that maps grad → Δz, where applying
            Δz to Z yields the new base sequence Z_{t+1}. AMUSE passes
            ``params=state.z`` so the base optimizer's hyperball / weight decay
            sees Z (the right reference frame for the SF Z-update).
            The base should NOT have its own SF-style β_1 momentum (AMUSE
            replaces that with the time-varying β_t interpolation). Muon's μ
            momentum is fine — it operates pre-orthogonalization, distinct
            from SF.
        learning_rate: schedule callable; used for the c_{t+1} averaging
            coefficient (η_t² / Σ η_i²).
        beta1: initial SF interpolation coefficient. Defaults to 0.6 per the
            paper's d512 LLM recipe (β_1 ∈ {0.4, 0.6}).
        rho: rate of β_t growth toward 1. Defaults to 0.8 per the paper
            (ρ ∈ {0.6, 0.8}). ρ=0 reduces AMUSE to fixed-β SF.
        warmup_steps: T_0. β_t = β_1 below this, schedule kicks in after.
        weight_lr_power: exponent ``p`` on η in the per-step weight
            w_t = max_lr^p · t^r. Defaults to 2.0, matching
            optax.contrib.schedule_free / Defazio 2024.
        beta_schedule: which β_t schedule to use. Default ``"amuse"``.
        beta_final: final β for ``"sfplus"`` log-linear annealing. Paper
            default is 0.965 (Defazio, arxiv 2605.19095, fig. 1).
        total_steps: required when ``beta_schedule="sfplus"``; the run length
            over which to interpolate (1-β) log-linearly.
        c_warmup_steps: hold c_t = 1 for the first N steps (X tracks Z; no
            averaging). The SF+ heuristic is ~2× the LR warmup. Default 0
            (off; matches optax.contrib.schedule_free).
        r_weighting: exponent ``r`` in w_t = max_lr^p · t^r. Default 0.0
            (matches optax). SF+ recommends r=1 for long runs to bias the
            average toward later iterates (w_t = t · max_lr^p).
    """

    def _lr(step: chex.Array) -> chex.Array:
        # learning_rate may arrive as a Python scalar, a schedule callable, or
        # a JAX-traced array (the standard inject_hyperparams pathway). Don't
        # try to coerce to Python ``float`` — that fails on tracers — just
        # convert to a fp32 array. Callable schedules get evaluated at ``step``.
        lr = learning_rate(step) if callable(learning_rate) else learning_rate
        return jnp.asarray(lr, dtype=jnp.float32)

    def init_fn(params: optax.Params) -> AmuseState:
        z = jax.tree.map(lambda t: t.copy(), params)
        x = jax.tree.map(lambda t: t.copy(), params)
        return AmuseState(
            z=z,
            x=x,
            base_state=base_optimizer.init(params),
            weight_sum=jnp.zeros([], jnp.float32),
            max_lr=jnp.zeros([], jnp.float32),
            step=jnp.zeros([], jnp.int32),
        )

    def update_fn(grads, state: AmuseState, params=None, *, lr_scale=1.0, z_extra_delta=None, **_extra):
        # ``_extra`` (e.g., value=loss) is silently dropped — vanilla AMUSE uses
        # a fixed schedule for the LR, not the loss-based Polyak step. Accepting
        # it here lets the trainer pass the same kwargs to every optimizer.
        # ``lr_scale`` (default 1.0) is forwarded INTO the base optimizer so the
        # effective LR is scaled where it enters each group (inside the MuonH /
        # AdamH hyperball, where the step is nonlinear in the LR; as an output
        # scale on the linear AdamW group). An external per-batch line search
        # uses this to set the effective base LR per window — scaling the output
        # Δz here instead would be wrong for the (nonlinear) hyperball groups.
        if params is None:
            raise ValueError("AMUSE requires params (=Y_t) to compute the Y→Y update")

        # params here is Y_t. state.x is X_t (maintained across steps); we
        # don't need to re-derive it. Advance step / β_t / X recursion below.
        next_step = state.step + 1

        # Base optimizer step: Δz = base_optimizer(grad, base_state, params=z).
        # Passing params=z so any param-dependent piece of the base (hyperball,
        # weight-decay scaling, etc.) operates on the base sequence Z.
        base_update, new_base_state = base_optimizer.update(grads, state.base_state, params=state.z, lr_scale=lr_scale)
        new_z = jax.tree.map(lambda z, du: z + du, state.z, base_update)
        # Optional externally-supplied Z increment applied WITHOUT lr_scale (used
        # to fold the router-bias QB feedback into Z so it flows through the same
        # X/Y averaging+interpolation as the gradient-trained params). Zero on all
        # leaves except the router biases.
        if z_extra_delta is not None:
            new_z = jax.tree.map(lambda nz, ed: nz + ed, new_z, z_extra_delta)

        # LR-weighted online average (matches optax.contrib.schedule_free): the
        # per-step weight is w_t = max_lr^p · t^r, normalized by Σ over the
        # trajectory. SF+ adds the t^r factor (r=1 biases toward later iterates).
        # ``lr_scale`` enters here too: the weight is NONLINEAR (w ∝ lr^p, p=2) in
        # the LR, so the per-batch line-searched LR must scale the effective lr
        # used for the average — not just the Δz step — or the X-average weights
        # each window by the wrong lr².
        lr_t = lr_scale * _lr(next_step)
        new_max_lr = jnp.maximum(state.max_lr, lr_t)
        step_f = next_step.astype(jnp.float32)
        weight = (new_max_lr ** float(weight_lr_power)) * jnp.power(
            step_f, jnp.asarray(float(r_weighting), dtype=jnp.float32)
        )
        # c-warmup: hold c_t = 1 for the first ``c_warmup_steps`` (X tracks Z).
        # During c-warmup we do NOT accumulate weight_sum, so averaging effectively
        # restarts at step c_warmup_steps + 1 (the SF+ recipe in arxiv 2605.19095).
        in_c_warmup = next_step <= jnp.asarray(int(c_warmup_steps), dtype=next_step.dtype)
        new_weight_sum = jnp.where(in_c_warmup, state.weight_sum, state.weight_sum + weight)
        denom = jnp.where(new_weight_sum > 0.0, new_weight_sum, jnp.float32(1.0))
        c_post = weight / denom
        c_next = jnp.where(in_c_warmup, jnp.float32(1.0), c_post)
        new_x = jax.tree.map(lambda x, z: (1.0 - c_next) * x + c_next * z, state.x, new_z)

        # New Y_{t+1} = (1 - β_{t+1}) Z_{t+1} + β_{t+1} X_{t+1}.
        beta_next = _beta_t(
            next_step,
            schedule=beta_schedule,
            beta1=beta1,
            rho=rho,
            warmup_steps=warmup_steps,
            beta_final=beta_final,
            total_steps=total_steps,
        )
        new_y = jax.tree.map(lambda z, x: (1.0 - beta_next) * z + beta_next * x, new_z, new_x)

        # Optional Y-hyperball: compute the per-matrix scale factor that makes
        # ||new_y|| = ||params|| and apply it uniformly to Y, Z, AND X. Applying
        # only to Y would break Y = (1−β)Z + βX over time (trainer's Y drifts
        # from AMUSE's internal bookkeeping).
        if y_normalize:
            scales = jax.tree.map(_norm_scale_factor, new_y, params)
            new_y = jax.tree.map(_apply_norm_scale, new_y, scales)
            new_z = jax.tree.map(_apply_norm_scale, new_z, scales)
            new_x = jax.tree.map(_apply_norm_scale, new_x, scales)

        # Return update = new_Y - Y_t so optax.apply_updates(Y_t, update) = Y_{t+1}.
        update = jax.tree.map(lambda y_new, y_old: y_new - y_old, new_y, params)
        new_state = AmuseState(
            z=new_z,
            x=new_x,
            base_state=new_base_state,
            weight_sum=new_weight_sum,
            max_lr=new_max_lr,
            step=next_step,
        )
        return update, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


class AmusePolyakState(NamedTuple):
    """State for the AMUSE + Polyak step-size wrapper.

    Same fields as ``AmuseState`` except ``max_lr`` becomes ``max_gamma`` (the
    Polyak step is now the LR) and we add ``last_loss`` to stash f(y_t) for
    use as the stale f(z_t) proxy on the next step.
    """

    z: optax.Params
    x: optax.Params
    base_state: optax.OptState
    weight_sum: chex.Array
    max_gamma: chex.Array
    last_loss: chex.Array  # f(y_{t-1}), stale proxy for f(z_t)
    step: chex.Array
    last_gamma: chex.Array  # for logging


def amuse_polyak(
    base_optimizer: optax.GradientTransformation,
    polyak_denom_fn: Callable[[optax.Updates, optax.Params], chex.Array],
    *,
    f_star: float = 0.0,
    polyak_warmup_steps: int = 400,
    polyak_eps: float = 1e-12,
    polyak_gamma_max: float | None = None,
    # SF interpolation (same as ``amuse``).
    beta1: float = 0.6,
    rho: float = 0.8,
    warmup_steps: int = 2000,
    weight_lr_power: float = 2.0,
    beta_schedule: BetaSchedule = "amuse",
    beta_final: float = 0.965,
    total_steps: int = 0,
    c_warmup_steps: int = 0,
    r_weighting: float = 0.0,
    y_normalize: bool = False,
) -> optax.GradientTransformationExtraArgs:
    """AMUSE + Polyak step size (SF+ arxiv 2605.19095 Eq. 1).

    Polyak step size replaces the externally-supplied ``learning_rate``:

        γ_t = (f(z_t) − f_* + β_t · ⟨∇f(y_t), z_t − x_t⟩) / D_t

    where the denominator ``D_t`` is the sum of per-group natural Polyak
    denominators (nuclear norm for Muon-grouped leaves, √(π/2)·||g||_1 for
    Adam-grouped leaves). See ``make_polyak_denom_fn``.

    Conventions:
      * The wrapped ``base_optimizer`` must produce the *unit* descent
        direction at the leaf scale we want γ_t to multiply — i.e., its
        internal LR should be 1.0. AMUSE rescales by γ_t.
      * ``f(z_t)`` is approximated by the prior step's ``f(y_{t-1})`` (stored
        in state). One-step lag, no extra forward.
      * Polyak warmup: γ_t is multiplied by ``min(1, t / polyak_warmup_steps)``
        (SF+ §326 figure 8 recipe; multiplicative LR warmup).

    Args:
        base_optimizer: same as in ``amuse`` but configured with internal LR=1.
        polyak_denom_fn: see ``make_polyak_denom_fn``.
        f_star: estimate of optimal loss. 0.0 is fine for LLM cross-entropy.
        polyak_warmup_steps: linear γ ramp from 0 to 1 over this many steps.
        polyak_eps: numerical guard in the γ denominator.
        polyak_gamma_max: optional cap on γ to prevent early-step blowup.
        Other args: same semantics as ``amuse``.
    """

    def _gamma_t(loss_proxy, grads, params, z, x, step):
        beta_t_now = _beta_t(
            step,
            schedule=beta_schedule,
            beta1=beta1,
            rho=rho,
            warmup_steps=warmup_steps,
            beta_final=beta_final,
            total_steps=total_steps,
        )
        # SF correction term: β_t · ⟨g, z − x⟩, summed across all leaves.
        zx = jax.tree.map(lambda z_, x_: z_ - x_, z, x)
        per_leaf = jax.tree.map(lambda g, d: jnp.sum(g * d), grads, zx)
        sf_correction = beta_t_now * jnp.asarray(sum(jax.tree.leaves(per_leaf), jnp.float32(0.0)), dtype=jnp.float32)
        numerator = jnp.maximum(loss_proxy - jnp.float32(f_star) + sf_correction, jnp.float32(0.0))
        denom = polyak_denom_fn(grads, params)
        gamma_raw = numerator / (denom + jnp.float32(polyak_eps))
        # Polyak warmup (multiplicative ramp).
        warm = jnp.minimum(
            step.astype(jnp.float32) / jnp.maximum(jnp.float32(polyak_warmup_steps), 1.0),
            jnp.float32(1.0),
        )
        gamma_t = gamma_raw * warm
        if polyak_gamma_max is not None:
            gamma_t = jnp.minimum(gamma_t, jnp.float32(polyak_gamma_max))
        return gamma_t, beta_t_now

    def init_fn(params: optax.Params) -> AmusePolyakState:
        z = jax.tree.map(lambda t: t.copy(), params)
        x = jax.tree.map(lambda t: t.copy(), params)
        return AmusePolyakState(
            z=z,
            x=x,
            base_state=base_optimizer.init(params),
            weight_sum=jnp.zeros([], jnp.float32),
            max_gamma=jnp.zeros([], jnp.float32),
            last_loss=jnp.zeros([], jnp.float32),
            step=jnp.zeros([], jnp.int32),
            last_gamma=jnp.zeros([], jnp.float32),
        )

    def update_fn(grads, state: AmusePolyakState, params=None, *, value=None, **_extra):
        if params is None:
            raise ValueError("amuse_polyak requires params (=Y_t)")
        if value is None:
            raise ValueError(
                "amuse_polyak requires `value=loss` (=f(Y_t)) via optax extra_args; "
                "patch the trainer to pass it through `optimizer.update`."
            )

        next_step = state.step + 1
        # Use the prior step's loss as the stale f(z_t) proxy. On the first
        # step there is no prior loss → fall back to the current loss
        # (slightly biased, but only matters for one step).
        is_first = next_step == 1
        loss_proxy = jnp.where(is_first, value.astype(jnp.float32), state.last_loss)

        # Compute γ_t (Polyak) and β_t.
        gamma_t, _beta_now = _gamma_t(
            loss_proxy=loss_proxy,
            grads=grads,
            params=params,
            z=state.z,
            x=state.x,
            step=next_step,
        )

        # Apply γ_t as the actual learning rate by threading it as ``lr_scale``
        # into the base (built at internal LR=1, so effective LR = γ_t). This is
        # CRITICAL for the hyperball groups: their step is NONLINEAR in the LR
        # (norm-preserving angular projection), so linearly scaling a lr=1 output
        # (γ_t · base_direction) is wrong — lr=1 is a ~90° rotation and γ·that ≠
        # the hyperball step at lr=γ_t. lr_scale recomputes the angular step at the
        # true γ_t (MuonH/AdamH consume it inside the hyperball; AdamW is linear).
        base_update, new_base_state = base_optimizer.update(grads, state.base_state, params=state.z, lr_scale=gamma_t)
        new_z = jax.tree.map(lambda z, du: z + du, state.z, base_update)

        # X averaging — same as ``amuse``, but ``max_gamma`` replaces ``max_lr``.
        new_max_gamma = jnp.maximum(state.max_gamma, gamma_t)
        step_f = next_step.astype(jnp.float32)
        weight = (new_max_gamma ** float(weight_lr_power)) * jnp.power(
            step_f, jnp.asarray(float(r_weighting), dtype=jnp.float32)
        )
        in_c_warmup = next_step <= jnp.asarray(int(c_warmup_steps), dtype=next_step.dtype)
        new_weight_sum = jnp.where(in_c_warmup, state.weight_sum, state.weight_sum + weight)
        denom_w = jnp.where(new_weight_sum > 0.0, new_weight_sum, jnp.float32(1.0))
        c_post = weight / denom_w
        c_next = jnp.where(in_c_warmup, jnp.float32(1.0), c_post)
        new_x = jax.tree.map(lambda x, z: (1.0 - c_next) * x + c_next * z, state.x, new_z)

        # New Y_{t+1} = (1 − β_{t+1}) Z + β_{t+1} X.
        beta_next = _beta_t(
            next_step,
            schedule=beta_schedule,
            beta1=beta1,
            rho=rho,
            warmup_steps=warmup_steps,
            beta_final=beta_final,
            total_steps=total_steps,
        )
        new_y = jax.tree.map(lambda z, x: (1.0 - beta_next) * z + beta_next * x, new_z, new_x)
        if y_normalize:
            scales = jax.tree.map(_norm_scale_factor, new_y, params)
            new_y = jax.tree.map(_apply_norm_scale, new_y, scales)
            new_z = jax.tree.map(_apply_norm_scale, new_z, scales)
            new_x = jax.tree.map(_apply_norm_scale, new_x, scales)
        update = jax.tree.map(lambda y_new, y_old: y_new - y_old, new_y, params)

        new_state = AmusePolyakState(
            z=new_z,
            x=new_x,
            base_state=new_base_state,
            weight_sum=new_weight_sum,
            max_gamma=new_max_gamma,
            last_loss=value.astype(jnp.float32),
            step=next_step,
            last_gamma=gamma_t,
        )
        return update, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def find_amuse_state(opt_state) -> "AmuseState | AmusePolyakState | None":
    """Walk ``opt_state`` and return the first AMUSE-flavored state, or None.

    Matches both ``AmuseState`` and ``AmusePolyakState`` so the grug trainer
    can mirror ``state.x`` into ``ema_params`` regardless of variant.
    """
    matches = []

    def _is_amuse(x):
        if isinstance(x, (AmuseState, AmusePolyakState)):
            matches.append(x)
            return True
        return False

    jax.tree_util.tree_flatten(opt_state, is_leaf=_is_amuse)
    return matches[0] if matches else None


__all__ = [
    "AmusePolyakState",
    "AmuseState",
    "amuse",
    "amuse_polyak",
    "find_amuse_state",
    "make_polyak_denom_fn",
]
