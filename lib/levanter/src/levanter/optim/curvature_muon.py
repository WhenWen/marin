# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Curvature-corrected Muon: Nesterov Muon with a one-sided Shampoo curvature penalty, on the
hyperball (MuonH) base.

Per matrix ``W`` (oriented so rows ≥ cols), the orthogonal update direction approximately solves

    max_{XᵀX=I}  ⟨N_t, X⟩  −  (λ/2)·tr(Xᵀ P_t X),     P_t = EMA(G_t G_tᵀ)   (left/output curvature)

via the inner Newton–Schulz fixed point

    X⁽⁰⁾   = msign(N_t)
    X⁽ᵏ⁺¹⁾ = msign( N_t + λ·(√e_max·I − P_t/√e_max)·X⁽ᵏ⁾ ),   e_max = λ_max(P_t)  (power iteration)

``msign`` is the usual Muon Newton–Schulz orthogonalization. The operator ``√e_max·I − P_t/√e_max`` is
**PSD** (eigenvalues ``(e_max − p_i)/√e_max ≥ 0``: ``√e_max`` in flat directions, ``0`` in the top-curvature
one), so the fixed point is a contraction — **stable at any λ, no α shift needed**. And ``P/√e_max`` has
gradient units (``P ~ G²`` ⟹ ``P/√e_max ~ G``), matching ``N``, so **λ is dimensionless** (no ``‖N‖``
factor) and its LR coupling is unambiguous. Curvature enters only through matmuls + a power iteration —
no eigendecomposition. ``X_t`` is then mapped through the MuonH hyperball (scale-invariant,
constant-Frobenius-norm) reparam instead of ``√(Out/In)`` scaling.

λ = 0 ⟹ exactly MuonH (Nesterov Muon + hyperball). K = 1 is a single curvature-corrected polishing step.
"""

import dataclasses
import functools
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.adamh import scale_by_adamh
from levanter.optim.config import OptimizerConfig
from levanter.optim.util import (
    NEWTON_SCHULZ_COEFFICIENTS,
    CoefficientType,
    flatten_linear_layers,
    label_linear_like_module,
    unflatten_linear_layers,
    zeropower_via_newtonschulz5,
)
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("curvature_muon")
@dataclass(frozen=True)
class CurvatureMuonConfig(OptimizerConfig):
    """Curvature-corrected Muon on the hyperball base (cf. MuonH). λ=0 recovers MuonH exactly."""

    adam_lr: float = 6e-4
    momentum: float = 0.95  # μ
    nesterov: bool = True
    backend_steps: int = 5  # Newton-Schulz steps for msign
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    coefficient_type: CoefficientType = "quintic"
    # --- curvature knobs ---
    # Inner fixed point: X = msign( N + λ·(α·√e_max·I − C)·X ), P = EMA(G Gᵀ), e_max = λ_max(P), with the
    # curvature term C = P/√e_max (curv_power="linear", penalty ∝ p_i) or P^{1/2} (curv_power="sqrt", ∝ √p_i).
    # C has max eigenvalue √e_max, so α·√e_max·I − C is PSD for α ≥ 1 (α=1 zeroes the top-curvature
    # direction; α>1 leaves a strictly-positive floor — gentler). C has gradient units (P~G² ⟹ C~G),
    # matching N, so λ is dimensionless and its LR coupling is clean (no ‖N‖ factor). λ=0 ⟹ MuonH.
    curvature_beta: float = 0.95  # ρ, EMA decay of P = EMA(G Gᵀ)
    curvature_lambda: float = 0.0  # λ, curvature strength (0 ⟹ MuonH)
    curvature_alpha: float = 1.0  # α ≥ 1, multiplier on the √e_max shift (α=1 = boundary PSD)
    curv_power: str = "sqrt"  # "sqrt" (C = P^{1/2}, default) or "linear" (C = P/√e_max); ignored if two_sided
    two_sided: bool = (
        False  # two-sided curvature P_L^{1/4} X P_R^{1/4} + Shampoo warm-start msign(P_L^{-1/4} N P_R^{-1/4})
    )
    mudam_init: bool = True  # warm-start X⁰ = msign(P^{-1/2} N) (Mudam direction, coupled-NS q_k); default on
    mudam_steps: int = 5  # coupled-NS steps for the Mudam warm-start (kept small — stable only under-converged)
    inner_steps: int = 1  # K, inner iterations (fixed-point or Riemannian-Muon ascent)
    # Inner solver for the Stiefel subproblem max_{XᵀX=I} ⟨N,X⟩ − (λ/2)tr(XᵀCX):
    #   "fixed_point"     — Newton-Schulz fixed point X ← msign(N + λ(α√e_max·I − C)X) (default; K-sensitive)
    #   "riemannian_muon" — Riemannian-gradient ascent, msign-orthogonalized direction, Armijo backtracking
    #                       line search (robust across λ, K-stable). Optional warm start across outer steps.
    inner_solver: str = "fixed_point"
    riemannian_maxbt: int = 10  # backtracking steps for the Riemannian line search (τ ∈ {0.5·0.5ʲ}_{j<maxbt})
    # Carry the inner X across outer steps as the warm start. Toy regret sims show msign(N_t) — the
    # fully-current λ=0 optimum — is a *better* start than the stale carried X under realistic momentum
    # smoothing (μ=0.95), so warm-start does not help; default off (cold msign(N_t) restart each step).
    riemannian_warm_start: bool = False
    # Feasible set for the riemannian solver: "stiefel" (XᵀX=I, msign retraction) or "ball" (XᵀX⪯I, mclip
    # SVD-truncation projection — lets singular values shrink below 1 in high-curvature directions).
    constraint: str = "stiefel"
    # If > 0, jax.debug.print the summed inner objective φ vs inner-step k every `log_phi_every` train steps
    # (diagnostic for whether the inner solve saturates at the chosen K). 0 = off.
    log_phi_every: int = 0
    power_iters: int = 8  # power-iteration steps for e_max(P) (warm-started) + Newton-Schulz steps for P^{1/2}
    # Hyperball denominator: "actual" divides the direction by ‖u_t‖ (the realized norm — normalizes magnitude
    # away, so the ball's σ<1 shrinkage has no effect on step size). "isometry" divides by the THEORETICAL norm
    # at XᵀX=I, i.e. √min(out,in); identical to "actual" for σ=1 (Stiefel) so the LR stays calibrated, but for
    # the ball (σ<1) the step scales by ‖u‖/√N < 1 → curvature-damped (the σ-shrinkage now reduces the step).
    hyperball_denom: str = "actual"
    # If set, the curvature strength tracks the LR schedule: lambda_t = curvature_lambda * lr_t / peak_lr.
    # So curvature is strongest at peak LR and fades during warmup / cosine decay (curvature_lambda = peak).
    lambda_tracks_lr: bool = False

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def curvature_muon_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_curvature_muon(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        learning_rate,
                        self.coefficient_type,
                        self.curvature_beta,
                        self.curvature_lambda,
                        self.curvature_alpha,
                        self.curv_power,
                        self.two_sided,
                        self.mudam_init,
                        self.mudam_steps,
                        self.inner_steps,
                        self.power_iters,
                        self.learning_rate,
                        self.lambda_tracks_lr,
                        self.inner_solver,
                        self.riemannian_maxbt,
                        self.riemannian_warm_start,
                        self.constraint,
                        self.log_phi_every,
                        self.hyperball_denom,
                    )
                )
                return optax.chain(*components)

            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "curvature_muon": curvature_muon_transform(),
                "adamh": adamh_transform(),
                "adam": adam_transform(),
            }
            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        """Embeddings → adam, lm_head → adamh, Linear weights → curvature_muon, else → adam (matches MuonH)."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str:
                return "adam"
            elif "lm_head" in path_str:
                return "adamh"
            elif isinstance(param, haliax.nn.Linear):
                return label_linear_like_module(param, weight_label="curvature_muon", bias_label="adam")
            else:
                return "adam"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))


class ScaleByCurvatureMuonState(NamedTuple):
    momentum_buffer: optax.Updates  # B, full param tree
    curvature: optax.Updates  # P_L (left Gram), flattened-linear tree of [..., M, M] (M = max(out, in))
    power_vec: optax.Updates  # q_L, flattened-linear tree of [..., M]
    curvature_r: optax.Updates  # P_R (right Gram), flattened-linear tree of [..., N, N] (N = min(out, in))
    power_vec_r: optax.Updates  # q_R, flattened-linear tree of [..., N]
    inner_x: optax.Updates  # carried inner solution X (oriented [..., M, N]); warm start for riemannian_muon
    count: jax.Array  # scalar step counter (for the φ-vs-K diagnostic print cadence)


_EMAX_MARGIN = 1.05  # inflate the (lower-bound) Rayleigh e_max estimate so the operator stays strictly PSD
_POW4_FLOOR = 1e-2  # spectrum floor for the inverse-4th-root NS (two-sided warm start) so it stays bounded


def _matrix_sqrt_ns(a, iters):
    """A^{1/2} and A^{-1/2} for SPD A with spectrum in (0, 1], via the coupled Newton-Schulz (Denman–Beavers).

    Returns (Y, Z), Y → A^{1/2}, Z → A^{-1/2}. Matmul-only, no eigh. Stable as long as A's spectrum is
    floored away from 0 (else Z blows up). Z is used for the optional Mudam-style P^{-1/2} N warm start.
    """
    n = a.shape[0]
    eye = jnp.eye(n, dtype=a.dtype)
    y = a
    z = eye
    for _ in range(int(iters)):
        t = 1.5 * eye - 0.5 * (z @ y)
        y = y @ t
        z = t @ z
    return y, z


def _qr_refine(sq, spec=None):
    """Q = qr(S·Q_prev)[0] — ONE orthogonal-iteration step refining the MAINTAINED eigenbasis across outer
    steps (the paper's Step 3b, amortized: one QR/step, Q tracks the slowly-changing Gram). ``sq`` is the
    already-formed S·Q_prev. TPU-safe: ``jnp.linalg.qr``/``eigh`` trip an internal ``select`` under the
    explicit mesh, so the batched (stack-sharded) case runs per-matrix QR inside a manual-mode ``shard_map``
    over the stack axis — each device QRs its LOCAL matrices as plain arrays, fully distributed, NO reshard.
    """
    if sq.ndim < 3 or spec is None or jax.sharding.get_abstract_mesh().empty:
        return jnp.linalg.qr(sq)[0]
    return jax.shard_map(
        lambda a: jnp.linalg.qr(a)[0],
        mesh=jax.sharding.get_abstract_mesh(),
        in_specs=spec,
        out_specs=spec,
        check_vma=False,
    )(sq)


def _pow4_inv_quarter(p, iters, floor, eps):
    """(P^{1/4}, P^{-1/4}) for SPD P, via two trace-normalized coupled-sqrt-NS passes (matmul-only).

    A = P/trace(P) has spectrum ≤ 1 (trace ≥ λ_max), so neither sqrt-NS pass can diverge. First pass →
    A^{1/2}; second pass on A^{1/2} → A^{1/4} (Y) and, with a spectrum floor, the bounded/saturating
    A^{-1/4} (Z). Rescale by trace^{±1/4}. The 4th-root inverse is gentler than P^{-1/2} (smaller exponent).
    """
    tr = jnp.trace(p) + eps
    eye = jnp.eye(p.shape[0], dtype=p.dtype)
    # Floor the spectrum away from 0 BEFORE the forward roots, not just the inverse. The coupled
    # Denman–Beavers sqrt-NS tracks the inverse iterate z alongside y via t = 1.5I − 0.5·z@y; for a
    # near-singular Gram (a low-token MoE expert ⟹ rank-deficient GGᵀ) z diverges and, through the shared
    # t, poisons the forward root y → NaN P^{1/4} → NaN expert weights (train loss stays finite because the
    # NaN expert is rarely routed in-batch, but eval routes to it → NaN). Flooring keeps every eigenvalue
    # ≥ floor so z stays bounded. Spectrum becomes [floor, 1+floor]; still inside the NS sqrt basin.
    a = p / tr + floor * eye
    a_half, _ = _matrix_sqrt_ns(a, iters)  # A^{1/2}
    a_quarter, _ = _matrix_sqrt_ns(a_half, iters)  # A^{1/4}
    _, a_inv_quarter = _matrix_sqrt_ns(a_half, iters)  # ≈ A^{-1/4} (a already floored ⟹ bounded)
    return tr**0.25 * a_quarter, tr**-0.25 * a_inv_quarter


_MUON_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def _mudam_direction(n_t, p, steps, eps):
    """polar( P^{-1/2}_coarse · N ) via the Mudam coupled-NS product form (Muon coeffs, eigh-free).

    Ports levanter.optim.mudam.ns_generalized (muon branch + another_muon): never materializes P^{-1/2};
    the few-step, under-converged Muon-coeff iteration SATURATES, giving the stable q_k inverse-sqrt of
    PR #6588 (= the Mudam inner direction). Used as the optional warm start X⁰ for the curvature fixed
    point. `steps` is kept small (~5) — the iteration is only stable while under-converged.
    """
    a, b, c = _MUON_NS_COEFFS
    m = p.shape[0]
    eye = jnp.eye(m, dtype=p.dtype)
    x = n_t  # [M, N]
    pp = p - x @ x.T  # so A₀ = X Xᵀ + P = p (the curvature); +εI below guards indefiniteness
    nf = jnp.sqrt(jnp.sqrt(jnp.trace(pp @ pp) + eps) + eps + jnp.linalg.norm(x) ** 2)
    x = x / nf
    pp = pp / (nf * nf) + eps * eye
    for _ in range(int(steps)):
        amat = x @ x.T + pp
        bmat = b * amat + c * (amat @ amat)
        x = a * x + bmat @ x
        pp = a * a * pp + a * (bmat @ pp + pp @ bmat) + bmat @ pp @ bmat
    # another_muon: re-orthogonalize (msign) the whitened direction
    x = x / (jnp.linalg.norm(x) + eps)
    for _ in range(int(steps)):
        amat = x @ x.T
        bmat = b * amat + c * (amat @ amat)
        x = a * x + bmat @ x
    return x


def _sym(a):
    return 0.5 * (a + a.T)


def _mclip(m, msign):
    """Project m onto the spectral-norm ball {‖·‖₂ ≤ 1} (clip σ → min(σ, 1)) — matmul-only, no SVD.

    Uses the "mclip2" identity   mclip(m) = (m + msign(m) + (msign(m) − m)·msign(mᵀm − I)) / 2,
    which for m = UΣVᵀ gives U·min(Σ,1)·Vᵀ exactly (msign(mᵀm−I) = V·sign(Σ²−1)·Vᵀ). Benchmarks vs SVD:
    exact in fp32 across σ∈[0,57] (MAE ~1e-4); most bf16-robust of the msign-based variants (σ_out ≤ 1.01
    in-context). One full msign [M,N] + one smaller msign [N,N]. (SVD truncation would be exact too but is
    the only non-matmul op — costly on TPU at ~maxbt·K calls/step.) ``msign`` is the (pre-configured) NS
    orthogonalizer.
    """
    ms1 = msign(m)
    gram = m.T @ m - jnp.eye(m.shape[-1], dtype=m.dtype)
    ms2 = msign(gram)
    return (m + ms1 + (ms1 - m) @ ms2) / 2.0


def _riem_solve(
    n_t, apply_curv, x0, lam_coef, inner_steps, maxbt, msign, constraint, tau_init=0.25, tau0=0.5, beta=0.5, c=1e-4
):
    """Ascent on φ(X)=⟨N,X⟩−(λ/2)⟨X,𝒞X⟩ with an Armijo backtracking line search.

    apply_curv(X) applies the curvature operator 𝒞X — one-sided ``C·X`` or two-sided ``P_L^{1/4} X P_R^{1/4}``.
    constraint:
      "stiefel" (XᵀX=I)  — Riemannian gradient G_R = Z − X·sym(XᵀZ); retract via msign (Newton–Schulz).
      "ball"    (XᵀX⪯I)  — Euclidean gradient Z; project via mclip (SVD-truncate σ→min(σ,1)).
    Direction is always D = msign(grad) (Muon steepest direction under the spectral norm). τ is the largest
    of {τ₀·βʲ}_{j<maxbt} with φ(proj(X+τD)) ≥ φ(X) + c·τ·⟨grad,D⟩. A closed-form τ ∝ ⟨grad,D⟩/(λ·…) blows
    up as λ→0, so the bounded-τ₀ backtracking is used instead (robust across λ).
    Returns (x, phi_traj) with phi_traj[k] = φ at inner iterate k (k=0..K) — for saturation diagnostics.

    The K inner loop (lax.scan) and the maxbt backtracking (lax.fori_loop) are ROLLED, not Python-unrolled:
    the body compiles once instead of K·maxbt copies. Critical for the ball — unrolled, its first-step XLA
    compile took ~540s (mclip = 2 msign/backtrack), exceeding the v5p preemption window ⟹ restart loop.
    """
    ball = constraint == "ball"

    def phi(z):
        return jnp.sum(n_t * z) - 0.5 * lam_coef * jnp.sum(z * apply_curv(z))

    def project(y):
        if ball:
            return _mclip(y, msign)
        return msign(y)

    def sel(cond, a, b):
        # Arithmetic select (exact for boolean cond). Used instead of jnp.where because, when this solve is
        # vmapped over a SHARDED expert axis, the line-search scalars pick up that axis; jnp.where/select
        # strictly requires matching operand shardings (replicated constant vs expert-sharded scalar fails),
        # whereas multiply/add use the lenient broadcast_shardings. Keeps the curvature solve batch-shardable.
        cf = cond.astype(a.dtype)
        return cf * a + (1.0 - cf) * b

    def line_search(x, d, dd, f0, tau_start):
        # largest accepted τ over {τ_start·βʲ}; fori_loop body compiles once (no maxbt unroll).
        def body(_, carry):
            tau, acc = carry
            good = phi(project(x + tau * d)) >= f0 + c * tau * dd
            acc = sel(good & (acc == 0.0), tau, acc)
            tau = sel(good, tau, tau * beta)
            return (tau, acc)

        _, acc = jax.lax.fori_loop(0, int(maxbt), body, (tau_start, jnp.zeros((), x.dtype)))
        return acc

    def step(carry, _):
        x, last_tau = carry
        z = n_t - lam_coef * apply_curv(x)  # Euclidean ascent gradient ∇φ
        grad = z if ball else z - x @ _sym(x.T @ z)  # ball: full gradient; stiefel: tangent projection
        d = msign(grad)
        dd = jnp.sum(grad * d)  # ⟨grad, D⟩ ≥ 0 (directional derivative)
        # Warm-start τ near the previous step's accepted value (allow ×2 growth, cap τ₀); fall back to τ₀ if
        # none accepted last step. The accepted step changes little between inner iterations, so a warm τ₀
        # brackets it in ~2-3 backtracks (toy: maxbt=3 warm ≡ maxbt=10 cold at every λ) — ~2.5× cheaper.
        tau_start = sel(last_tau > 0, jnp.minimum(2.0 * last_tau, tau0), jnp.asarray(tau0, x.dtype))
        acc = line_search(x, d, dd, phi(x), tau_start)
        x = project(x + acc * d)
        return (x, sel(acc > 0, acc, last_tau)), phi(x)

    # Warm-start the inner step-size τ from tau_init (the previous OUTER step's accepted τ); falls back to
    # 0.25 on step 1. Return the final accepted τ so it can be carried to the next outer step.
    init = (x0, jnp.asarray(tau_init, x0.dtype))
    (x_final, tau_final), phis = jax.lax.scan(step, init, None, length=int(inner_steps))  # scan body compiles once
    phi_traj = jnp.concatenate([phi(x0)[None], phis])  # [K+1]
    return x_final, phi_traj, tau_final


def _fw_solve(n_t, apply_curv, x0, lam_coef, inner_steps, msign, eps=1e-12):
    """Frank-Wolfe on the relaxed ball max_{‖X‖₂≤1} ⟨N,X⟩ − (λ/2)⟨X,𝒞X⟩ — closed-form step, NO backtracking.

    Per step: G=N−λ𝒞X, S=msign(G) (LMO over the spectral ball), D=S−X, α=clip(⟨G,D⟩/(λ⟨D,𝒞D⟩),0,1),
    X←X+αD. The convex combination keeps XᵀX⪯I automatically, so NO mclip projection. ~1 msign/step vs the
    Armijo solver's 1+2·maxbt — far cheaper on TPU; matches the line search at small K (O(1/k) tail at large K).
    """

    def step(x, _):
        g = n_t - lam_coef * apply_curv(x)
        s = msign(g)
        d = s - x
        cd = apply_curv(d)
        numer = jnp.sum(g * d)
        denom = lam_coef * jnp.sum(d * cd)
        alpha = jnp.clip(numer / jnp.where(denom > eps, denom, 1.0), 0.0, 1.0)
        return x + alpha * d, None

    x_final, _ = jax.lax.scan(step, x0, None, length=int(inner_steps))
    return x_final


_GPI_SAFETY = 0.05  # shift margin: c = λ·max(diag)·(1+ε) so B = cI − λ𝒞 ⪰ 0 (shifted-polar monotonicity)
_NCG_JCG = 4  # CG steps per Newton-CG round
_NCG_MU_REL = 0.03  # damping fraction: μ = μ_rel·(λ·max diag + ‖S‖_F)
_BAND_TAU = 0.7071067811865476  # √2/2 — singular-value floor for the soft-Stiefel band_ncg solver
_BAND_UCAP = 1.4142135623730951  # √2 — singular-value cap (srank ≥ R/u² = R/2 ⟹ anti-collapse)


def _mclip_ab(m, alpha, beta, msign, em):
    """Double-sided spectral clip σ(m) → clip(σ, α, β) via the polar/sign trick (no SVD): S=msign(m) (polar
    factor UVᵀ), A_α=msign(mᵀm−α²I), A_β=msign(mᵀm−β²I) (symmetric signs V·sign(σ²−·²)·Vᵀ). Then
    ½[(α+β)S + (m−αS)A_α + (βS−m)A_β] = U·clip(σ,α,β)·Vᵀ. Accurate to ~1e-6 in fp32 when σ is in a benign
    range (the operating regime here, σ≈[τ,u]); the floor is unreliable only for σ≪α (tiny/zero), which the
    projection avoids by clipping every step from a well-conditioned start."""
    s = msign(m)
    mtm = em("...ki,...kj->...ij", m, m)
    eye = jnp.broadcast_to(jnp.eye(mtm.shape[-1], dtype=m.dtype), mtm.shape)
    a_a = msign(mtm - (alpha * alpha) * eye)
    a_b = msign(mtm - (beta * beta) * eye)
    return 0.5 * (
        (alpha + beta) * s + em("...ik,...kj->...ij", m - alpha * s, a_a) + em("...ik,...kj->...ij", beta * s - m, a_b)
    )


def _band_ncg_solve(n_t, apply_curv, lam_coef, msign, em, n_steps, tau=_BAND_TAU, ucap=_BAND_UCAP, eps=1e-12):
    """Projected Riemannian PR+ CG for the HARD soft-Stiefel constraint {‖Y‖_F²=R, τ≤σ_i(Y)≤u} in the EK-FAC
    eigenbasis. Frobenius sphere gives the energy budget; the spectral band [τ,u] gives soft singular-value
    control (Frobenius-primary, σ-soft). Base objective max ⟨B,Y⟩−½ΣA·Y² (A=λ·d_scale). Projection onto the
    constraint = mclip_ab(√R·Ỹ/‖Ỹ‖_F, τ, u): pre-scale to the sphere, then band-clip (band exact; the
    downstream hyperball normalizes the direction so only the σ-shape matters). PR+ conjugate directions, sphere
    tangent transport, ADAPTIVE trust-region line search: carry a step scale ``ts`` per matrix; probe at
    {ts, 0.25·ts}+keep-Y, pick best-φ (monotone), then grow ts×2 (cap 1) on a full-step accept and shrink
    ts×0.25 on keep-Y. The shrink is essential — with a FIXED 2-point {1,0.25} the optimum needs sub-0.25 steps
    near convergence, both probes overshoot, only keep-Y fires, and φ PLATEAUS (verified: half stalls at ~2%,
    qt frozen). Adaptive ts reaches the sub-0.25 regime and converges to ~0. Stable in fp32 (stress-tested)."""
    b = n_t
    a = lam_coef * apply_curv(jnp.ones_like(n_t))
    rr = float(min(b.shape[-2], b.shape[-1]))  # Frobenius budget R = min(d1,d2)
    rdot = lambda x, y: jnp.sum(x * y, axis=(-2, -1), keepdims=True)

    def project(y):
        ys = jnp.sqrt(rr) * y / (jnp.linalg.norm(y, axis=(-2, -1), keepdims=True) + eps)
        return _mclip_ab(ys, tau, ucap, msign, em)

    phi = lambda y: rdot(b, y) - 0.5 * rdot(a, y * y)
    egrad = lambda y: b - a * y
    tang = lambda y, z: z - y * (rdot(y, z) / rr)  # Frobenius-sphere tangent projection

    y = project(msign(b))
    g = tang(y, egrad(y))
    ts0 = jnp.ones_like(rdot(b, b))  # per-matrix step scale [..., 1, 1]

    def step(carry, _):
        y, g, p, ts = carry
        yc1 = project(y + ts * p)  # full step at current scale
        yc2 = project(y + 0.25 * ts * p)  # half-decade-smaller probe
        f0, f1, f2 = phi(y), phi(yc1), phi(yc2)
        big = (f1 >= f0) & (f1 >= f2)  # full-step accepted and best
        keep = (f1 < f0) & (f2 < f0)  # neither probe improved ⟹ overshoot
        yn = jnp.where(big, yc1, jnp.where(keep, y, yc2))
        ts = jnp.where(big, jnp.minimum(2.0 * ts, 1.0), jnp.where(keep, 0.25 * ts, ts))  # trust-region adapt
        gn = tang(yn, egrad(yn))
        beta = jnp.maximum(0.0, rdot(gn, gn - tang(yn, g)) / (rdot(g, g) + eps))  # Polak–Ribière+
        pn = gn + beta * tang(yn, p)
        asc = (rdot(gn, pn) > 0).astype(y.dtype)  # restart if not an ascent direction
        return (yn, gn, asc * pn + (1.0 - asc) * gn, ts), None

    (y, _, _, _), _ = jax.lax.scan(step, (y, g, g, ts0), None, length=int(n_steps))
    return y


def _ng_init(n_t, apply_curv, lam_coef, msign, eps):
    """Damped natural-gradient warm start msign(Ñ ⊘ (λ·diag)), diag = apply_curv(1) (the elementwise curvature
    diagonal in the EK-FAC eigenbasis). This lands the solve in the basin where the shifted-polar GPI and
    Newton-CG converge — cold msign(N) does NOT for strong curvature (Newton-CG stalls there). For the EK-FAC
    path apply_curv is elementwise so apply_curv(1)=d_scale exactly; falls back toward msign(N) where d_scale→0.
    """
    return msign(n_t / (lam_coef * apply_curv(jnp.ones_like(n_t)) + eps))


def _gpi_solve(n_t, apply_curv, x0, lam_coef, inner_steps, msign):
    """Shifted-polar generalized power iteration on row-Stiefel: X ← msign(Ñ + cX − λ𝒞X). Monotone MM step
    (B = cI − λ𝒞 ⪰ 0 for c ≥ λ·max diag), but the per-step contraction degrades on a clustered curvature
    spectrum, so it needs large K. Elementwise + msign only ⟹ works unchanged in the 2D and batched paths."""
    c = lam_coef * jnp.max(apply_curv(jnp.ones_like(n_t)), axis=(-2, -1), keepdims=True) * (1.0 + _GPI_SAFETY)

    def step(x, _):
        return msign(n_t + c * x - lam_coef * apply_curv(x)), None

    return jax.lax.scan(step, x0, None, length=int(inner_steps))[0]


def _ncg_solve(n_t, apply_curv, x0, lam_coef, n_rounds, msign, em, es, bsym, j_cg=_NCG_JCG, eps=1e-12):
    """Damped Riemannian Newton-CG on column-Stiefel (XᵀX=I — matches the riemannian solver's convention).
    Per round: Z = Ñ − λ𝒞X; S = sym(XᵀZ); Riemannian grad G_R = Z − X·S; solve the damped Newton system
    (μI − H)η = G_R by ``j_cg`` CG steps with H[η] = Π(−λ𝒞η − η·S), Π_X(Y) = Y − X·sym(XᵀY); retract
    X ← msign(X + η). μ = μ_rel·(λ·max diag + ‖S‖_F) (Frobenius bound on ‖S‖₂ — TPU-safe, no eigh). Converges
    where the GPI power-iteration tail stalls; needs the NG warm start (cold msign(N) leaves (μI−H) indefinite).
    ``em``/``es``/``bsym`` are the path's sharding-aware einsum/reduction/symmetrize primitives."""
    diagmax = lam_coef * jnp.max(apply_curv(jnp.ones_like(n_t)), axis=(-2, -1), keepdims=True)
    xtY = lambda x, Y: bsym(em("...ki,...kj->...ij", x, Y))  # sym(Xᵀ Y)  [..., N, N]
    proj = lambda x, Y: Y - em("...ik,...kj->...ij", x, xtY(x, Y))  # Π_X(Y) = Y − X·sym(XᵀY)
    rdot = lambda a, b: es("...mn,...mn->...", a, b)[..., None, None]

    def rnd(x, _):
        z = n_t - lam_coef * apply_curv(x)
        S = xtY(x, z)  # sym(Xᵀ Z)  [..., N, N]
        g_r = z - em("...ik,...kj->...ij", x, S)  # Riemannian gradient
        mu = _NCG_MU_REL * (diagmax + jnp.sqrt(rdot(S, S)))  # μ_rel·(λ·max diag + ‖S‖_F)
        hvp = lambda e: proj(x, -lam_coef * apply_curv(e) - em("...ik,...kj->...ij", e, S))
        eta = jnp.zeros_like(g_r)
        r = g_r
        p = g_r
        rs = rdot(r, r)
        for _ in range(int(j_cg)):  # CG solve (μI − H) η = G_R in the tangent space
            ap = mu * p - hvp(p)
            denom = rdot(p, ap)
            a = rs / jnp.where(jnp.abs(denom) > eps, denom, 1.0)
            eta = eta + a * p
            r = r - a * ap
            rs2 = rdot(r, r)
            p = r + (rs2 / jnp.where(rs > eps, rs, 1.0)) * p
            rs = rs2
        return msign(x + eta), None

    return jax.lax.scan(rnd, x0, None, length=int(n_rounds))[0]


def _ek_secular_solve(n_t, apply_curv, lam_coef, bisect_iters=40, target=1.0, eps=1e-12):
    """Relaxation 2: fixed row/column norms in the EK-FAC rotated basis. The EK-FAC inner objective is the
    diagonal quadratic φ(Y) = Σ B_ij Y_ij − ½ Σ A_ij Y_ij², with A = λ·d_scale and B = Q_BᵀN Q_A (= ``n_t``
    here, already rotated). Imposing the relaxed unit norm on the dimension the Stiefel manifold fixes —
    column-Stiefel (XᵀX=I, tall) ⟹ unit columns; row-Stiefel (XXᵀ=I, wide) ⟹ unit rows; ties → rows — makes
    it SEPARABLE: each row/column is an independent secular problem with closed-form Y_ij = B_ij/(A_ij+ν) and
    ν the unique root (>−min A) of Σ B²/(A+ν)² = target. s(ν) is monotone decreasing on (−min A, ∞), so a
    vectorized bisection over the per-line ν solves all lines at once. No msign, no eigh — pure elementwise +
    reductions, so it shards trivially (matrix dims replicated in the batched path)."""
    B = n_t
    A = lam_coef * apply_curv(jnp.ones_like(n_t))  # A = λ·d_scale ≥ 0
    axis = -1 if B.shape[-2] <= B.shape[-1] else -2  # wide→row-Stiefel→per-row; tall→col-Stiefel→per-col; tie→row
    s = lambda nu: jnp.sum((B / (A + nu)) ** 2, axis=axis, keepdims=True)
    lo = -jnp.min(A, axis=axis, keepdims=True) + eps  # s(lo)→∞ ≥ target
    hi = jnp.sqrt(jnp.sum(B * B, axis=axis, keepdims=True) / target) + eps  # A≥0 ⟹ s(hi) ≤ target (valid bracket)

    def step(c, _):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        big = s(mid) > target  # root is to the right ⟹ raise lo
        return (jnp.where(big, mid, lo), jnp.where(big, hi, mid)), None

    (lo, hi), _ = jax.lax.scan(step, (lo, hi), None, length=int(bisect_iters))
    return B / (A + 0.5 * (lo + hi))


def _ek_doublenorm_solve(n_t, apply_curv, lam_coef, r_val, c_val, outer_iters=15, bisect_iters=30, eps=1e-12):
    """Double-norm relaxation in the EK-FAC ROTATED basis (additive-Sinkhorn dual). The rotated objective is
    diagonal: max Σ B_ij Y_ij − ½ Σ A_ij Y_ij² (A=λ·d_scale, B=Q_BᵀNQ_A=``n_t``) s.t. Σ_j Y_ij²=r_i AND
    Σ_i Y_ij²=c_j. KKT gives the closed form Y_ij=B_ij/(A_ij+u_i+v_j); the dual potentials u,v enforce the
    row/col budgets and are found by alternating per-line secular bisection (the additive analogue of Sinkhorn
    — monotone scalar roots, globally optimal over the transport polytope). Gauge (u+t, v−t) pinned by
    mean(v)=0. Only matmuls are the rotations (in apply_curv setup / post); the solve is elementwise +
    reductions. r_val=R/d1, c_val=R/d2, R=min(d1,d2)."""
    B = n_t
    A = lam_coef * apply_curv(jnp.ones_like(n_t))  # A = λ·d_scale ≥ 0
    B2 = B * B

    def potential(aeff, budget, axis):  # solve s (keepdims along `axis`) s.t. Σ_axis B2/(aeff+s)² = budget
        s = lambda sp: jnp.sum(B2 / (aeff + sp) ** 2, axis=axis, keepdims=True)
        lo = -jnp.min(aeff, axis=axis, keepdims=True) + eps  # aeff+s ≥ eps ⟹ s(lo)→large
        hi = lo + jnp.sqrt(jnp.sum(B2, axis=axis, keepdims=True) / budget) + eps  # ⟹ s(hi) ≤ budget

        def step(cc, _):
            lo, hi = cc
            mid = 0.5 * (lo + hi)
            big = s(mid) > budget
            return (jnp.where(big, mid, lo), jnp.where(big, hi, mid)), None

        (lo, hi), _ = jax.lax.scan(step, (lo, hi), None, length=int(bisect_iters))
        return 0.5 * (lo + hi)

    u = jnp.zeros_like(jnp.sum(B2, axis=-1, keepdims=True))  # [..., M, 1]
    v = jnp.zeros_like(jnp.sum(B2, axis=-2, keepdims=True))  # [..., 1, N]

    def outer(carry, _):
        u, v = carry
        u = potential(A + v, r_val, -1)  # row budgets: reduce over cols
        v = potential(A + u, c_val, -2)  # col budgets: reduce over rows
        t = jnp.mean(v, axis=-1, keepdims=True)  # gauge fix
        return (u + t, v - t), None

    (u, v), _ = jax.lax.scan(outer, (u, v), None, length=int(outer_iters))
    return B / (A + u + v)


def _orignorm_solve(n_orig, apply_curv_orig, x0, inner_steps, normalize, c):
    """Shifted normalization-GPI for the ORIGINAL-coordinate row/column-norm relaxation: X ← normalize(N + cX
    − 𝒞[X]), monotone for c ≥ λ·max W. Unlike the EK-basis secular solve, the normalization (col/row-norm) is
    in the ORIGINAL basis — Q_A/Q_B mix columns/rows, so it is NOT eigenbasis-separable and each step must
    rotate in/out (apply_curv_orig = λ·Q_B(W⊙(Q_BᵀXQ_A))Q_Aᵀ). normalize = colnorm (col-Stiefel relaxation,
    diag(XᵀX)=1) or rownorm (row-Stiefel, diag(XXᵀ)=1). X stays in original coordinates throughout."""

    def step(x, _):
        return normalize(n_orig + c * x - apply_curv_orig(x)), None

    return jax.lax.scan(step, x0, None, length=int(inner_steps))[0]


def _orignorm_ncg_solve(
    n_orig, curv_orig, x0, n_rounds, normalize, red_axis, lam_max_w, j_cg=4, mu_rel=0.03, eps=1e-12
):
    """Damped product-sphere Riemannian Newton-CG for the ORIGINAL-coordinate row/column-norm relaxation.
    The product-of-spheres tangent projection and the Λ-scaling are ELEMENTWISE per-line (each row/column is a
    sphere): Π_X(Y) = Y − ⟨X,Y⟩_line·X, Λ-scale = ⟨X,Z⟩_line·η, with ⟨·,·⟩_line the reduction along ``red_axis``
    (−1 for row-norm, −2 for col-norm). Per round: Z=N−𝒞[X]; G_R=Z−⟨X,Z⟩·X; (μI−H)η=G_R via j_cg CG steps,
    H[η]=Π(−𝒞[η]−⟨X,Z⟩·η); retract X←normalize(X+η). μ=μ_rel(λ·maxW+max|Λ|) (TPU-safe). The only matmuls are
    in ``curv_orig`` (𝒞 needs the EK rotations); everything else is elementwise. Needs the NG warm start x0 —
    cold normalize(N) leaves (μI−H) indefinite and the solve stalls."""

    def rnd(x, _):
        z = n_orig - curv_orig(x)
        lam_vec = jnp.sum(x * z, axis=red_axis, keepdims=True)  # Λ diagonal (per row/col)
        g_r = z - lam_vec * x
        mu = mu_rel * (lam_max_w + jnp.max(jnp.abs(lam_vec), axis=(-2, -1), keepdims=True))
        proj = lambda Y: Y - jnp.sum(x * Y, axis=red_axis, keepdims=True) * x
        hvp = lambda e: proj(-curv_orig(e) - lam_vec * e)
        eta = jnp.zeros_like(g_r)
        r = g_r
        p = g_r
        rs = jnp.sum(r * r, axis=(-2, -1), keepdims=True)
        for _ in range(int(j_cg)):
            ap = mu * p - hvp(p)
            denom = jnp.sum(p * ap, axis=(-2, -1), keepdims=True)
            a = rs / jnp.where(jnp.abs(denom) > eps, denom, 1.0)
            eta = eta + a * p
            r = r - a * ap
            rs2 = jnp.sum(r * r, axis=(-2, -1), keepdims=True)
            p = r + (rs2 / jnp.where(rs > eps, rs, 1.0)) * p
            rs = rs2
        return normalize(x + eta), None

    return jax.lax.scan(rnd, x0, None, length=int(n_rounds))[0]


def _doublenorm_ncg_solve(
    n_orig, curv_orig, x0, n_rounds, r_val, c_val, lam_max_w, j_cg=4, n_proj=20, n_sink=20, mu_rel=0.1, eps=1e-12
):
    """Damped Riemannian Newton-CG on the DOUBLE-norm manifold {diag(XXᵀ)=r, diag(XᵀX)=c} (EK-FAC curvature).
    Tangent proj Π_X(Y)=Y−a⊙_row X−X⊙_col b, with (a,b) the multipliers of the [[Diag r,S],[Sᵀ,Diag c]] system
    (S=X²) found by a few block-coordinate sweeps (Π is gauge-invariant; the gauge mode is pinned by mean(b)=0).
    Retraction = Sinkhorn scaling of squared entries to row sums r / col sums c → exactly feasible. HVP
    H[η]=Π(−𝒞η−a⊙_row η−η⊙_col b). NG warm start (Sinkhorn-retracted at entry). Only 𝒞[·] is matmul; the
    projection/retraction are elementwise + row/col reductions. r_val=R/d1, c_val=R/d2, R=min(d1,d2)."""

    def mults(Y, X):
        S = X * X
        u = jnp.sum(Y * X, axis=-1, keepdims=True)  # diag(YXᵀ)  [..., M, 1]
        v = jnp.sum(Y * X, axis=-2, keepdims=True)  # diag(XᵀY)  [..., 1, N]
        a = jnp.zeros_like(u)
        b = jnp.zeros_like(v)
        for _ in range(int(n_proj)):
            a = (u - jnp.sum(S * b, axis=-1, keepdims=True)) / r_val
            b = (v - jnp.sum(S * a, axis=-2, keepdims=True)) / c_val
            b = b - jnp.mean(b, axis=-1, keepdims=True)  # gauge fix
        return a, b

    def proj(Y, X):
        a, b = mults(Y, X)
        return Y - a * X - X * b, a, b

    def retract(Y):
        tt = Y * Y
        p = jnp.ones_like(jnp.sum(tt, axis=-1, keepdims=True))
        q = jnp.ones_like(jnp.sum(tt, axis=-2, keepdims=True))
        for _ in range(int(n_sink)):
            p = r_val / (jnp.sum(tt * q, axis=-1, keepdims=True) + eps)
            q = c_val / (jnp.sum(tt * p, axis=-2, keepdims=True) + eps)
        return jnp.sqrt(p) * Y * jnp.sqrt(q)

    def rnd(x, _):
        z = n_orig - curv_orig(x)
        g_r, a, b = proj(z, x)
        mu = mu_rel * (
            lam_max_w
            + jnp.max(jnp.abs(a), axis=(-2, -1), keepdims=True)
            + jnp.max(jnp.abs(b), axis=(-2, -1), keepdims=True)
        )
        hvp = lambda e: proj(-curv_orig(e) - a * e - e * b, x)[0]
        eta = jnp.zeros_like(g_r)
        r = g_r
        p = g_r
        rs = jnp.sum(r * r, axis=(-2, -1), keepdims=True)
        for _ in range(int(j_cg)):
            ap = mu * p - hvp(p)
            denom = jnp.sum(p * ap, axis=(-2, -1), keepdims=True)
            al = rs / jnp.where(jnp.abs(denom) > eps, denom, 1.0)
            eta = eta + al * p
            r = r - al * ap
            rs2 = jnp.sum(r * r, axis=(-2, -1), keepdims=True)
            p = r + (rs2 / jnp.where(rs > eps, rs, 1.0)) * p
            rs = rs2
        return retract(x + eta), None

    return jax.lax.scan(rnd, retract(x0), None, length=int(n_rounds))[0]


def _best_phi_orig(cands, n_orig, curv_orig):
    """Per-matrix argmax of φ(z)=⟨N,z⟩−½⟨z,𝒞z⟩ in ORIGINAL coords (curv_orig includes λ) — cheap guard so the
    returned direction is never worse than the warm start / cold normalize(N)."""
    f = lambda z: jnp.sum(n_orig * z, axis=(-2, -1), keepdims=True) - 0.5 * jnp.sum(
        z * curv_orig(z), axis=(-2, -1), keepdims=True
    )
    best, bf = cands[0], f(cands[0])
    for c in cands[1:]:
        fc = f(c)
        best = jnp.where(fc >= bf, c, best)
        bf = jnp.maximum(fc, bf)
    return best


def _best_phi(cands, n_t, apply_curv, lam_coef, es):
    """Per-matrix argmax of φ(z)=⟨Ñ,z⟩−(λ/2)⟨z,𝒞z⟩ over candidate directions — a cheap (no msign) guard so
    the returned direction is never worse than the warm start / cold msign(N)."""
    f = lambda z: es("...mn,...mn->...", n_t, z) - 0.5 * lam_coef * es("...mn,...mn->...", z, apply_curv(z))
    best, bf = cands[0], f(cands[0])
    for c in cands[1:]:
        fc = f(c)
        best = jnp.where((fc >= bf)[..., None, None], c, best)
        bf = jnp.maximum(fc, bf)
    return best


def _power_iter(mat, q, iters, eps):
    for _ in range(int(iters)):
        mq = mat @ q
        q = mq / (jnp.linalg.norm(mq) + eps)
    return q, jnp.dot(q, mat @ q) * _EMAX_MARGIN  # (updated q, inflated Rayleigh ≈ λ_max)


def _curv_direction_2d(
    g,
    n,
    p,
    q,
    p_r,
    q_r,
    inner_x,
    *,
    rho,
    lam_static,
    lam_coef,
    alpha,
    steps,
    eps,
    ctype,
    inner_steps,
    power_iters,
    curv_power,
    mudam_init,
    mudam_steps,
    two_sided,
    floor,
    inner_solver,
    maxbt,
    warm_start,
    constraint,
    shard_ns: bool = True,
    bias_t=None,
    warm_tau=None,
    kl_shampoo: bool = False,
    ekfac: bool = False,
    aug_eig=None,
    ekfac_power: str = "half",
    q_a_in=None,
    q_b_in=None,
):
    """One matrix. g, n: [out, in]; p/q = left Gram P_L [M,M] + power vec; p_r/q_r = right Gram P_R [N,N] + vec.
    q_a_in/q_b_in: MAINTAINED eigenbases (refined one QR step/outer-step); used for the KL inverse + EK-FAC.

    one-sided (two_sided=False): X = msign(N + λ(α√e_max·I − C)X), C = P_L^{1/2} (sqrt) or P_L/√e_max (linear);
    warm start msign(P_L^{-1/2} N) (Mudam q_k).
    two-sided (two_sided=True): X = msign(N + λ(α(e_L·e_R)^{1/4}·X − P_L^{1/4} X P_R^{1/4})); warm start the
    Shampoo direction msign(P_L^{-1/4} N P_R^{-1/4}). 1/4+1/4 = 1/2 total power ⟹ same ~G units, λ dimensionless.

    inner_solver="riemannian_muon": gradient ascent with Armijo line search, cold start msign(N) (optionally
    warm-started from carried inner_x). One- or two-sided curvature; constraint="stiefel" (msign retraction)
    or "ball" (XᵀX⪯I, mclip/SVD projection). Robust across λ, K-stable.
    Returns (new_p, new_q, new_p_r, new_q_r, new_inner_x, phi_traj, direction); phi_traj[k]=φ at inner step k.
    """
    # ``shard_ns=False`` is the grug MoE path: this is vmapped over the expert axis under an explicit mesh,
    # and the per-expert inputs carry a ``model`` sharding. The many curvature matmuls (Gram g_tᵀg_t, power
    # iteration, P^{1/4}) then contract over ``model``-sharded dims → "ambiguous output sharding", and the
    # NS sharding constraint is a failing assert. So replicate every input to P(None, …) first (mirrors
    # MuonH's _zeropower_via_newtonschulz_replicated) and run NS unsharded. Dense (qwen3) keeps shard_ns=True.
    msign = functools.partial(
        zeropower_via_newtonschulz5, steps=steps, eps=eps, coefficient_type=ctype, shard=shard_ns
    )
    if not shard_ns:
        rep = lambda a: jax.sharding.reshard(a, jax.sharding.PartitionSpec(*([None] * a.ndim)))
        g, n, p, q, p_r, q_r, inner_x = rep(g), rep(n), rep(p), rep(q), rep(p_r), rep(q_r), rep(inner_x)

    out, inn = g.shape
    transpose = out < inn
    g_t = g.T if transpose else g  # [M, N], M ≥ N
    n_t = n.T if transpose else n

    da, db = g_t.shape[0], g_t.shape[1]

    def _inv_q(a, qm):  # κ-damped S^{-1} = Q Diag(1/(λ+κ)) Qᵀ using the MAINTAINED eigenbasis qm; λ=diag(qmᵀ a qm)
        lam = jnp.maximum(jnp.sum(qm * (a @ qm), axis=0), 0.0)  # Rayleigh eigenvalues [n]
        inv = 1.0 / (lam + floor * (jnp.sum(lam) + eps))
        return (qm * inv) @ qm.T

    # KL-Shampoo (arXiv 2509.03378) Gram update: whiten each outer product by the OTHER factor's
    # (κ-damped) inverse using the INCOMING maintained eigenbasis, with 1/d scaling — the coupled-MLE estimate
    # S_a←(1-β)S_a+(β/d_b)G S_b^{-1}Gᵀ. Standard Shampoo (kl_shampoo=False) uses the plain outer product GGᵀ.
    if kl_shampoo and two_sided:
        sb_inv = _inv_q(p_r, q_b_in)  # S_b^{-1} = Q_b Diag(1/λ_b) Q_bᵀ (maintained eigenbasis, no eigh)
        delta_a = (g_t @ sb_inv @ g_t.T) / db
    else:
        delta_a = g_t @ g_t.T
    new_p = rho * p + (1.0 - rho) * delta_a  # P_L [M, M] (stored UNcorrected)
    # Bias-correct the Gram for the curvature operator only: P̂ = P/(1-rho^t). The raw EMA new_p goes into
    # state; the debiased p_c derives the curvature (e_max, P^{1/4}) so the operator has its true scale from
    # step 1 (not biased toward the eps·I init). pdiv→1 as t→∞. bias_t=None ⟹ no correction (qwen3 parity).
    pdiv = 1.0 if bias_t is None else (1.0 - rho ** jnp.asarray(bias_t, new_p.dtype))
    p_c = new_p / pdiv
    new_q, emax = _power_iter(p_c, q, power_iters, eps)
    eye = jnp.eye(new_p.shape[0], dtype=new_p.dtype)

    # Right Gram P_R + its 1/4 powers (two-sided), or the one-sided curvature matrix C.
    if two_sided:
        if kl_shampoo:
            sa_inv = _inv_q(p, q_a_in)  # S_a^{-1} = Q_a Diag(1/λ_a) Q_aᵀ (maintained eigenbasis)
            delta_b = (g_t.T @ sa_inv @ g_t) / da
        else:
            delta_b = g_t.T @ g_t
        new_p_r = rho * p_r + (1.0 - rho) * delta_b  # P_R [N, N] (stored UNcorrected)
        pr_c = new_p_r / pdiv
        new_q_r, emax_r = _power_iter(pr_c, q_r, power_iters, eps)
        pl4, plinv4 = _pow4_inv_quarter(p_c, power_iters, floor, eps)  # P_L^{1/4}, P_L^{-1/4}
        pr4, prinv4 = _pow4_inv_quarter(pr_c, power_iters, floor, eps)  # P_R^{1/4}, P_R^{-1/4}
    else:
        new_p_r, new_q_r = p_r, q_r
        se = jnp.sqrt(emax) + eps
        if curv_power == "sqrt":
            # tr(P) ≥ λ_max ⟹ P/tr spectrum ≤ 1 ⟹ NS sqrt can't diverge. C = √tr·(P/tr)^{1/2} = P^{1/2}.
            tr = jnp.trace(p_c) + eps
            y_half, _ = _matrix_sqrt_ns(p_c / tr, power_iters)
            curv = jnp.sqrt(tr) * y_half
        else:
            curv = p_c / se  # P/√e_max

    # EK-FAC augmented eigenvalues (arXiv 2509.03378, KL-SOAP): replace the Kronecker eigenvalues λ_a⊗λ_b
    # with a full per-coordinate D = EMA((Q_aᵀ G Q_b)²) in the KL-Shampoo eigenbasis (Q_a,Q_b = eigvecs of the
    # bias-corrected Grams). The curvature operator becomes rotate→scale by (D̂)^{1/4}→rotate-back: P[X] =
    # Q_a((D̂^{1/4})⊙(Q_aᵀ X Q_b))Q_bᵀ. D̂≈λ_aλ_b (second-moment eigenvalue), so the 1/4 exponent matches the
    # two-sided P^{1/4} convention and keeps λ comparable. D is bias-corrected by pdiv like the Grams.
    # Refine the maintained eigenbasis one QR orthogonal-iteration step from the NEW Grams (Q tracks the
    # slowly-changing Gram across outer steps). Passthrough when no maintained Q is supplied (qwen3 path).
    new_qa = _qr_refine(new_p @ q_a_in) if q_a_in is not None else q_a_in
    new_qb = _qr_refine(new_p_r @ q_b_in) if (q_b_in is not None and two_sided) else q_b_in

    new_D = aug_eig
    ekfac_on = ekfac and two_sided
    if ekfac_on:
        qa, qb = new_qa, new_qb
        ghat = qa.T @ g_t @ qb
        base_D = aug_eig if aug_eig is not None else jnp.zeros_like(ghat)
        new_D = rho * base_D + (1.0 - rho) * (ghat * ghat)
        d_hat = new_D / pdiv
        # Curvature scale (both units-G ⟹ λ dimensionless): "half" = √S = D̂^{1/2} (natural-gradient curvature);
        # "quarter_trace" = D̂^{1/4}·tr_D^{1/4}, which reproduces the existing P_L^{1/4}·P_R^{1/4} (S^{1/4}) shape.
        if ekfac_power == "quarter_trace":
            d_scale = jnp.power(d_hat + eps, 0.25) * jnp.power(jnp.sum(d_hat) + eps, 0.25)
        else:
            d_scale = jnp.power(d_hat + eps, 0.5)

    phi_traj = jnp.zeros((int(inner_steps) + 1,), dtype=n_t.dtype)  # default; only riemannian fills it
    # Warm-start τ carried across outer steps; default 0.25 on step 1. tau_out is returned so the caller can
    # store the accepted τ for the next outer step (non-riemannian paths just pass it through).
    tau_init = 0.25 if warm_tau is None else warm_tau
    tau_out = jnp.asarray(tau_init, n_t.dtype)

    # Build the inner-solve problem. EK-FAC: solve in the eigenbasis where the curvature is the ELEMENTWISE
    # d_scale ⊙ X̂ — rotate N in ONCE, rotate the solution out ONCE (msign/mclip are orthogonally equivariant
    # and ⟨·,·⟩/the ball are rotation-invariant, so the hat-space optimum rotates back exactly). This avoids
    # the two matmuls per apply_curv call that a naive Q(D⊙QᵀXQ)Qᵀ operator would incur every inner step.
    if ekfac_on:
        n_solve = qa.T @ n_t @ qb
        apply_curv = lambda X: d_scale * X
        post = lambda X: qa @ X @ qb.T
    else:
        n_solve = n_t
        apply_curv = (lambda X: pl4 @ X @ pr4) if two_sided else (lambda X: curv @ X)
        post = lambda X: X

    if inner_solver == "riemannian_muon":
        cold = msign(n_solve)  # cold start msign(N) (λ=0 optimum); warm start from carried X optional
        if lam_static > 0.0:
            x0 = jnp.where(jnp.linalg.norm(inner_x) > eps, inner_x, cold) if warm_start else cold
            x, phi_traj, tau_out = _riem_solve(
                n_solve, apply_curv, x0, lam_coef, inner_steps, maxbt, msign, constraint, tau_init=tau_init
            )
        else:
            x = cold
        xr = post(x)
        return (
            new_p,
            new_q,
            new_p_r,
            new_q_r,
            xr,
            phi_traj,
            (xr.T if transpose else xr),
            tau_out,
            new_D,
            new_qa,
            new_qb,
        )

    if inner_solver == "frank_wolfe":
        cold = msign(n_solve)
        x = _fw_solve(n_solve, apply_curv, cold, lam_coef, inner_steps, msign) if lam_static > 0.0 else cold
        xr = post(x)
        return (
            new_p,
            new_q,
            new_p_r,
            new_q_r,
            xr,
            phi_traj,
            (xr.T if transpose else xr),
            tau_out,
            new_D,
            new_qa,
            new_qb,
        )

    if inner_solver == "secular":
        x = _ek_secular_solve(n_solve, apply_curv, lam_coef) if lam_static > 0.0 else msign(n_solve)
        xr = post(x)
        return (
            new_p,
            new_q,
            new_p_r,
            new_q_r,
            xr,
            phi_traj,
            (xr.T if transpose else xr),
            tau_out,
            new_D,
            new_qa,
            new_qb,
        )

    if inner_solver == "band_ncg":
        emf = lambda s, *a: jnp.einsum(s, *a)
        x = (
            _band_ncg_solve(n_solve, apply_curv, lam_coef, msign, emf, inner_steps)
            if lam_static > 0.0
            else msign(n_solve)
        )
        xr = post(x)
        return (
            new_p,
            new_q,
            new_p_r,
            new_q_r,
            xr,
            phi_traj,
            (xr.T if transpose else xr),
            tau_out,
            new_D,
            new_qa,
            new_qb,
        )

    if inner_solver == "doublenorm_secular":
        md, nd = n_solve.shape[-2], n_solve.shape[-1]
        rmin = min(md, nd)
        x = (
            _ek_doublenorm_solve(n_solve, apply_curv, lam_coef, rmin / md, rmin / nd)
            if lam_static > 0.0
            else msign(n_solve)
        )
        xr = post(x)
        return (
            new_p,
            new_q,
            new_p_r,
            new_q_r,
            xr,
            phi_traj,
            (xr.T if transpose else xr),
            tau_out,
            new_D,
            new_qa,
            new_qb,
        )

    if inner_solver in ("orignorm", "orignorm_ncg"):
        cn = lambda M: M / (jnp.sqrt(jnp.sum(M * M, axis=-2, keepdims=True)) + eps)  # unit columns (col-Stiefel)
        rn = lambda M: M / (jnp.sqrt(jnp.sum(M * M, axis=-1, keepdims=True)) + eps)  # unit rows (row-Stiefel)
        row_mode = n_t.shape[-2] <= n_t.shape[-1]  # wide→row-Stiefel; tall→col-Stiefel; tie→row
        normalize = rn if row_mode else cn
        if lam_static <= 0.0:
            x = normalize(n_t)
        else:
            rin = lambda Z: qa.T @ Z @ qb  # rotate into the EK-FAC eigenbasis
            curv_orig = lambda X: lam_coef * (qa @ (d_scale * rin(X)) @ qb.T)  # 𝒞[X] in ORIGINAL coords
            if inner_solver == "orignorm":  # plain shifted-normalization GPI
                cc = lam_coef * jnp.max(d_scale) * (1.0 + _GPI_SAFETY)
                x = _orignorm_solve(n_t, curv_orig, normalize(n_t), inner_steps, normalize, cc)
            else:  # product-sphere Newton-CG with NG warm start
                ng = normalize(qa @ (rin(n_t) / (lam_coef * d_scale + eps)) @ qb.T)
                red = -1 if row_mode else -2
                x = _orignorm_ncg_solve(n_t, curv_orig, ng, inner_steps, normalize, red, lam_coef * jnp.max(d_scale))
                x = _best_phi_orig([normalize(n_t), ng, x], n_t, curv_orig)
        return (
            new_p,
            new_q,
            new_p_r,
            new_q_r,
            x,
            phi_traj,
            (x.T if transpose else x),
            tau_out,
            new_D,
            new_qa,
            new_qb,
        )

    if inner_solver == "doublenorm_ncg":
        if lam_static <= 0.0:
            x = n_t
        else:
            md, nd = n_t.shape[-2], n_t.shape[-1]
            rmin = min(md, nd)
            rin = lambda Z: qa.T @ Z @ qb
            curv_orig = lambda X: lam_coef * (qa @ (d_scale * rin(X)) @ qb.T)
            ng = qa @ (rin(n_t) / (lam_coef * d_scale + eps)) @ qb.T  # NG direction (retracted inside the solver)
            x = _doublenorm_ncg_solve(
                n_t, curv_orig, ng, inner_steps, rmin / md, rmin / nd, lam_coef * jnp.max(d_scale)
            )
        return (
            new_p,
            new_q,
            new_p_r,
            new_q_r,
            x,
            phi_traj,
            (x.T if transpose else x),
            tau_out,
            new_D,
            new_qa,
            new_qb,
        )

    if inner_solver in ("gpi", "ncg"):
        emf = lambda s, a, b: jnp.einsum(s, a, b)
        bsymf = lambda a: 0.5 * (a + jnp.swapaxes(a, -1, -2))
        if lam_static <= 0.0:
            x = msign(n_solve)
        else:
            ng = _ng_init(n_solve, apply_curv, lam_coef, msign, eps)
            if inner_solver == "gpi":
                x = _gpi_solve(n_solve, apply_curv, ng, lam_coef, inner_steps, msign)
            else:
                x = _ncg_solve(n_solve, apply_curv, ng, lam_coef, inner_steps, msign, emf, emf, bsymf)
            x = _best_phi([msign(n_solve), ng, x], n_solve, apply_curv, lam_coef, emf)
        xr = post(x)
        return (
            new_p,
            new_q,
            new_p_r,
            new_q_r,
            xr,
            phi_traj,
            (xr.T if transpose else xr),
            tau_out,
            new_D,
            new_qa,
            new_qb,
        )

    # --- fixed-point inner solver ---
    if two_sided:
        if mudam_init:
            x = msign(plinv4 @ n_t @ prinv4)
        else:
            x = msign(n_t)
        if lam_static > 0.0:
            shift2 = alpha * (emax * emax_r) ** 0.25  # = max singular value of X ↦ P_L^{1/4} X P_R^{1/4}
            for _ in range(int(inner_steps)):
                arg = n_t + lam_coef * (shift2 * x - pl4 @ x @ pr4)
                x = msign(arg)
        return (
            new_p,
            new_q,
            new_p_r,
            new_q_r,
            inner_x,
            phi_traj,
            (x.T if transpose else x),
            tau_out,
            new_D,
            new_qa,
            new_qb,
        )

    x = _mudam_direction(n_t, new_p, mudam_steps, eps) if mudam_init else msign(n_t)
    if lam_static > 0.0:
        operator = lam_coef * (alpha * se * eye - curv)  # PSD for α≥1
        for _ in range(int(inner_steps)):
            x = msign(n_t + operator @ x)
    return new_p, new_q, new_p_r, new_q_r, inner_x, phi_traj, (x.T if transpose else x), tau_out, new_D, new_qa, new_qb


def curv_direction_batched(
    g,
    n,
    p,
    q,
    p_r,
    q_r,
    *,
    rho,
    lam_coef,
    lam_static,
    steps,
    eps,
    ctype,
    inner_steps,
    power_iters,
    floor,
    maxbt,
    constraint,
    out_p=None,
    bias_t=None,
    warm_tau=None,
    solver="riemannian_muon",
    kl_shampoo=False,
    ekfac=False,
    aug_eig=None,
    ekfac_power="half",
    q_a_in=None,
    q_b_in=None,
):
    """Batched two-sided curvature direction over a leading stack axis (e.g. MoE experts).

    Same Riemannian two-sided P^{1/4} solve as ``_curv_direction_2d`` (riemannian_muon, mudam_init=False,
    warm_start=False, alpha=1), but vectorized over leading dims with ellipsis einsums instead of ``vmap``
    (the grug/KLSOAP STACK_BATCH layout). This is what lets the stack axis stay SHARDED: every contraction
    is over a (replicated) matrix dim, so the leading-axis sharding propagates unambiguously and the
    line-search scalars become per-stack ``[...]`` vectors — no ``vmap`` ⟹ no ``unmapped_aval`` and the
    arithmetic select is sharding-clean. ``out_p`` (matrix out_sharding) follows KLSOAP; pass the
    batch-sharded ``P(stack, None, None)`` to distribute, or None to let the propagator infer.
    Returns (new_p, new_q, new_p_r, new_q_r, inner_x, direction); direction has g's shape.
    """
    em = lambda eq, *a: jnp.einsum(eq, *a, out_sharding=out_p)
    es = lambda eq, *a: jnp.einsum(eq, *a)  # scalar/vector reductions: let the propagator infer
    bt = lambda a: jnp.swapaxes(a, -1, -2)
    coeffs = NEWTON_SCHULZ_COEFFICIENTS[ctype]

    def msign(x):
        x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
        tr = x.shape[-2] > x.shape[-1]
        if tr:
            x = bt(x)
        for i in range(int(steps)):
            a, b, c = coeffs[i % len(coeffs)]
            amat = em("...ik,...jk->...ij", x, x)
            bmat = b * amat + c * em("...ik,...kj->...ij", amat, amat)
            x = a * x + em("...ik,...kj->...ij", bmat, x)
        return bt(x) if tr else x

    def sqrtns(a, iters):  # coupled Denman–Beavers; returns (Y→a^{1/2}, Z→a^{-1/2})
        eye = jnp.broadcast_to(jnp.eye(a.shape[-1], dtype=a.dtype), a.shape)
        y, z = a, eye
        for _ in range(int(iters)):
            t = 1.5 * eye - 0.5 * em("...ik,...kj->...ij", z, y)
            y = em("...ik,...kj->...ij", y, t)
            z = em("...ik,...kj->...ij", t, z)
        return y, z

    def pow4(pp):  # P^{1/4}, trace-normalized + spectrum-floored (see _pow4_inv_quarter)
        eye = jnp.broadcast_to(jnp.eye(pp.shape[-1], dtype=pp.dtype), pp.shape)
        # trace via multiply+sum, NOT einsum-diagonal: the latter masks with a replicated eye via lax.select,
        # which rejects the stack-sharded pp under the explicit mesh. Multiply broadcasts leniently.
        tr = jnp.sum(pp * eye, axis=(-2, -1))[..., None, None] + eps
        a = pp / tr + floor * eye
        a_half, _ = sqrtns(a, power_iters)
        a_q, _ = sqrtns(a_half, power_iters)
        return tr**0.25 * a_q

    def inv_q(a, qm):  # κ-damped S^{-1}=Q Diag(1/(λ+κ)) Qᵀ via the MAINTAINED eigenbasis qm; λ=diag(qmᵀ a qm)
        lam = jnp.maximum(jnp.sum(qm * em("...ik,...kj->...ij", a, qm), axis=-2), 0.0)  # Rayleigh [.., n]
        inv = 1.0 / (lam + floor * (jnp.sum(lam, axis=-1, keepdims=True) + eps))
        return em("...ik,...kj->...ij", qm * inv[..., None, :], bt(qm))

    def power_iter(mat, qv):
        for _ in range(int(power_iters)):
            mq = es("...nk,...k->...n", mat, qv)
            qv = mq / (jnp.linalg.norm(mq, axis=-1, keepdims=True) + eps)
        return qv

    def mclip(m):
        ms1 = msign(m)
        eye = jnp.broadcast_to(jnp.eye(m.shape[-1], dtype=m.dtype), m.shape[:-2] + (m.shape[-1], m.shape[-1]))
        ms2 = msign(em("...ki,...kj->...ij", m, m) - eye)
        return (m + ms1 + em("...ik,...kj->...ij", ms1 - m, ms2)) / 2.0

    out, inn = g.shape[-2], g.shape[-1]
    transpose = out < inn
    g_t = bt(g) if transpose else g
    n_t = bt(n) if transpose else n

    da, db = g_t.shape[-2], g_t.shape[-1]
    # KL-Shampoo whitened Gram update (see _curv_direction_2d); else plain Shampoo outer product.
    if kl_shampoo:
        sb_inv = inv_q(p_r, q_b_in)  # S_b^{-1} via maintained incoming Q_b
        sa_inv = inv_q(p, q_a_in)  # S_a^{-1} via maintained incoming Q_a (Jacobi)
        delta_a = em("...ik,...kj->...ij", em("...ik,...kj->...ij", g_t, sb_inv), bt(g_t)) / db
        delta_b = em("...ik,...kj->...ij", em("...ki,...kj->...ij", g_t, sa_inv), g_t) / da
    else:
        delta_a = em("...ik,...jk->...ij", g_t, g_t)
        delta_b = em("...ki,...kj->...ij", g_t, g_t)
    new_p = rho * p + (1.0 - rho) * delta_a  # P_L (stored UNcorrected)
    # Bias-correct the Gram for the curvature operator: P̂ = P/(1-rho^t) (see _curv_direction_2d). pdiv→1.
    pdiv = 1.0 if bias_t is None else (1.0 - rho ** jnp.asarray(bias_t, new_p.dtype))
    p_c = new_p / pdiv
    new_q = power_iter(p_c, q)
    new_p_r = rho * p_r + (1.0 - rho) * delta_b  # P_R (stored UNcorrected)
    pr_c = new_p_r / pdiv
    new_q_r = power_iter(pr_c, q_r)
    pl4 = pow4(p_c)
    pr4 = pow4(pr_c)

    # EK-FAC augmented eigenvalues (see _curv_direction_2d): batched eigh of the Grams gives the eigenbasis;
    # D = EMA((Q_aᵀG Q_b)²). We solve in the eigenbasis (rotate N in ONCE, solution out ONCE) where the
    # curvature is the ELEMENTWISE d_scale ⊙ X̂ — avoiding two matmuls per apply_curv call.
    # Refine maintained eigenbases one QR step from the NEW Grams (distributed shard_map QR via _qr_refine).
    new_qa = _qr_refine(em("...ik,...kj->...ij", new_p, q_a_in), out_p) if q_a_in is not None else q_a_in
    new_qb = _qr_refine(em("...ik,...kj->...ij", new_p_r, q_b_in), out_p) if q_b_in is not None else q_b_in

    new_D = aug_eig
    if ekfac:
        qa, qb = new_qa, new_qb
        ghat = em("...ik,...kj->...ij", em("...ki,...kj->...ij", qa, g_t), qb)  # Q_aᵀ G Q_b
        base_D = aug_eig if aug_eig is not None else jnp.zeros_like(ghat)
        new_D = rho * base_D + (1.0 - rho) * (ghat * ghat)
        d_hat = new_D / pdiv
        if ekfac_power == "quarter_trace":  # D̂^{1/4}·tr_D^{1/4} (per-stack trace); else "half" = D̂^{1/2}
            trd = jnp.sum(d_hat, axis=(-2, -1), keepdims=True) + eps
            d_scale = jnp.power(d_hat + eps, 0.25) * jnp.power(trd, 0.25)
        else:
            d_scale = jnp.power(d_hat + eps, 0.5)
        n_solve = ghat * 0.0 + em("...ik,...kj->...ij", em("...ki,...kj->...ij", qa, n_t), qb)  # Q_aᵀ N Q_b
        apply_curv = lambda x: d_scale * x  # elementwise in the eigenbasis
        post = lambda x: em("...ik,...kj->...ij", em("...ik,...kj->...ij", qa, x), bt(qb))  # Q_a (·) Q_bᵀ
    else:
        n_solve = n_t
        apply_curv = lambda x: em("...ik,...kj->...ij", em("...ik,...kj->...ij", pl4, x), pr4)
        post = lambda x: x

    ball = constraint == "ball"

    def phi(z):
        return es("...mn,...mn->...", n_solve, z) - 0.5 * lam_coef * es("...mn,...mn->...", z, apply_curv(z))

    def project(y):
        return mclip(y) if ball else msign(y)

    def sel(cond, a, b):
        cf = cond.astype(a.dtype)
        return cf * a + (1.0 - cf) * b

    def bsym(a):  # batched symmetrize (a.T transposes all axes; we want only the trailing two)
        return 0.5 * (a + bt(a))

    cold = msign(n_solve)
    # Warm-start τ across outer steps (per-stack [...] vector). Default carry value returned for the lam=0
    # path; the scan overwrites it with the accepted τ when lam>0.
    tau_final = (es("...mn,...mn->...", n_solve, n_solve) * 0.0) if warm_tau is None else warm_tau

    if solver == "frank_wolfe":
        # Frank-Wolfe (closed-form step, no backtracking, no mclip — see _fw_solve). ~1 msign/step.
        def fw_step(x, _):
            g = n_solve - lam_coef * apply_curv(x)
            s = msign(g)
            d = s - x
            cd = apply_curv(d)
            numer = es("...mn,...mn->...", g, d)
            denom = lam_coef * es("...mn,...mn->...", d, cd)
            alpha = jnp.clip(numer / jnp.where(denom > 1e-12, denom, 1.0), 0.0, 1.0)
            return x + alpha[..., None, None] * d, None

        x = jax.lax.scan(fw_step, cold, None, length=int(inner_steps))[0] if lam_static > 0.0 else cold
        xr = post(x)
        direction = bt(xr) if transpose else xr
        return new_p, new_q, new_p_r, new_q_r, xr, direction, tau_final, new_D, new_qa, new_qb

    if solver == "secular":
        x = _ek_secular_solve(n_solve, apply_curv, lam_coef) if lam_static > 0.0 else cold
        xr = post(x)
        direction = bt(xr) if transpose else xr
        return new_p, new_q, new_p_r, new_q_r, xr, direction, tau_final, new_D, new_qa, new_qb

    if solver == "band_ncg":
        x = _band_ncg_solve(n_solve, apply_curv, lam_coef, msign, em, inner_steps) if lam_static > 0.0 else cold
        xr = post(x)
        direction = bt(xr) if transpose else xr
        return new_p, new_q, new_p_r, new_q_r, xr, direction, tau_final, new_D, new_qa, new_qb

    if solver == "doublenorm_secular":
        md, nd = n_solve.shape[-2], n_solve.shape[-1]
        rmin = min(md, nd)
        x = _ek_doublenorm_solve(n_solve, apply_curv, lam_coef, rmin / md, rmin / nd) if lam_static > 0.0 else cold
        xr = post(x)
        direction = bt(xr) if transpose else xr
        return new_p, new_q, new_p_r, new_q_r, xr, direction, tau_final, new_D, new_qa, new_qb

    if solver in ("orignorm", "orignorm_ncg"):
        cn = lambda M: M / (jnp.sqrt(jnp.sum(M * M, axis=-2, keepdims=True)) + eps)
        rn = lambda M: M / (jnp.sqrt(jnp.sum(M * M, axis=-1, keepdims=True)) + eps)
        row_mode = n_t.shape[-2] <= n_t.shape[-1]
        normalize = rn if row_mode else cn
        if lam_static <= 0.0:
            x = normalize(n_t)
        else:
            rin = lambda Z: em("...ik,...kj->...ij", em("...ki,...kj->...ij", qa, Z), qb)  # rotate into eigenbasis
            curv_orig = lambda X: lam_coef * post(d_scale * rin(X))  # 𝒞[X] in ORIGINAL coords
            if solver == "orignorm":
                cc = lam_coef * jnp.max(d_scale, axis=(-2, -1), keepdims=True) * (1.0 + _GPI_SAFETY)
                x = _orignorm_solve(n_t, curv_orig, normalize(n_t), inner_steps, normalize, cc)
            else:
                ng = normalize(post(rin(n_t) / (lam_coef * d_scale + eps)))
                red = -1 if row_mode else -2
                lmw = lam_coef * jnp.max(d_scale, axis=(-2, -1), keepdims=True)
                x = _orignorm_ncg_solve(n_t, curv_orig, ng, inner_steps, normalize, red, lmw)
                x = _best_phi_orig([normalize(n_t), ng, x], n_t, curv_orig)
        direction = bt(x) if transpose else x
        return new_p, new_q, new_p_r, new_q_r, x, direction, tau_final, new_D, new_qa, new_qb

    if solver == "doublenorm_ncg":
        if lam_static <= 0.0:
            x = n_t
        else:
            md, nd = n_t.shape[-2], n_t.shape[-1]
            rmin = min(md, nd)
            rin = lambda Z: em("...ik,...kj->...ij", em("...ki,...kj->...ij", qa, Z), qb)
            curv_orig = lambda X: lam_coef * post(d_scale * rin(X))
            ng = post(rin(n_t) / (lam_coef * d_scale + eps))  # NG direction (retracted inside the solver)
            lmw = lam_coef * jnp.max(d_scale, axis=(-2, -1), keepdims=True)
            x = _doublenorm_ncg_solve(n_t, curv_orig, ng, inner_steps, rmin / md, rmin / nd, lmw)
        direction = bt(x) if transpose else x
        return new_p, new_q, new_p_r, new_q_r, x, direction, tau_final, new_D, new_qa, new_qb

    if solver in ("gpi", "ncg"):
        if lam_static <= 0.0:
            x = cold
        else:
            ng = _ng_init(n_solve, apply_curv, lam_coef, msign, eps)
            if solver == "gpi":
                x = _gpi_solve(n_solve, apply_curv, ng, lam_coef, inner_steps, msign)
            else:
                x = _ncg_solve(n_solve, apply_curv, ng, lam_coef, inner_steps, msign, em, es, bsym)
            x = _best_phi([cold, ng, x], n_solve, apply_curv, lam_coef, es)
        xr = post(x)
        direction = bt(xr) if transpose else xr
        return new_p, new_q, new_p_r, new_q_r, xr, direction, tau_final, new_D, new_qa, new_qb

    if lam_static > 0.0:
        tau0, beta, cc = 0.5, 0.5, 1e-4
        # Per-stack [...] zero that carries the stack-axis sharding (derived from a reduction over n_t), so the
        # scan/fori_loop scalar carries match the body's (stack-sharded) outputs — a replicated jnp.zeros(lead)
        # would mismatch the carry type under the explicit mesh.
        zlead = es("...mn,...mn->...", n_solve, n_solve) * 0.0

        def line_search(x, d, dd, f0, tau_start):
            def body(_, carry):
                tau, acc = carry
                good = phi(project(x + tau[..., None, None] * d)) >= f0 + cc * tau * dd
                acc = sel(good & (acc == 0.0), tau, acc)
                tau = sel(good, tau, tau * beta)
                return (tau, acc)

            _, acc = jax.lax.fori_loop(0, int(maxbt), body, (tau_start, zlead))
            return acc

        def step(carry, _):
            x, last_tau = carry
            z = n_solve - lam_coef * apply_curv(x)
            grad = z if ball else z - em("...ik,...kj->...ij", x, bsym(em("...ki,...kj->...ij", x, z)))
            d = msign(grad)
            dd = es("...mn,...mn->...", grad, d)
            tau_start = sel(last_tau > 0, jnp.minimum(2.0 * last_tau, tau0), zlead + tau0)
            acc = line_search(x, d, dd, phi(x), tau_start)
            x = project(x + acc[..., None, None] * d)
            return (x, sel(acc > 0, acc, last_tau)), None

        tau_init = (zlead + 0.25) if warm_tau is None else warm_tau
        (x, tau_final), _ = jax.lax.scan(step, (cold, tau_init), None, length=int(inner_steps))
    else:
        x = cold

    xr = post(x)
    direction = bt(xr) if transpose else xr
    return new_p, new_q, new_p_r, new_q_r, xr, direction, tau_final, new_D, new_qa, new_qb


def scale_with_curvature_muon(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    learning_rate=0.02,
    coefficient_type="quintic",
    curvature_beta=0.95,
    curvature_lambda=0.0,
    curvature_alpha=1.0,
    curv_power="sqrt",
    two_sided=False,
    mudam_init=True,
    mudam_steps=5,
    inner_steps=1,
    power_iters=8,
    peak_lr=0.02,
    lambda_tracks_lr=False,
    inner_solver="fixed_point",
    riemannian_maxbt=10,
    riemannian_warm_start=False,
    constraint="stiefel",
    log_phi_every=0,
    hyperball_denom="actual",
):
    steps = int(steps)
    mu = float(momentum)
    rho = float(curvature_beta)
    lam = float(curvature_lambda)  # static peak strength (also the on/off switch)
    alpha = float(curvature_alpha)
    cpow = str(curv_power)
    two_side = bool(two_sided)
    mudam = bool(mudam_init)
    mudam_k = int(mudam_steps)
    peak_lr = float(peak_lr)
    tracks_lr = bool(lambda_tracks_lr)
    solver = str(inner_solver)
    maxbt = int(riemannian_maxbt)
    warm_start = bool(riemannian_warm_start)
    constraint = str(constraint)
    log_phi = int(log_phi_every)
    iso_denom = str(hyperball_denom) == "isometry"

    def _is_linear_weight(layer):
        return isinstance(layer, haliax.nn.Linear) and isinstance(layer.weight, haliax.NamedArray)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        flat = flatten_linear_layers(params)

        def to_p(layer):
            if not _is_linear_weight(layer):
                return layer
            a = layer.weight.array
            m = max(a.shape[-2], a.shape[-1])
            return jnp.broadcast_to(muon_eps * jnp.eye(m, dtype=a.dtype), a.shape[:-2] + (m, m))

        def to_q(layer):
            if not _is_linear_weight(layer):
                return layer
            a = layer.weight.array
            m = max(a.shape[-2], a.shape[-1])
            return jnp.broadcast_to(jnp.ones(m, dtype=a.dtype) / jnp.sqrt(m), a.shape[:-2] + (m,))

        def to_p_r(layer):  # right Gram P_R [..., N, N], N = min(out, in)
            if not _is_linear_weight(layer):
                return layer
            a = layer.weight.array
            nn = min(a.shape[-2], a.shape[-1])
            return jnp.broadcast_to(muon_eps * jnp.eye(nn, dtype=a.dtype), a.shape[:-2] + (nn, nn))

        def to_q_r(layer):
            if not _is_linear_weight(layer):
                return layer
            a = layer.weight.array
            nn = min(a.shape[-2], a.shape[-1])
            return jnp.broadcast_to(jnp.ones(nn, dtype=a.dtype) / jnp.sqrt(nn), a.shape[:-2] + (nn,))

        def to_x(layer):  # carried inner solution, oriented [..., M, N] (M = max, N = min); 0 ⟹ cold start
            if not _is_linear_weight(layer):
                return layer
            a = layer.weight.array
            m, nn = max(a.shape[-2], a.shape[-1]), min(a.shape[-2], a.shape[-1])
            return jnp.zeros(a.shape[:-2] + (m, nn), dtype=a.dtype)

        tm = haliax.tree_util.tree_map
        return ScaleByCurvatureMuonState(
            momentum_buffer=momentum_buffer,
            curvature=tm(to_p, flat, is_leaf=_is_linear_weight),
            power_vec=tm(to_q, flat, is_leaf=_is_linear_weight),
            curvature_r=tm(to_p_r, flat, is_leaf=_is_linear_weight),
            power_vec_r=tm(to_q_r, flat, is_leaf=_is_linear_weight),
            inner_x=tm(to_x, flat, is_leaf=_is_linear_weight),
            count=jnp.zeros((), dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        # Momentum buffer + scale-preserving Nesterov signal N = (1-μ)G + μB.
        buf = jax.tree.map(
            lambda m, g: None if g is None else mu * m + (1.0 - mu) * g,
            state.momentum_buffer,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            signal = jax.tree.map(
                lambda b, g: None if g is None else (1.0 - mu) * g + mu * b,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            signal = buf

        flat_grad = flatten_linear_layers(updates)
        flat_signal = flatten_linear_layers(signal)

        # Effective coefficient: tracks the LR schedule when requested (traced scalar), else the static peak.
        lam_coef = lam * (learning_rate / peak_lr) if tracks_lr else lam

        def per_layer(g_layer, n_layer, p, q, p_r, q_r, xx):
            if not _is_linear_weight(g_layer):
                return n_layer  # passthrough (these are routed to adam/adamh anyway)
            g = g_layer.weight.array
            n = n_layer.weight.array
            fn = lambda gg, nn, pp, qq, ppr, qqr, xi: _curv_direction_2d(
                gg,
                nn,
                pp,
                qq,
                ppr,
                qqr,
                xi,
                rho=rho,
                lam_static=lam,
                lam_coef=lam_coef,
                alpha=alpha,
                steps=steps,
                eps=muon_eps,
                ctype=coefficient_type,
                inner_steps=inner_steps,
                power_iters=power_iters,
                curv_power=cpow,
                mudam_init=mudam,
                mudam_steps=mudam_k,
                two_sided=two_side,
                floor=_POW4_FLOOR,
                inner_solver=solver,
                maxbt=maxbt,
                warm_start=warm_start,
                constraint=constraint,
            )
            if g.ndim == 3:
                new_p, new_q, new_p_r, new_q_r, new_x, phi_traj, x, _tau, _D, _qa, _qb = jax.vmap(fn)(
                    g, n, p, q, p_r, q_r, xx
                )
            else:
                new_p, new_q, new_p_r, new_q_r, new_x, phi_traj, x, _tau, _D, _qa, _qb = fn(g, n, p, q, p_r, q_r, xx)
            new_w = dataclasses.replace(n_layer.weight, array=x)
            # 7-tuple: (direction-layer, P_L, q_L, P_R, q_R, inner_x, phi_traj)
            return (dataclasses.replace(n_layer, weight=new_w), new_p, new_q, new_p_r, new_q_r, new_x, phi_traj)  # type: ignore

        combined = haliax.tree_util.tree_map(
            per_layer,
            flat_grad,
            flat_signal,
            state.curvature,
            state.power_vec,
            state.curvature_r,
            state.power_vec_r,
            state.inner_x,
            is_leaf=_is_linear_weight,
        )

        is_tup = lambda c: isinstance(c, tuple) and len(c) == 7
        flat_dir = jax.tree.map(lambda c: c[0] if is_tup(c) else c, combined, is_leaf=is_tup)
        new_curvature = jax.tree.map(lambda c: c[1] if is_tup(c) else c, combined, is_leaf=is_tup)
        new_power = jax.tree.map(lambda c: c[2] if is_tup(c) else c, combined, is_leaf=is_tup)
        new_curvature_r = jax.tree.map(lambda c: c[3] if is_tup(c) else c, combined, is_leaf=is_tup)
        new_power_r = jax.tree.map(lambda c: c[4] if is_tup(c) else c, combined, is_leaf=is_tup)
        new_inner_x = jax.tree.map(lambda c: c[5] if is_tup(c) else c, combined, is_leaf=is_tup)
        direction = unflatten_linear_layers(signal, flat_dir)

        new_count = state.count + 1
        if log_phi > 0:
            # Sum the inner objective trajectory φ(k=0..K) over all curvature_muon matrices; print every
            # `log_phi` steps so saturation (φ_K ≈ φ_{K-1}) is visible in the training logs.
            phi_leaves = [c[6] for c in jax.tree.leaves(combined, is_leaf=is_tup) if is_tup(c)]
            phi_sum = sum(jnp.sum(pt, axis=tuple(range(pt.ndim - 1))) for pt in phi_leaves)  # [K+1]

            def _emit(_):
                jax.debug.print("[curvmuon] step={s} sum_phi(k=0..K)={p}", s=new_count, p=phi_sum)
                return jnp.int32(0)

            jax.lax.cond(new_count % log_phi == 0, _emit, lambda _: jnp.int32(0), operand=None)

        # Hyperball: constant-Frobenius-norm scale-invariant update (identical to MuonH).
        # Denominator: actual ‖u‖, or (iso_denom) the XᵀX=I norm √min(out,in) — keeps LR calibrated for σ=1
        # but lets the ball's σ<1 shrink the step (‖u‖/√N < 1).
        def scale_invariant_update(p, u):
            if p is None:
                return None
            iso = jnp.sqrt(jnp.asarray(min(p.shape[-2], p.shape[-1]), dtype=p.dtype))  # ‖u‖ at XᵀX=I
            if p.ndim == 2:
                denom = iso if iso_denom else jnp.maximum(jnp.linalg.norm(u), 1e-10)
                new_p = p - learning_rate * u * jnp.linalg.norm(p) / denom
                return new_p / jnp.linalg.norm(new_p) * jnp.linalg.norm(p) - p
            else:
                axes = tuple(range(1, p.ndim))
                p_norm = jnp.sqrt(jnp.sum(jnp.square(p), axis=axes, keepdims=True))
                u_norm = jnp.sqrt(jnp.sum(jnp.square(u), axis=axes, keepdims=True))
                denom = iso if iso_denom else jnp.maximum(u_norm, 1e-10)
                new_p = p - learning_rate * u * p_norm / denom
                new_p_norm = jnp.sqrt(jnp.sum(jnp.square(new_p), axis=axes, keepdims=True))
                return new_p / jnp.maximum(new_p_norm, 1e-10) * p_norm - p

        hyperball_updates = jax.tree_util.tree_map(
            scale_invariant_update, params, direction, is_leaf=lambda x: x is None
        )
        return hyperball_updates, ScaleByCurvatureMuonState(
            momentum_buffer=buf,
            curvature=new_curvature,
            power_vec=new_power,
            curvature_r=new_curvature_r,
            power_vec_r=new_power_r,
            inner_x=new_inner_x,
            count=new_count,
        )

    return optax.GradientTransformation(init_fn, update_fn)
