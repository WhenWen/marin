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


def _pow4_inv_quarter(p, iters, floor, eps):
    """(P^{1/4}, P^{-1/4}) for SPD P, via two trace-normalized coupled-sqrt-NS passes (matmul-only).

    A = P/trace(P) has spectrum ≤ 1 (trace ≥ λ_max), so neither sqrt-NS pass can diverge. First pass →
    A^{1/2}; second pass on A^{1/2} → A^{1/4} (Y) and, with a spectrum floor, the bounded/saturating
    A^{-1/4} (Z). Rescale by trace^{±1/4}. The 4th-root inverse is gentler than P^{-1/2} (smaller exponent).
    """
    tr = jnp.trace(p) + eps
    a = p / tr
    eye = jnp.eye(p.shape[0], dtype=p.dtype)
    a_half, _ = _matrix_sqrt_ns(a, iters)  # A^{1/2}
    a_quarter, _ = _matrix_sqrt_ns(a_half, iters)  # A^{1/4} (forward; no floor needed)
    _, a_inv_quarter = _matrix_sqrt_ns(a_half + floor * eye, iters)  # ≈ A^{-1/4}, floored ⟹ bounded
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


def _mclip(m, steps, eps, ctype):
    """Project m onto the spectral-norm ball {‖·‖₂ ≤ 1} (clip σ → min(σ, 1)) — matmul-only, no SVD.

    Uses the "mclip2" identity   mclip(m) = (m + msign(m) + (msign(m) − m)·msign(mᵀm − I)) / 2,
    which for m = UΣVᵀ gives U·min(Σ,1)·Vᵀ exactly (msign(mᵀm−I) = V·sign(Σ²−1)·Vᵀ). Benchmarks vs SVD:
    exact in fp32 across σ∈[0,57] (MAE ~1e-4); most bf16-robust of the msign-based variants (σ_out ≤ 1.01
    in-context). One full msign [M,N] + one smaller msign [N,N]. (SVD truncation would be exact too but is
    the only non-matmul op — costly on TPU at ~maxbt·K calls/step.)
    """
    ms1 = zeropower_via_newtonschulz5(m, steps=steps, eps=eps, coefficient_type=ctype)
    gram = m.T @ m - jnp.eye(m.shape[-1], dtype=m.dtype)
    ms2 = zeropower_via_newtonschulz5(gram, steps=steps, eps=eps, coefficient_type=ctype)
    return (m + ms1 + (ms1 - m) @ ms2) / 2.0


def _riem_solve(
    n_t, apply_curv, x0, lam_coef, inner_steps, maxbt, steps, eps, ctype, constraint, tau0=0.5, beta=0.5, c=1e-4
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
            return _mclip(y, steps, eps, ctype)
        return zeropower_via_newtonschulz5(y, steps=steps, eps=eps, coefficient_type=ctype)

    def line_search(x, d, dd, f0, tau_start):
        # largest accepted τ over {τ_start·βʲ}; fori_loop body compiles once (no maxbt unroll).
        def body(_, carry):
            tau, acc = carry
            good = phi(project(x + tau * d)) >= f0 + c * tau * dd
            acc = jnp.where(good & (acc == 0.0), tau, acc)
            tau = jnp.where(good, tau, tau * beta)
            return (tau, acc)

        _, acc = jax.lax.fori_loop(0, int(maxbt), body, (tau_start, jnp.zeros((), x.dtype)))
        return acc

    def step(carry, _):
        x, last_tau = carry
        z = n_t - lam_coef * apply_curv(x)  # Euclidean ascent gradient ∇φ
        grad = z if ball else z - x @ _sym(x.T @ z)  # ball: full gradient; stiefel: tangent projection
        d = zeropower_via_newtonschulz5(grad, steps=steps, eps=eps, coefficient_type=ctype)
        dd = jnp.sum(grad * d)  # ⟨grad, D⟩ ≥ 0 (directional derivative)
        # Warm-start τ near the previous step's accepted value (allow ×2 growth, cap τ₀); fall back to τ₀ if
        # none accepted last step. The accepted step changes little between inner iterations, so a warm τ₀
        # brackets it in ~2-3 backtracks (toy: maxbt=3 warm ≡ maxbt=10 cold at every λ) — ~2.5× cheaper.
        tau_start = jnp.where(last_tau > 0, jnp.minimum(2.0 * last_tau, tau0), jnp.asarray(tau0, x.dtype))
        acc = line_search(x, d, dd, phi(x), tau_start)
        x = project(x + acc * d)
        return (x, jnp.where(acc > 0, acc, last_tau)), phi(x)

    init = (x0, jnp.asarray(0.25, x0.dtype))
    (x_final, _), phis = jax.lax.scan(step, init, None, length=int(inner_steps))  # scan body compiles once
    phi_traj = jnp.concatenate([phi(x0)[None], phis])  # [K+1]
    return x_final, phi_traj


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
):
    """One matrix. g, n: [out, in]; p/q = left Gram P_L [M,M] + power vec; p_r/q_r = right Gram P_R [N,N] + vec.

    one-sided (two_sided=False): X = msign(N + λ(α√e_max·I − C)X), C = P_L^{1/2} (sqrt) or P_L/√e_max (linear);
    warm start msign(P_L^{-1/2} N) (Mudam q_k).
    two-sided (two_sided=True): X = msign(N + λ(α(e_L·e_R)^{1/4}·X − P_L^{1/4} X P_R^{1/4})); warm start the
    Shampoo direction msign(P_L^{-1/4} N P_R^{-1/4}). 1/4+1/4 = 1/2 total power ⟹ same ~G units, λ dimensionless.

    inner_solver="riemannian_muon": gradient ascent with Armijo line search, cold start msign(N) (optionally
    warm-started from carried inner_x). One- or two-sided curvature; constraint="stiefel" (msign retraction)
    or "ball" (XᵀX⪯I, mclip/SVD projection). Robust across λ, K-stable.
    Returns (new_p, new_q, new_p_r, new_q_r, new_inner_x, phi_traj, direction); phi_traj[k]=φ at inner step k.
    """
    out, inn = g.shape
    transpose = out < inn
    g_t = g.T if transpose else g  # [M, N], M ≥ N
    n_t = n.T if transpose else n

    new_p = rho * p + (1.0 - rho) * (g_t @ g_t.T)  # P_L [M, M]
    new_q, emax = _power_iter(new_p, q, power_iters, eps)
    eye = jnp.eye(new_p.shape[0], dtype=new_p.dtype)

    # Right Gram P_R + its 1/4 powers (two-sided), or the one-sided curvature matrix C.
    if two_sided:
        new_p_r = rho * p_r + (1.0 - rho) * (g_t.T @ g_t)  # P_R [N, N]
        new_q_r, emax_r = _power_iter(new_p_r, q_r, power_iters, eps)
        pl4, plinv4 = _pow4_inv_quarter(new_p, power_iters, floor, eps)  # P_L^{1/4}, P_L^{-1/4}
        pr4, prinv4 = _pow4_inv_quarter(new_p_r, power_iters, floor, eps)  # P_R^{1/4}, P_R^{-1/4}
    else:
        new_p_r, new_q_r = p_r, q_r
        se = jnp.sqrt(emax) + eps
        if curv_power == "sqrt":
            # tr(P) ≥ λ_max ⟹ P/tr spectrum ≤ 1 ⟹ NS sqrt can't diverge. C = √tr·(P/tr)^{1/2} = P^{1/2}.
            tr = jnp.trace(new_p) + eps
            y_half, _ = _matrix_sqrt_ns(new_p / tr, power_iters)
            curv = jnp.sqrt(tr) * y_half
        else:
            curv = new_p / se  # P/√e_max

    phi_traj = jnp.zeros((int(inner_steps) + 1,), dtype=n_t.dtype)  # default; only riemannian fills it

    if inner_solver == "riemannian_muon":
        # Curvature operator 𝒞X: two-sided P_L^{1/4} X P_R^{1/4}, else one-sided C·X.
        apply_curv = (lambda X: pl4 @ X @ pr4) if two_sided else (lambda X: curv @ X)
        # Cold start msign(N) (λ=0 optimum; best init for small λ). Optional warm start from the carried X.
        cold = zeropower_via_newtonschulz5(n_t, steps=steps, eps=eps, coefficient_type=ctype)
        if lam_static > 0.0:
            if warm_start:
                x0 = jnp.where(jnp.linalg.norm(inner_x) > eps, inner_x, cold)
            else:
                x0 = cold
            x, phi_traj = _riem_solve(n_t, apply_curv, x0, lam_coef, inner_steps, maxbt, steps, eps, ctype, constraint)
        else:
            x = cold
        return new_p, new_q, new_p_r, new_q_r, x, phi_traj, (x.T if transpose else x)

    # --- fixed-point inner solver ---
    if two_sided:
        if mudam_init:
            x = zeropower_via_newtonschulz5(plinv4 @ n_t @ prinv4, steps=steps, eps=eps, coefficient_type=ctype)
        else:
            x = zeropower_via_newtonschulz5(n_t, steps=steps, eps=eps, coefficient_type=ctype)
        if lam_static > 0.0:
            shift2 = alpha * (emax * emax_r) ** 0.25  # = max singular value of X ↦ P_L^{1/4} X P_R^{1/4}
            for _ in range(int(inner_steps)):
                arg = n_t + lam_coef * (shift2 * x - pl4 @ x @ pr4)
                x = zeropower_via_newtonschulz5(arg, steps=steps, eps=eps, coefficient_type=ctype)
        return new_p, new_q, new_p_r, new_q_r, inner_x, phi_traj, (x.T if transpose else x)

    x = (
        _mudam_direction(n_t, new_p, mudam_steps, eps)
        if mudam_init
        else zeropower_via_newtonschulz5(n_t, steps=steps, eps=eps, coefficient_type=ctype)
    )
    if lam_static > 0.0:
        operator = lam_coef * (alpha * se * eye - curv)  # PSD for α≥1
        for _ in range(int(inner_steps)):
            x = zeropower_via_newtonschulz5(n_t + operator @ x, steps=steps, eps=eps, coefficient_type=ctype)
    return new_p, new_q, new_p_r, new_q_r, inner_x, phi_traj, (x.T if transpose else x)


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
                new_p, new_q, new_p_r, new_q_r, new_x, phi_traj, x = jax.vmap(fn)(g, n, p, q, p_r, q_r, xx)
            else:
                new_p, new_q, new_p_r, new_q_r, new_x, phi_traj, x = fn(g, n, p, q, p_r, q_r, xx)
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
