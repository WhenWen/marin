# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Muon optimizer for models using raw JAX arrays with (fan_in, fan_out) layout,
such as Grug models.

All 2D arrays are routed to Muon, except those whose path contains
'embed', 'lm_head', or 'output' (case-insensitive), which use AdamW.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec
from jax.sharding import reshard
from optax import tree_utils as otu

from levanter.optim.config import OptimizerConfig
from levanter.optim.curvature_muon import _POW4_FLOOR, _curv_direction_2d, curv_direction_batched
from levanter.optim.muon import MuonConfig, ScaleByMuonState
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS, CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

VMAP_REPLICATED = "vmap_replicated"
STACK_BATCH_SHARDED = "stack_batch_sharded"
ORTHOGONALIZATION_LAYOUTS = (VMAP_REPLICATED, STACK_BATCH_SHARDED)


def _target_sharding(array) -> jax.sharding.Sharding | None:
    if array is None or not hasattr(array, "shape"):
        return None

    sharding = getattr(array, "sharding", None)
    if sharding is not None:
        return sharding

    aval = jax.typeof(array)
    return getattr(aval, "sharding", None)


def _batch_sharded_stack_target_pspec(array) -> PartitionSpec | None:
    if array is None or not hasattr(array, "shape") or array.ndim != 3:
        return None

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return None

    mesh_shape = tuple((axis_name, axis_size) for axis_name, axis_size in mesh.shape.items() if axis_size > 1)
    if not mesh_shape:
        return None

    batch_axis = tuple(axis_name for axis_name, _ in mesh_shape)
    batch_shards = math.prod(axis_size for _, axis_size in mesh_shape)
    if array.shape[0] % batch_shards != 0:
        return None

    if len(batch_axis) == 1:
        return PartitionSpec(batch_axis[0], None, None)
    return PartitionSpec(batch_axis, None, None)


@OptimizerConfig.register_subclass("grug_muon")
@dataclass(frozen=True)
class GrugMuonConfig(MuonConfig):
    """
    Muon optimizer for models that use raw JAX arrays in (fan_in, fan_out) layout.

    Routing rules:
    - 2D arrays whose path does NOT contain 'embed', 'lm_head', or 'output' -> Muon
    - Everything else -> AdamW
    """

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muon_transform():
                components = []
                components.append(
                    _grug_scale_with_muon(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        self.use_kimi_scaling,
                        self.coefficient_type,
                    )
                )
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                components.append(_match_update_sharding())
                return optax.chain(*components)

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
                if adam_weight_decay > 0:
                    components.append(optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params, use_kimi_scaling=True):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "embed" in path_lower or "lm_head" in path_lower or "output" in path_lower:
                return "adamw"
            elif hasattr(param, "ndim") and param.ndim == 2:
                return "muon"
            elif (
                hasattr(param, "ndim")
                and param.ndim == 3
                and ("w_up_gate" in path_lower or "w_gate_up" in path_lower or "w_down" in path_lower)
            ):
                return "muon"
            else:
                return "adamw"

        return jax.tree.map(mask_fn, params, paths)


def _grug_scale_with_muon(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    use_kimi_scaling=False,
    coefficient_type="quintic",
    orthogonalization_layout: str = STACK_BATCH_SHARDED,
):
    """Muon gradient transformation for raw arrays with matrix-shaped trailing dimensions."""
    steps = int(steps)
    if orthogonalization_layout not in ORTHOGONALIZATION_LAYOUTS:
        raise ValueError(
            f"Unknown orthogonalization_layout={orthogonalization_layout!r}. "
            f"Expected one of {ORTHOGONALIZATION_LAYOUTS!r}."
        )

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        return ScaleByMuonState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates = buf

        def transform_array(x, param):
            if not hasattr(x, "ndim") or x.ndim not in (2, 3):
                return x
            if x.ndim == 2:
                updated = _zeropower_via_newtonschulz_replicated(
                    x,
                    steps,
                    muon_eps,
                    coefficient_type,
                    None,
                )
            else:
                if orthogonalization_layout == VMAP_REPLICATED:
                    updated = jax.vmap(
                        lambda matrix: _zeropower_via_newtonschulz_replicated(
                            matrix,
                            steps,
                            muon_eps,
                            coefficient_type,
                            None,
                        )
                    )(x)
                else:
                    stack_target_pspec = _batch_sharded_stack_target_pspec(param)
                    if stack_target_pspec is None:
                        updated = jax.vmap(
                            lambda matrix: _zeropower_via_newtonschulz_replicated(
                                matrix,
                                steps,
                                muon_eps,
                                coefficient_type,
                                None,
                            )
                        )(x)
                    else:
                        updated = _zeropower_via_newtonschulz_batched_stack_sharded(
                            x,
                            steps,
                            muon_eps,
                            coefficient_type,
                            stack_target_pspec,
                        )

            fan_in, fan_out = updated.shape[-2:]
            if not use_kimi_scaling:
                scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
            else:
                scale = 0.2 * jnp.sqrt(jnp.maximum(fan_in, fan_out))
            updated *= scale
            return updated

        if params is None:
            updates = jax.tree.map(lambda x: transform_array(x, None), updates)
        else:
            updates = jax.tree.map(transform_array, updates, params)

        return updates, ScaleByMuonState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


class _GrugCurvState(NamedTuple):
    momentum_buffer: optax.Updates
    curvature: optax.Updates  # P_L per matrix [..., M, M]
    power_vec: optax.Updates  # q_L [..., M]
    curvature_r: optax.Updates  # P_R [..., N, N]
    power_vec_r: optax.Updates  # q_R [..., N]
    inner_x: optax.Updates  # carried inner solution [..., M, N]
    inner_tau: optax.Updates  # carried line-search step-size τ [...] (warm-started across outer steps)
    aug_eig: optax.Updates  # EK-FAC augmented eigenvalues D [..., M, N] (per-coordinate, in the eigenbasis)
    curv_qa: optax.Updates  # maintained left eigenbasis Q_a [..., M, M] (refined via QR each step)
    curv_qb: optax.Updates  # maintained right eigenbasis Q_b [..., N, N]
    count: jax.Array  # scalar step counter for Adam-style bias correction of N (momentum) and P (Gram)


class _CurvOut(NamedTuple):
    """Per-matrix curvature output, packed as one leaf so it can be split back into the direction +
    the 5 state trees. A dedicated type (not a bare 6-tuple) is essential: ``is_leaf`` detection by
    ``len(c) == 6`` collides with model containers of length 6 (e.g. a 6-layer ``blocks`` tuple),
    which would make the split grab ``blocks[0]`` and mangle the tree."""

    direction: jax.Array
    curvature: jax.Array
    power_vec: jax.Array
    curvature_r: jax.Array
    power_vec_r: jax.Array
    inner_x: jax.Array
    inner_tau: jax.Array  # carried line-search step-size τ (warm-started across outer steps)
    aug_eig: jax.Array  # EK-FAC augmented eigenvalues D
    qa: jax.Array  # refined left eigenbasis Q_a
    qb: jax.Array  # refined right eigenbasis Q_b


def _grug_scale_with_curvature_muon(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    coefficient_type="quintic",
    curvature_beta=0.95,
    curvature_lambda=0.3,
    inner_steps=10,
    riemannian_maxbt=10,
    two_sided=True,
    constraint="stiefel",
    curv_power="sqrt",
    power_iters=8,
    inner_solver="riemannian_muon",
    kl_shampoo=False,
    ekfac=False,
    ekfac_power="half",
):
    """Curvature-corrected Muon for raw grug arrays (matrix trailing dims). Drop-in for
    _grug_scale_with_muon: replaces msign(N) with the Riemannian curvature inner-solve, applies the same
    sqrt(fan_out/fan_in) scale; the hyperball is applied by the caller. curvature_lambda=0 ⟹ plain MuonH.
    Reuses the tested _curv_direction_2d (replicated NS msign — fine for d512)."""
    rho = float(curvature_beta)
    lam = float(curvature_lambda)
    steps = int(steps)
    none_leaf = lambda x: x is None

    def _ismat(x):
        return hasattr(x, "ndim") and x.ndim in (2, 3)

    # KL-Shampoo whitens by the OTHER factor's inverse, so the Grams must start at identity (not muon_eps·I):
    # an eps·I init makes S^{-1} ~ 1/eps and the first whitened update explodes. I = "no preconditioning yet".
    p_init = 1.0 if kl_shampoo else muon_eps

    def _mk(x, kind):
        if not _ismat(x):
            return None
        m, n, lead = max(x.shape[-2], x.shape[-1]), min(x.shape[-2], x.shape[-1]), x.shape[:-2]
        if kind == "p":
            return jnp.broadcast_to(p_init * jnp.eye(m, dtype=x.dtype), lead + (m, m))
        if kind == "q":
            return jnp.broadcast_to(jnp.ones(m, dtype=x.dtype) / jnp.sqrt(m), lead + (m,))
        if kind == "pr":
            return jnp.broadcast_to(p_init * jnp.eye(n, dtype=x.dtype), lead + (n, n))
        if kind == "qr":
            return jnp.broadcast_to(jnp.ones(n, dtype=x.dtype) / jnp.sqrt(n), lead + (n,))
        if kind == "tau":
            return jnp.zeros(lead, dtype=x.dtype)  # per-stack [...] step-size carry
        if kind == "d":
            return jnp.zeros(lead + (m, n), dtype=x.dtype)  # EK-FAC augmented eigenvalues [.., max, min] (g_t shape)
        if kind == "qa":
            return jnp.broadcast_to(jnp.eye(m, dtype=x.dtype), lead + (m, m))  # maintained eigenbasis, init I
        if kind == "qb":
            return jnp.broadcast_to(jnp.eye(n, dtype=x.dtype), lead + (n, n))
        return jnp.zeros(lead + (m, n), dtype=x.dtype)

    def init_fn(params):
        tm = jax.tree.map
        return _GrugCurvState(
            otu.tree_zeros_like(params),
            tm(lambda x: _mk(x, "p"), params, is_leaf=none_leaf),
            tm(lambda x: _mk(x, "q"), params, is_leaf=none_leaf),
            tm(lambda x: _mk(x, "pr"), params, is_leaf=none_leaf),
            tm(lambda x: _mk(x, "qr"), params, is_leaf=none_leaf),
            tm(lambda x: _mk(x, "x"), params, is_leaf=none_leaf),
            tm(lambda x: _mk(x, "tau"), params, is_leaf=none_leaf),
            tm(lambda x: _mk(x, "d"), params, is_leaf=none_leaf),
            tm(lambda x: _mk(x, "qa"), params, is_leaf=none_leaf),
            tm(lambda x: _mk(x, "qb"), params, is_leaf=none_leaf),
            jnp.zeros([], jnp.int32),
        )

    def update_fn(updates, state, params=None, *, lam_scale=1.0, **_extra):
        # ``lam_scale`` (default 1.0, traced) multiplies the curvature coefficient per step so λ can track a
        # schedule (e.g. lam_scale = lr_t/peak_lr). lam_static (the static peak, gates the inner-solve branch)
        # is kept as the constant ``lam``; lam_coef (the numeric penalty weight) becomes lam·lam_scale.
        lam_coef = lam * lam_scale
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g, state.momentum_buffer, updates, is_leaf=none_leaf
        )
        signal = (
            jax.tree.map(lambda m, g: None if g is None else momentum * m + g, buf, updates, is_leaf=none_leaf)
            if nesterov
            else buf
        )

        # Adam-style bias correction. N (momentum): the sum-convention nesterov buffer is converted to a
        # debiased MEAN via factor (1-momentum)/(1-momentum^t). msign is scale-invariant, so this leaves the
        # cold-start direction msign(N) identical, but fixes the N (~g) vs P^{1/4} (~√g) scale ratio so the
        # curvature penalty/⟨N,X⟩ ratio is ~lambda (dimensionless, gradient-scale-independent). P (Gram) is
        # debiased inside the solve by 1/(1-rho^t) (see bias_t). The step counter t starts at 1.
        t = state.count + 1
        n_corr = (1.0 - momentum) / (1.0 - momentum ** t.astype(jnp.float32))
        signal = jax.tree.map(lambda s: None if s is None else s * n_corr, signal, is_leaf=none_leaf)

        has_mesh = not jax.sharding.get_abstract_mesh().empty

        def per(g, n, p, q, pr, qr, xx, xt, xd, xqa, xqb):
            if not _ismat(g):
                return n  # passthrough (non-matrix params unchanged here)
            if g.ndim == 3:
                # Expert-stacked weights: batched-einsum curvature solve (curv_direction_batched, no vmap).
                # Reshard inputs to the STACK-SHARDED layout (stack axis sharded as the grad's, matrix dims
                # replicated) so experts stay distributed across the mesh; every contraction is over a
                # replicated matrix dim, so the propagator keeps the stack axis sharded with no ambiguity and
                # no vmap (⟹ no unmapped_aval). out_p pins the matrix einsums to that layout.
                out_p = None
                if has_mesh:
                    lead = jax.typeof(g).sharding.spec[0]
                    mp = lambda a: reshard(a, PartitionSpec(lead, None, None))
                    vp = lambda a: reshard(a, PartitionSpec(lead, None))
                    g, n, p, pr, xx, xd, xqa, xqb = mp(g), mp(n), mp(p), mp(pr), mp(xx), mp(xd), mp(xqa), mp(xqb)
                    q, qr = vp(q), vp(qr)
                    xt = reshard(xt, PartitionSpec(lead))  # carried τ is per-stack [E]
                    out_p = PartitionSpec(lead, None, None)
                np_, nq, npr, nqr, nx, d, ntau, nd, nqa, nqb = curv_direction_batched(
                    g,
                    n,
                    p,
                    q,
                    pr,
                    qr,
                    rho=rho,
                    lam_coef=lam_coef,
                    lam_static=lam,
                    steps=steps,
                    eps=muon_eps,
                    ctype=coefficient_type,
                    inner_steps=inner_steps,
                    power_iters=power_iters,
                    floor=_POW4_FLOOR,
                    maxbt=riemannian_maxbt,
                    constraint=constraint,
                    out_p=out_p,
                    bias_t=t,
                    warm_tau=xt,
                    solver=inner_solver,
                    kl_shampoo=kl_shampoo,
                    ekfac=ekfac,
                    aug_eig=xd,
                    ekfac_power=ekfac_power,
                    q_a_in=xqa,
                    q_b_in=xqb,
                )
            else:
                # 2-D dense matrix (attn / shared / gated-norm): per-matrix solve, replicate inner so the NS
                # constraint + Gram matmuls don't contract over a model-sharded dim (shard_ns=False).
                np_, nq, npr, nqr, nx, _pt, d, ntau, nd, nqa, nqb = _curv_direction_2d(
                    g,
                    n,
                    p,
                    q,
                    pr,
                    qr,
                    xx,
                    rho=rho,
                    lam_static=lam,
                    lam_coef=lam_coef,
                    alpha=1.0,
                    steps=steps,
                    eps=muon_eps,
                    ctype=coefficient_type,
                    inner_steps=inner_steps,
                    power_iters=power_iters,
                    curv_power=curv_power,
                    mudam_init=False,
                    mudam_steps=5,
                    two_sided=two_sided,
                    floor=_POW4_FLOOR,
                    inner_solver=inner_solver,
                    maxbt=riemannian_maxbt,
                    warm_start=False,
                    constraint=constraint,
                    shard_ns=False,
                    bias_t=t,
                    warm_tau=xt,
                    kl_shampoo=kl_shampoo,
                    ekfac=ekfac,
                    aug_eig=xd,
                    ekfac_power=ekfac_power,
                    q_a_in=xqa,
                    q_b_in=xqb,
                )
            fan_in, fan_out = d.shape[-2:]
            return _CurvOut(
                d * jnp.sqrt(jnp.maximum(1.0, fan_out / fan_in)), np_, nq, npr, nqr, nx, ntau, nd, nqa, nqb
            )

        comb = jax.tree.map(
            per,
            updates,
            signal,
            state.curvature,
            state.power_vec,
            state.curvature_r,
            state.power_vec_r,
            state.inner_x,
            state.inner_tau,
            state.aug_eig,
            state.curv_qa,
            state.curv_qb,
            is_leaf=none_leaf,
        )
        is_out = lambda c: isinstance(c, _CurvOut)
        pick = lambda i: jax.tree.map(lambda c: c[i] if is_out(c) else c, comb, is_leaf=is_out)
        return pick(0), _GrugCurvState(
            buf, pick(1), pick(2), pick(3), pick(4), pick(5), pick(6), pick(7), pick(8), pick(9), t
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _match_update_sharding():
    """Ensure updates inherit the parameter sharding expected by apply_updates."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        def match_sharding(update, param):
            if update is None:
                return None
            target_sharding = _target_sharding(param)
            if target_sharding is None:
                return update
            return jax.sharding.reshard(update, target_sharding)

        updates = jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _zeropower_via_newtonschulz_replicated(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
    target_pspec: PartitionSpec | None = None,
) -> jax.Array:
    """Legacy Grug Muon orthogonalization that fully replicates each matrix.

    Replicates the array across devices before iterating to avoid sharding
    ambiguities in the X @ X.T contractions. The caller is responsible for
    restoring the final parameter layout. Kept for A/B benchmarking.
    """
    P = PartitionSpec
    assert X.ndim == 2
    del target_pspec  # Kept for signature parity with the other Newton-Schulz helpers.

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    has_mesh = not jax.sharding.get_abstract_mesh().empty
    if has_mesh:
        X = reshard(X, P(None, None))
    X = X / (jnp.linalg.norm(X) + eps)

    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        out_sharding = P(None, None) if has_mesh else None
        A = jnp.einsum("ik,jk->ij", X, X, out_sharding=out_sharding)
        B = b * A + c * jnp.einsum("ik,kj->ij", A, A, out_sharding=out_sharding)
        X = a * X + jnp.einsum("ik,kj->ij", B, X, out_sharding=out_sharding)

    if transpose:
        X = X.T

    return X


def _zeropower_via_newtonschulz_batched_stack_sharded(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
    target_pspec: PartitionSpec | None = None,
) -> jax.Array:
    """Run Newton-Schulz on a stacked batch of matrices with only the batch axis sharded."""
    assert X.ndim == 3

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    has_mesh = not jax.sharding.get_abstract_mesh().empty
    X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) + eps)

    transpose = False
    if X.shape[-2] > X.shape[-1]:
        X = jnp.swapaxes(X, -1, -2)
        transpose = True

    if target_pspec is None:
        target_pspec = _batch_sharded_stack_target_pspec(X)

    if has_mesh and target_pspec is not None:
        X = reshard(X, target_pspec)

    X_out_sharding = target_pspec if (has_mesh and target_pspec is not None) else None
    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        A = jnp.einsum("...ik,...jk->...ij", X, X, out_sharding=X_out_sharding)
        B = b * A + c * jnp.einsum("...ik,...kj->...ij", A, A, out_sharding=X_out_sharding)
        X = a * X + jnp.einsum("...ik,...kj->...ij", B, X, out_sharding=X_out_sharding)

    if transpose:
        X = jnp.swapaxes(X, -1, -2)

    return X
