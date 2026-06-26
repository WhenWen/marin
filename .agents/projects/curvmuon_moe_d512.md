# Curvature-corrected Muon on MoE May-Arch (d512)

Goal: port the curvature-corrected Riemannian-Muon inner solve onto the **MoE May-Arch** (PR #6153,
branch `moe_may_pr`, worktree `marin-mayarch`) and test at **d512**, instead of qwen3-130m.

## What carries over from the qwen3 study (use these defaults)
- **Keep the standard hyperball normalization (`÷‖u‖`).** The "isometry" denom (`÷√N`) HURT by ~+0.011 bpb
  on qwen3 (actual 1.1659 vs iso 1.1772 at λ=0.3). Do NOT remove update normalization.
- **Inner solver = riemannian_muon** (Armijo line search), NOT the fixed point (K-unstable).
- **warm-τ + maxbt=3** line search: precision-identical to maxbt=10, ~2.5× cheaper. Roll loops (lax.scan/
  fori_loop) — unrolled K×maxbt graph caused a ~540s first-step compile (preemption death-loop).
- **K=5** (K=3 was too aggressive: fast-config 1.184 worst; that run also had the iso confound).
- **warm-init** (carry inner X across outer steps) helps the **two-sided ball** at small K (16× lower regret
  at K=3) but NOT one-sided. Test as a variant.
- qwen3 verdict: curvature ~NEUTRAL at best (1.1659 @λ=0.3 ≈ MuonH 1.1661). d512/MoE may differ — that's the test.

## Integration (the real work)
The MoE does NOT use levanter `CurvatureMuonConfig`. Its optimizer lives in
`experiments/grug/moe/optimizer.py`:
- `_grug_scale_with_muon` → core msign(N) direction on raw matrix-trailing-dim arrays.  ← **insert curvature here**
- `scale_with_grug_muonh` → wraps + `_scale_invariant_hyperball_updates` (the hyperball).
- `GrugMoeMuonHConfig` / `GrugMoeAmuseConfig` (AMUSE schedule-free) → the configs.
- d512 launcher: `experiments/grug/moe/amuse_mayarch_d512.py` builds `GrugMoeAmuseConfig` (MuonH+AMUSE) via
  `heuristic_v2.build_muonh_config`. d512 cell: bs=32, steps=10_980 (compute-optimal).

Steps:
1. Add a curvature inner-solve to `_grug_scale_with_muon` (port `_curv_direction_2d`/`_riem_solve` from
   `lib/levanter/.../curvature_muon.py` — already copied to this worktree for reference): per matrix, EMA
   P_L (+P_R two-sided), power-iter e_max, C=P^{1/2} (or P^{1/4} both sides), Riemannian ascent + warm-τ
   line search + mclip (matmul mclip2) for ball. Gate on `curvature_lambda` (0 ⟹ plain MuonH = no-op).
   Extra optimizer STATE: per-matrix P_L, q_L, (P_R, q_R), inner_x, scalar count. NOTE MoE arrays have
   matrix-shaped TRAILING dims (experts stacked) — vmap over leading dims like the levanter g.ndim==3 path.
2. Add curvature knobs to `GrugMoeMuonHConfig` + `GrugMoeAmuseConfig`: curvature_lambda, curvature_beta,
   inner_steps(K=5), riemannian_maxbt(3), inner_solver, two_sided, constraint, curv_power.
3. d512 experiment: add curvature env knobs to `amuse_mayarch_d512.py` (or a sibling), default λ=0.
4. Smoke (CPU): MoE optimizer builds + a couple steps finite, λ=0 ≡ MuonH bit-identical.
5. **FIRST: two head-to-head d512 runs (user directive):**
   (a) standard **MuonH** (λ=0), (b) **two-sided curvature, SLOW/SAFE**: cold init (msign(N)), K=10, full
   Riemannian ascent, maxbt=10 (full line search), actual-denom (÷‖u‖). λ=0.3 (qwen3 optimum) for (b).
   Goal: does curvature help on MoE d512 *at all* before any speedup. Compare FINAL c4_en/bpb.
   THEN (if it helps): sweep λ, add the speedups (warm-τ+maxbt3, K=5, warm-init/ball).

## Status
- [x] curvature_muon.py copied to marin-mayarch/lib/levanter (imports OK) — reference only; MoE uses its own optimizer.
- [ ] port curvature into `_grug_scale_with_muon` + configs
- [ ] d512 experiment + smoke
- [ ] launch d512 sweep
