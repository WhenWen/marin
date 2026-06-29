# EK-FAC Curvature-Corrected Muon — Launch Plan

Concrete plan for the curvature-corrected Muon inner-solver study on **MoE may-arch d512**
(PR #6153 baseline). All variants share the EK-FAC eigenbasis + second-moment (`d̂`) machinery and
differ only in **constraint geometry + inner solver** (selected via `CURV_SOLVER`). Launch via
`experiments/grug/moe/launch_curvature_variants.sh` (env-knob reference + ready sweep blocks there).

## Fixed setup (all runs)
- **Cluster/region:** marin iris, **us-east5**, **v5p-8 preemptible** (data mirrored east5; `MARIN_PREFIX=gs://marin-us-east5`).
- **Launch:** `iris job run --no-wait --preemptible --region us-east5 --extra tpu ... -- python -m experiments.grug.moe.launch`
  (CPU launcher → submits the TPU training job via `executor_main`). **Tee output to a file** — job ids get eaten by shell buffering otherwise.
- **Optimizer base:** `GrugMoeMuonHConfig` (standard **linear-decay** schedule, NOT AMUSE), `GRUG_EP=1` (exact MoE), `COEFF_TYPE=polar_express`, `BACKEND_STEPS=8`.
- **Curvature shared:** `CURV_TWO_SIDED=1`, `CURV_CONSTRAINT=stiefel`, `CURV_EKFAC=1`, `CURV_LAMBDA_TRACKS_LR=1` (λ tracks LR: `λ_t = CURV_LAMBDA·lr_t/peak`).
- **wandb:** project `curv-muon`. **Metric for comparison:** `eval/paloma/c4_en/bpb` at the **FINAL step**, vs the plain **MuonH** baseline (do NOT compare mid-run or train/loss).

## Variants and proper K (from convergence analysis)
| `CURV_SOLVER` | constraint geometry | solver | proper `CURV_K` | `EKFAC_POWER` |
|---|---|---|---|---|
| `ncg` | Stiefel XᵀX=I, quadratic curvature | Newton-CG | **4** | half / quarter_trace |
| `secular` | EK-basis single row/col-norm | closed-form secular (exact) | **1** | (any; minor) |
| `doublenorm_secular` | EK-basis double norm | additive-Sinkhorn dual (exact) | **1** | (any) |
| `orignorm_ncg` | original-coord single norm | product-sphere Newton-CG | **6** | half / quarter_trace |
| `doublenorm_ncg` | original-coord double norm | Sinkhorn-retraction NCG | **6** | half / quarter_trace |
| `band_ncg` | Frobenius sphere + soft σ∈[√2/2, √2] | adaptive-τ projected CG | **20** | half / quarter_trace |
| `sqrt_ncg` | Stiefel, scale-invariant √-penalty | Newton-CG (rank-1 corrected) | **12** | **ignored** (uses raw `d̂`) |

Notes / rationale:
- `EKFAC_POWER`: `half` = `d̂^{1/2}` (natural-gradient curvature); `quarter_trace` = `d̂^{1/4}·tr^{1/4}`
  (reproduces the `P^{1/4}` shape; its `tr^{1/4}` factor inflates effective λ ≈ 4.6× — so qt at a given λ is much stronger than half).
- `sqrt_ncg`: the √ is **outside**, so the curvature inside is the **raw second moment `d̂`** (units G²) ⟹ `√(tr XᵀD̂X)~G` matches `⟨N,X⟩~G` and **α (=`CURV_LAMBDA`) is dimensionless ~O(1)**. `EKFAC_POWER` has no effect here.
- `band_ncg`: needs the **adaptive trust-region** line search (carry step-scale τ, shrink ×0.25 on keep-Y); a fixed 2-point backtrack **plateaus** (half stalls ~2%, qt frozen). Soft band [√2/2,√2] ⟹ srank ≥ R/2 (anti-collapse).
- `gpi` (shifted-polar GPI) exists but converges slowly (degrading tail); **prefer `ncg`**.
- Compile: `msign` NS loop is now `lax.fori_loop` (numerically identical, ~8× smaller graph → faster compile). Keep `BACKEND_STEPS=8` (polar_express is a tuned 8-coeff schedule; truncating breaks the polar).

## Planned launches

### P1 — `sqrt_ncg` α-sweep (PRIMARY, current focus)
Scale-invariant √-penalty, raw `d̂`, half-EKFAC, scheduled, K=12. α dimensionless ~O(1); probe showed
penalty/reward ratio 0.2→1.6 and direction "bend" 11%→65% across the range:
```
for a in 0.25 0.5 1.0 2.0; do
  launch "sqrtNCG-k12-a${a}-sched" -e CURV_SOLVER sqrt_ncg -e CURV_K 12 -e CURV_EKFAC_POWER half -e CURV_LAMBDA "$a"
done
```
Goal: find α; α=1.0 is the unit-matched center (penalty ≈ reward).

### P2 — `band_ncg` (Frobenius + soft σ-band)
```
launch "bandNCG-k20-lam1.0-sched" -e CURV_SOLVER band_ncg -e CURV_K 20 -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0
```

### P3 — relaxation comparison at matched λ=1.0, half (one each)
```
launch "secular-lam1.0-sched-half"        -e CURV_SOLVER secular            -e CURV_K 1 -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0
launch "dnSec-lam1.0-sched-half"          -e CURV_SOLVER doublenorm_secular -e CURV_K 1 -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0
launch "orignormNCG-r6-lam1.0-sched-half" -e CURV_SOLVER orignorm_ncg       -e CURV_K 6 -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0
launch "doubleNCG-r6-lam1.0-sched-half"   -e CURV_SOLVER doublenorm_ncg     -e CURV_K 6 -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0
```

### P4 — `ncg` (Stiefel quadratic) 2×3 reference sweep
```
for p in half quarter_trace; do for l in 0.3 1.0 3.0; do
  launch "ncg-r4-lam${l}-sched-ekfacQ-${p}" -e CURV_SOLVER ncg -e CURV_K 4 -e CURV_EKFAC_POWER "$p" -e CURV_LAMBDA "$l"
done; done
```

## Operational rules (learned the hard way)
- **Don't over-launch.** A ~26-job launch+mass-stop churn **wedged the shared marin controller** (RPCs hang, OOM/bloated-DB). Launch a small set, babysit, don't mass-stop.
- **The iris controller is shared** — do NOT restart it or roll back its DB to fix your own backlog; that's the owner/admin's call.
- **Babysit to wandb**, not just "submitted": confirm `train/loss` is logging in `curv-muon` before walking away. Monitor reads `iris job logs` tqdm postfix; that's training, but the user sees wandb.
- Each iris CLI call re-tunnels (~84 s) — use timeouts ≥180 s, or a persistent `--controller-url`.

## Status snapshot (as of this writing)
- Code: all variants + compile fix committed on branch `ekfac-curvature-variants` (this push).
- The marin iris controller was **wedged**; the P1 sqrt α-sweep + P2 band runs were submitted but orphaned/not logging. **Re-launch P1–P4 cleanly once the controller is recovered** (owner/admin VM-level fix), using the script above with tee'd output.
