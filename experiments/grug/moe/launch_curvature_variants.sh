#!/bin/bash
# ---------------------------------------------------------------------------------------------------
# Launch script for the EK-FAC curvature-corrected Muon variants on MoE may-arch d512.
#
# Each variant is an INNER SOLVER for the curvature-corrected Muon update on the may-arch d512 MoE
# (PR #6153 baseline). They all share the EK-FAC eigenbasis / second-moment machinery and differ only
# in the constraint geometry + inner solver. Selected via CURV_SOLVER.
#
# PREREQS
#   - WANDB_API_KEY exported in your shell (NOT hardcoded here).
#   - iris controller reachable (`.venv/bin/iris --config lib/iris/config/marin.yaml cluster status`).
#   - Run from THIS worktree: `iris job run` bundles the working tree, so the bundle must contain this code.
#   - Data mirrored in us-east5 (MARIN_PREFIX=gs://marin-us-east5); v5p-8 preemptible there.
#
# HOW IT RUNS: `iris job run ... -- python -m experiments.grug.moe.launch` submits a small CPU *launcher*
# job; the launcher reads the GRUG_*/CURV_* env (below) to build GrugMoeMuonHConfig and submits the
# actual TPU *training* job via executor_main. wandb runs are created by the training job (project curv-muon).
#
# OUTPUT: every launch is tee'd to /tmp/launch_<tag>.log so the job id is never lost to shell buffering.
# ---------------------------------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/../../.."  # -> repo root (the marin worktree)
IRIS=".venv/bin/iris --config lib/iris/config/marin.yaml"

launch () {
  local tag="$1"; shift
  echo "=== launching $tag ==="
  $IRIS job run --no-wait --preemptible --region us-east5 --extra tpu \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT curv-muon -e MARIN_PREFIX gs://marin-us-east5 \
    -e GRUG_TPU v5p-8 -e GRUG_REGION us-east5 -e GRUG_ZONE us-east5-a -e GRUG_PREEMPTIBLE 1 \
    -e GRUG_EP 1 -e COEFF_TYPE polar_express -e BACKEND_STEPS 8 \
    -e CURV_TWO_SIDED 1 -e CURV_CONSTRAINT stiefel -e CURV_LAMBDA_TRACKS_LR 1 -e CURV_EKFAC 1 \
    -e GRUG_RUN_TAG "$tag" "$@" \
    -- python -m experiments.grug.moe.launch 2>&1 | tee "/tmp/launch_${tag}.log" \
    | grep -iE "submitted|/kaiyue/iris-run-job-2"
}

# ===================================================================================================
# ENV KNOBS (read by experiments/grug/moe/launch.py)
#   CURV_SOLVER        which inner solver (see table below)
#   CURV_K             inner iterations / rounds (per-solver "proper K" below)
#   CURV_LAMBDA        curvature strength λ  (for sqrt_ncg this is the dimensionless α)
#   CURV_LAMBDA_TRACKS_LR  1 → λ_t = CURV_LAMBDA · lr_t/peak  (schedule the curvature with the LR)
#   CURV_EKFAC         1 → EK-FAC augmented per-coordinate second moment D̂ (vs Kronecker Shampoo)
#   CURV_EKFAC_POWER   half = D̂^{1/2} (natural-gradient curvature) | quarter_trace = D̂^{1/4}·tr^{1/4}
#                      (IGNORED by sqrt_ncg — it uses the RAW second moment D̂ for unit consistency)
#   CURV_TWO_SIDED=1, CURV_CONSTRAINT=stiefel  : standard for these variants
#   COEFF_TYPE=polar_express, BACKEND_STEPS=8  : msign (polar) Newton-Schulz schedule (keep 8 — tuned)
#
# SOLVER MENU  (constraint geometry → solver → proper K from convergence analysis)
#   ncg                 Stiefel XᵀX=I, quadratic curvature, Newton-CG            K=4   power=half|quarter_trace
#   secular             EK-basis single row/col-norm (closed form, exact)        K=1   (any power)
#   doublenorm_secular  EK-basis double norm, additive-Sinkhorn dual (exact)     K=1
#   orignorm_ncg        original-coord single norm, product-sphere Newton-CG     K=6
#   doublenorm_ncg      original-coord double norm, Sinkhorn-retraction NCG      K=6
#   band_ncg            Frobenius sphere + soft σ-band [√2/2,√2], adaptive-τ CG  K=20
#   sqrt_ncg            scale-invariant √-penalty (raw D̂), Newton-CG             K=12  α=CURV_LAMBDA (~O(1))
#   (gpi: shifted-polar GPI — kept for reference but converges slowly; prefer ncg.)
# ===================================================================================================

# --------- pick what to run (uncomment) ----------

# (A) sqrt_ncg α-sweep — scale-invariant √-penalty, raw D̂, half-EKFAC, scheduled. α≈O(1).
# for a in 0.25 0.5 1.0 2.0; do
#   launch "sqrtNCG-k12-a${a}-sched" -e CURV_SOLVER sqrt_ncg -e CURV_K 12 -e CURV_EKFAC_POWER half -e CURV_LAMBDA "$a"
# done

# (B) band_ncg — Frobenius + soft singular-value band [√2/2, √2], half-EKFAC, λ=1.0, scheduled.
# launch "bandNCG-k20-lam1.0-sched" -e CURV_SOLVER band_ncg -e CURV_K 20 -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0

# (C) NCG (Stiefel) 2×3 sweep — quadratic curvature, {half,quarter_trace} × λ∈{0.3,1.0,3.0}.
# for p in half quarter_trace; do for l in 0.3 1.0 3.0; do
#   launch "ncg-r4-lam${l}-sched-ekfacQ-${p}" -e CURV_SOLVER ncg -e CURV_K 4 -e CURV_EKFAC_POWER "$p" -e CURV_LAMBDA "$l"
# done; done

# (D) secular / double-norm / orig-norm comparison (λ=1.0, half), one each:
# launch "secular-lam1.0-sched-half"          -e CURV_SOLVER secular            -e CURV_K 1  -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0
# launch "dnSec-lam1.0-sched-half"            -e CURV_SOLVER doublenorm_secular -e CURV_K 1  -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0
# launch "orignormNCG-r6-lam1.0-sched-half"   -e CURV_SOLVER orignorm_ncg       -e CURV_K 6  -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0
# launch "doubleNCG-r6-lam1.0-sched-half"     -e CURV_SOLVER doublenorm_ncg     -e CURV_K 6  -e CURV_EKFAC_POWER half -e CURV_LAMBDA 1.0

echo "Edit this script: uncomment the block(s) you want, then re-run. Job ids are tee'd to /tmp/launch_<tag>.log."
