---
topic: inkling-rope-relative-attention-gate1
issue: https://github.com/marin-community/marin/issues/7208
description: Real July baseline half-RoPE plus Inkling-parameterized learned relative attention.
author: kaiyuew
---

# Inkling RoPE + Relative Attention Gate 1: Task Logbook

## Scope
- Goal: preserve the real July baseline RoPE/GQA architecture, add the validated Inkling relative-attention parameterization, and run matched d512/d768 Gate 1 cells.
- Primary metrics: finite training loss, macro validation loss, matched Paloma losses, throughput, and exact final checkpoints.
- Constraints: branch from `marin/july_baseline`; keep half-RoPE and long-layer RoPE disablement; no duplicate run identities; do not restart Iris.
- Coordinating issue/PR: https://github.com/marin-community/marin/issues/7208

## Current TL;DR
- The architecture is based directly on upstream `marin/july_baseline` commit `52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`.
- Local value/gradient, half-RoPE ordering, optimizer-routing, Gate 1 recipe, generic Grug contract, type, and lint checks pass at commit `8581074ae`.
- Fresh d512 and d768 children exist under the two-cell parent and are pending normal v5p capacity with zero failures.

## Baseline
- Date: 2026-07-18
- Code refs: `marin/july_baseline@52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`; variant snapshot `8581074ae`.
- Baseline numbers: d512 macro loss 3.5667 at 352,609 tokens/s; d768 macro loss 3.2272 at 249,954 tokens/s.

## Entry Log
### 2026-07-18 08:00 - Rebased architecture and validated local contracts
- Hypothesis: adding the learned relative term after the July baseline's Q/K normalization and half-RoPE path preserves the baseline positional signal while exposing Inkling's direct causal relative-distance bias.
- Commit Hash: `8581074ae`
- Command:
  - `uv run --with pytest --with pytest-timeout pytest -q tests/test_grug_rope_relative_position.py experiments/grug/moe_rope_relative_position/test_optimizer.py tests/test_grug_variant_contracts.py`
  - `uv run --project . --group test pytest -q tests/kernels/test_relative_position_attention.py` from `lib/levanter`
  - `./infra/pre-commit.py --all-files --fix`
- Config: half-RoPE retained; designated long layers skip RoPE; GQA retained; head_dim=128; R=16; E=1024; learned Q/K RMSNorm gains initialized to one; content qk scale=1/128; independent untruncated N(0,0.02) relative table and W_r; all new learned parameters Adam-routed.
- Result: 23 experiment/contract tests passed, 4 Pallas kernel tests passed, and all pre-commit/type checks passed.
- Interpretation: the combined parameterization is locally consistent with both the July baseline and the previously validated relative-attention implementation.
- Next action: perform fresh-identity Iris duplicate checks, submit the two-cell Gate 1 parent, and verify actual child allocation plus W&B config.


### 2026-07-18 22:55 - Submitted fresh two-cell Gate 1 parent
- Hypothesis: the exact July-baseline rebase can launch both widths without reusing any prior relative-attention artifact or W&B identity.
- Commit Hash: `9a7a2327b08f54024347cd678f2247c405fc249f`
- Command: `WANDB_API_KEY=<set> /Users/kaiyuew/Downloads/Project/marin/.venv/bin/python scratch/20260718-2249_july_rope_relative_gate1_resubmit.py`
- Config: CPU parent `/kaiyuew/july-baseline-rope-relative-attention-gate1-7208`; two v5p-8 children; fresh W&B group `MOE-JULY-ROPE-RPE-INKP-gate1-issue-7208`.
- Result: parent is running; exactly two children were created. d512 and d768 are pending with zero failures/preemptions while the autoscaler brings up demand-routed `tpu_v5p-preemptible_8-us-east5-a` workers. Preflight Iris and W&B duplicate checks returned zero matches.
- Interpretation: the launch graph and width isolation are correct; current pending state is capacity, not failure.
- Next action: wait for allocation, verify first finite loss and W&B config for both children, then continue monitoring to terminal checkpoints and matched Paloma results.
