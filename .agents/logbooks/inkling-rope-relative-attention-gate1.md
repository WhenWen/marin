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
- The first fresh d512/d768 launch reached TPU execution but failed before step 1 on a July-JAX compatibility error in the ported Pallas output-shape helper. The compatibility fix is locally validated and awaiting a corrected snapshot relaunch.

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

### 2026-07-18 23:05 - Diagnosed pre-step-1 kernel compatibility failure
- Hypothesis: the failure is an API-version mismatch in the ported kernel rather than an architecture, data, or numerical-stability issue.
- Commit Hash: `e1303c0fd`
- Evidence: both W&B `output.log` files end at the first compiled training step with `AttributeError: 'ShapedArray' object has no attribute 'manual_axis_type'` in `relative_position_attention/pallas_tpu.py::_output_shape`; neither run reported a global step or loss.
- Fix: construct plain `jax.ShapeDtypeStruct(shape, dtype)` outputs, matching the real July baseline's JAX 0.9.2 API, instead of reading the later-JAX `manual_axis_type` attribute.
- Validation: all four relative-attention kernel value/gradient tests pass; targeted Ruff, Black, Pyrefly, and `git diff --check` pass. The repository-wide pre-commit wrapper reached only network timeouts while resolving already-known tool packages, not code findings.
- Interpretation: this is a deterministic startup compatibility fault shared by both widths; no checkpoint or training metric was produced, so corrected runs must use fresh identities.
- Next action: commit and push the compatibility fix, launch a fresh corrected parent with new Iris/W&B identities, and verify first finite metrics before returning to long-running babysitting.

### 2026-07-18 23:23 - Submitted corrected fresh-identity Gate 1 parent
- Hypothesis: removing the later-JAX-only output metadata access lets the otherwise unchanged July-baseline RoPE plus relative-attention cells compile under the real July TPU environment.
- Commit Hash: `7473d0f07` (includes compatibility commit `e1303c0fd`)
- Command: `WANDB_API_KEY=<set> /Users/kaiyuew/Downloads/Project/marin/.venv/bin/python scratch/20260718-2249_july_rope_relative_gate1_resubmit.py`
- Config: parent `/kaiyuew/july-baseline-rope-relative-attention-v2-gate1-7208`; W&B group `MOE-JULY-ROPE-RPE-INKP2-gate1-issue-7208`; fresh d512/d768 run IDs with the `INKP2` prefix. Architecture, optimizer routing, budgets, and widths are unchanged.
- Result: preflight found no Iris or W&B duplicates. The corrected parent is running with exactly the intended d512 and d768 children and no siblings. Both children are pending normal demand-routed v5p capacity with zero failures/preemptions.
- Next action: wait through allocation and TPU compilation, then require fresh finite step/loss signals from both arms.

### 2026-07-18 23:35 - Reduced July-JAX backward tiles after scoped VMEM compile OOM
- Hypothesis: the second startup failure is caused by the TPU wrapper ignoring the configured 128-token attention block size and selecting 512-token default backward tiles under July's JAX 0.9.2 compiler.
- Evidence: corrected d512 passed the earlier `manual_axis_type` site, then failed during first-step compilation with `CompileTimeScopedVmemOom`; `flash_mha_bwd_dq_block_q_major_512_block_k_major_512_block_k_512` requested 16.12 MB against a 16.00 MB scoped VMEM limit. No step or loss was emitted. d768 was stopped before completing the same unusable compile to release its TPU.
- Fix: the TPU call now explicitly maps the existing `attention_block_size=128` config into all forward and backward Pallas block sizes. No model, data, batch, sequence-length, optimizer, initialization, or positional parameter changed.
- Validation: eight model/recipe tests and four kernel value/gradient tests pass; the TPU dispatch regression test verifies the configured tile reaches the kernel; `UV_OFFLINE=1 ./infra/pre-commit.py --all-files --fix` passes completely.
- Interpretation: this is a compile-resource correction, not an architecture change. The final allowed recovery must use fresh Iris/W&B identities because the `INKP2` runs now exist.
- Next action: commit and push the 128-token TPU tile fix, submit the `INKP3`/v3 fresh identities, and require finite metrics from both widths.

### 2026-07-18 23:40 - Submitted final fresh-identity recovery
- Hypothesis: explicit 128-token forward/backward tiles fit July JAX within v5p scoped VMEM while preserving the exact Gate 1 mathematics and parameterization.
- Commit Hash: `2e1f5f9c3`
- Command: `WANDB_API_KEY=<set> /Users/kaiyuew/Downloads/Project/marin/.venv/bin/python scratch/20260718-2249_july_rope_relative_gate1_resubmit.py`
- Config: parent `/kaiyuew/july-baseline-rope-relative-attention-v3-gate1-7208`; W&B group `MOE-JULY-ROPE-RPE-INKP3-gate1-issue-7208`; output hashes d512 `e1a437` and d768 `b72fec`.
- Result: preflight found no Iris/W&B duplicates and exactly the two intended children materialized. Both workers were then preempted once during startup; Iris retained both jobs as running/pending with zero failures while replacement demand-routed workers are acquired.
- Interpretation: the current state is infrastructure preemption, not a code or numerical failure. This consumed the final configured recovery, so the identities must now be monitored in place rather than relaunched again.
- Next action: wait for replacement allocation and require both arms to pass compilation and emit finite step/loss metrics.
