---
topic: moe-yoco-kv-reuse
issue: https://github.com/marin-community/marin/issues/8196
baseline_issue: https://github.com/marin-community/marin/issues/6882
description: Evaluate parameter-preserving midpoint activation reuse for second-half K/V projections.
author: kaiyuew
---

# MoE Midpoint K/V Reuse: Research Logbook

## Current TL;DR

The d512 gate completed successfully on the exact July baseline commit
`52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`: terminal Paloma macro loss improved
from 3.57940 to 3.55697 while final-100-step throughput decreased 0.66%, for an
estimated 1.121x effective speedup. The generalized d768 cell is now running.
The design derives its reuse boundary from model depth: d512 caches layer 2 for
layers 3--5, and d768 caches layer 3 for layers 4--7. Every target layer retains
its own Q/K/V/O and norm parameters.

## Scope

- Goal: test a parameter-preserving YOCO-like variant where second-half query
  projections use the current activation while key and value projections use
  the last first-half layer output.
- Generality: derive the first reuse layer as `num_layers // 2` for the even
  depth d512 and d768 July recipes rather than hard-coding layer indices.
- Primary metrics: final `eval/paloma/macro_loss`, final-100-step mean
  `throughput/tokens_per_second`, and effective speedup.
- Constraints: exact issue #6882 July ancestry and recipe; no parameter sharing
  or removal; d512 before d768; v5p-8.

## Baseline

- Code ref: `july_baseline` commit
  `52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`.
- Historical d512: Paloma macro loss 3.5667, 352,609 tokens/s, batch 16,
  10,980 steps, 8192-token context, v5p-8.
- Exact-code d512 control from issue #8131: Paloma macro loss 3.5793972 and
  355,680 final-100-step tokens/s. This is the primary comparator because it
  exposed measurable run-to-run drift from the historical July result.
- W&B: https://wandb.ai/marin-community/dial_moe/runs/MOE-ROW-NORM-JULY-CTRL-001-d512

## Experiment Log

### 2026-08-12 - Design and exact-July implementation

- Hypothesis: preserving each attention layer's parameters while giving the
  second half a stable midpoint K/V representation can improve optimization or
  throughput enough to yield effective speedup.
- Design: `kv_reuse_start_layer` identifies the first target layer. The model
  caches the output of the preceding layer. Target-layer Q consumes its current
  attention input; K/V consume the cached source after that target layer's
  attention RMSNorm and GatedNorm.
- Config: d512 maps source layer 2 to layers 3--5; d768 maps source layer 3 to
  layers 4--7. Both mappings are derived from the heuristic's layer count.
- Validation: eight focused tests pass, including exact equality of every
  initialized array against the untouched July model at six and eight layers,
  exact recipe/optimizer equality apart from the new routing field, and an
  observable split between query and K/V inputs. A full six-layer backward
  pass is finite and reaches both the cached source layer and a reuse layer.
  The focused, optimizer, and repository Grug contract suites pass together
  (25 tests), and the required lint/format checks pass.
- Dry run: the d512 selector resolves only the d512 training step and its 36
  data dependencies; it does not select d768.
- Interpretation: the variant changes computation while retaining the July
  parameter count, shapes, values, optimizer scalars, and compute-optimal cell
  definitions.
- Snapshot: `acfb060db` on `codex/moe-yoco-kv-reuse-8196`.
- Next action: push the snapshot and submit only d512.

### 2026-08-12 - d512 gate result and d768 launch

- Iris: `/kaiyuew/moe-yoco-kv-july-d512-8196` and its single training child
  succeeded with zero failures after 10,980 steps. The run took 1:31:44,
  including evaluation and checkpointing.
- W&B: https://wandb.ai/marin-community/dial_moe/runs/MOE-YOCO-KV-JULY-001-d512
- Quality: terminal `eval/paloma/macro_loss` was 3.5569701 versus 3.5793972 for
  the exact-code July control, a reduction of 0.0224271.
- Performance: final-100-step throughput was 353,336.6 tokens/s versus
  355,680.4 tokens/s for the control, a 0.659% reduction. Using the experiment
  scaling-law gate (`alpha=0.0941`, `L_inf=1.6`) gives 1.1287x compute reduction
  at matched loss and 1.1213x effective speedup after throughput.
- Artifact: the durable final checkpoint exists at
  `gs://marin-us-central1/grug/moe_yoco_kv_reuse_july_d512-362057/checkpoints/step-10980`.
- Decision: the d512 result clears the greater-than-one gate, so d768 was
  submitted as `/kaiyuew/moe-yoco-kv-july-d768-8196` from the same snapshot.
  Exactly one child was created. Its live W&B configuration confirms hidden
  size 768, eight layers, `kv_reuse_start_layer=4`, 256 experts with top-4
  routing, batch 32, and 16,875 steps.
- W&B: https://wandb.ai/marin-community/dial_moe/runs/MOE-YOCO-KV-JULY-001-d768
- Next action: monitor d768 through finite training metrics and terminal state,
  then compare it with the exact July d768 control.

### 2026-08-12 - Generalize to the odd-depth d1024 cell

- User decision: run d1024 alongside the active d768 experiment.
- Design: use `ceil(num_layers / 2)` as the first reuse layer. This is unchanged
  for the even-depth cells. At d1024's 11-layer exact July configuration, layers
  0--5 remain standard, the output of layer 5 is cached, and layers 6--10 use it
  for their K/V projections.
- Exact cell: hidden size 1024, 11 layers, batch 64, 16,080 steps, 8,192-token
  context, 256 experts with top-4 routing, and v5p-8.
- Validation: all 12 focused recipe, initialization, forward, backward, and
  optimizer tests pass. The 11-layer model retains exact initialized parameter
  equality with the July baseline. The required full lint/type/format pass also
  succeeds.
- Dry run: the d1024 selector resolves only the d1024 training step and its 36
  data dependencies; it does not select d512 or d768.
- Snapshot: `c7a57ba7a` on `codex/moe-yoco-kv-reuse-8196`.
- Launch: submitted `/kaiyuew/moe-yoco-kv-july-d1024-8196`; exactly one training
  child was created. The live W&B config matches 1024 hidden size, 11 layers,
  `kv_reuse_start_layer=6`, 256 experts with top-4 routing, batch 64, and 16,080
  steps. After a 3:20 XLA compile, the run produced finite loss and throughput
  and saved its initial recovery checkpoint with zero failures.
- W&B: https://wandb.ai/marin-community/dial_moe/runs/MOE-YOCO-KV-JULY-001-d1024
- Next action: monitor d768 and d1024 to terminal state, verify their final
  checkpoints, and compare terminal Paloma and final-100-step throughput with
  their exact July controls.

### 2026-08-13 - d1280 scale cell and overtrained benchmark design

- User decision: extend fixed YOCO to d1280 on v5p-16 and benchmark it in an
  overtrained regime.
- d1280 config: exact July hidden size 1280, 13 layers, batch 128, 14,325 steps,
  8,192-token context, 256 experts with top-4 routing, and about 8.64B total
  parameters. Layers 0--6 remain standard; layers 7--12 take K/V input from the
  output of layer 6. Every layer retains its own parameters.
- Compute search: Iris supports v5p-16 in both configured v5p regions. The
  us-east5 pool is in provisioning backoff, while us-central1 has no active
  backoff signal. The job requests the topology generically so Iris can route
  it to available capacity.
- Overtraining source: issue #8062 defines the current MoE comparison regime as
  750 tokens per active parameter. Exact-July d512 has 20,730,368 active
  parameters excluding embedding and LM head, yielding a 15,547,776,000-token
  target. Rounding to complete batches gives 118,620 steps and 15,547,760,640
  realized tokens, about 10.8x the compute-optimal d512 token schedule.
- Comparison: launch a fresh unchanged d512 control and fixed-YOCO arm with the
  same batch, steps, data, seed, optimizer law, eval cadence, and v5p-8
  resources. This avoids comparing against a control with a different training
  horizon or code snapshot.
- Validation: all 16 focused recipe, initialization, forward, backward, and
  optimizer tests pass, including the 13-layer d1280 boundary and the 750-TPP
  rounding contract. Required lint/format checks pass. A broader untouched base
  CPU integration test still fails on this branch from an automatic-vs-explicit
  mesh mismatch; it is unrelated to the experiment files.
- Next action: dry-run all three new selectors, perform Iris and W&B duplicate
  checks, snapshot the branch, and submit the d1280 cell plus the matched
  overtraining pair.

### 2026-08-13 - Correct new-run placement

- Initial submissions inherited the CPU parent's `europe-west4` placement.
  Their Fray children therefore requested v5p-16 or v5p-8 in a region with no
  matching Iris group and failed before any TPU allocation, checkpoint, or W&B
  run was created.
- Resolution: explicitly pin all three new TPU children to `us-central1`, where
  both topology groups are configured, and place their lightweight parents in
  the same region. This also keeps checkpoints and input data in the existing
  us-central1 experiment bucket.
