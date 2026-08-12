---
topic: moe-yoco-kv-reuse
issue: https://github.com/marin-community/marin/issues/8196
baseline_issue: https://github.com/marin-community/marin/issues/6882
description: Evaluate parameter-preserving midpoint activation reuse for second-half K/V projections.
author: kaiyuew
---

# MoE Midpoint K/V Reuse: Research Logbook

## Current TL;DR

Implementation and validation are in progress on the exact July baseline commit
`52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`. The design derives its reuse
boundary from model depth: d512 caches layer 2 for layers 3--5, and d768 caches
layer 3 for layers 4--7. Every target layer retains its own Q/K/V/O and norm
parameters. The d512 cell is the first gate; d768 will only launch if d512 has
effective speedup greater than one.

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
- Interpretation: the variant changes computation while retaining the July
  parameter count, shapes, values, optimizer scalars, and compute-optimal cell
  definitions.
- Next action: complete repository contract checks, lint, dry-run resolution,
  snapshot the branch, then submit only d512.
