# Grug Archive: Experiments and Snapshots

This file is the paper trail for grug experiments.

## Principles

- `experiments/grug/base/` is the canonical template.
- Speedrun files are exploratory and may be deleted after upstreaming.
- Prefer deletion over long-term maintenance of stale experiment code.

## Entry Template

```text
### <experiment-id>
- Path: <repo-relative-path>
- Introduced: <commit-sha>
- Last known-good: <commit-sha>
- Status: active | superseded | deleted
- Purpose: <one line>
- Superseded by: <path or commit; optional>
- Issue: <url/id; optional>
```

## Experiments

### grug-base-template
- Path: `experiments/grug/base/`
- Introduced: TBD
- Last known-good: TBD
- Status: active
- Purpose: canonical grug template (model/train/launch).

### grugformer-vs-hackable-125m
- Path: `experiments/speedrun/grugformer_vs_hackable_125m/grugformer_vs_hackable_125m.py`
- Introduced: TBD
- Last known-good: TBD
- Status: deleted
- Purpose: historical head-to-head comparison.
- Superseded by: template-first workflow centered on `experiments/grug/base/`.

### moe-july-over-encoding-lr-sweep-7368
- Path: `experiments/grug/moe_july_over_encoding/`
- Introduced: `codex/july-baseline-oe-lr-sweep-7368`
- Last known-good: d512 LR sweep at `f4e545d852400c8265823850d37b240e53e533e5`; d768 Gate 1 pending
- Status: active
- Purpose: exact July Baseline fixed-C hierarchical Over-Encoding; tune the table LR at d512, then scale the selected 0.5x multiplier to d768.
- Diff: [against `july_baseline`](https://github.com/marin-community/marin/compare/july_baseline...WhenWen:codex/july-baseline-oe-lr-sweep-7368)
- Issue: [#7368](https://github.com/marin-community/marin/issues/7368)
