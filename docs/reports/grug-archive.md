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
- Origin: base | moe | <source variant; omit for entries predating the base template>
- Introduced: <commit-sha>
- Last known-good: <commit-sha; superseded and deleted entries only>
- Status: active | superseded | deleted
- Purpose: <one line>
- Superseded by: <path or commit; optional>
- Diff: <CI-posted grug diff comment, or local report path; optional>
- Issue: <url/id; optional>
```

Active entries track `main`, so they omit `Last known-good` rather than carry a SHA that
goes stale on the next commit.

## Experiments

### grug-base-template
- Path: `experiments/grug/base/`
- Origin: base
- Introduced: 8d752a775
- Status: active
- Purpose: canonical grug template (model/train/launch).

### grug-moe
- Path: `experiments/grug/moe/`
- Origin: base
- Introduced: 9181fd753
- Status: active
- Purpose: canonical Mixture-of-Experts variant; carries its own model, optimizer, train loop, and launch wiring so it can iterate independently of the dense template.
- Issue: https://github.com/marin-community/marin/pull/3046

### grug-moe-pipeline
- Path: `experiments/grug/moe_pipeline/`
- Origin: `moe`
- Introduced: 734108b003
- Status: active
- Purpose: canonical automatic JaxPP ZeroBubble and DualPipeV implementation for Grug MoE.
- Diff: https://marin-community.github.io/marin/grug-diffs/pr-8739/moe_pipeline-vs-moe/index.html
- Issue: https://github.com/marin-community/marin/issues/7024

### grug-moe-hero-ep
- Path: `experiments/grug/moe_hero_ep/`
- Origin: `moe_hero_fsdp` at PR 7876
- Introduced: b0d20062a
- Status: active
- Purpose: one-rack GB200 EP64 throughput and MFU baseline.
- Issue: https://github.com/marin-community/marin/issues/7279

### grug-fast-track
- Path: `experiments/grug/fast_track/`
- Origin: `moe_hero_ep`
- Introduced: a021438bc7
- Status: active
- Purpose: self-contained 16k-vocab dense-vs-MoE H100 scaling ladder + MFU study; trimmed from moe_hero_ep (HF/ragged/host-offload removed, H100 fixed-pooled-wave EP only).
- Diff: https://marin-community.github.io/marin/grug-diffs/pr-9287/fast_track/index.html
- Issue: https://github.com/marin-community/marin/pull/9287

### grugformer-vs-hackable-125m
- Path: `experiments/speedrun/grugformer_vs_hackable_125m/grugformer_vs_hackable_125m.py`
- Introduced: 5efe76834
- Last known-good: 5efe76834
- Status: deleted
- Purpose: historical head-to-head comparison.
- Superseded by: 8d752a775 — template-first workflow centered on `experiments/grug/base/`.

## Belay+CPLM TPU baseline

- Path: `experiments/grug/moe_cplm`. Origin: Fast Track Belay `358283f7`, TPU execution `aef63f29`, portable CPLM loss PR #10230.
- Purpose: fixed B16×8192 d512 Belay+CPLM on TPU v4-32, EP1 dropless. Status: active; native port completed 400 updates at 417400 sustained tokens/s with zero drops and unchanged losses.
- Diff: [CI variant report](https://marin-community.github.io/marin/grug-diffs/pr-10260/moe_cplm/index.html), [PR #10260](https://github.com/marin-community/marin/pull/10260/files).

## Belay CPLM Plus head stack

- Path: `experiments/grug/moe_cplm/head_stack_kda_model.py`. Origin: native Belay CPLM Plus KDA projections at `0e5182f48`.
- Purpose: treat four existing KDA heads as four stack slots while retaining 256 KiB recurrent state per sequence/layer. Status: active kernel benchmark; no LM training.
- Source: `4c4d0121c`. [Protocol and implementation](../../experiments/grug/moe_cplm/head_stack_kda_design.md), [theory issue](https://github.com/WhenWen/astra-d8-research/issues/59).

## Belay CPLM Plus boundary head mixing

- Path: `experiments/grug/moe_cplm/boundary_stack_kda_model.py`. Origin: native Belay CPLM Plus KDA projections.
- Purpose: preserve all four native head updates and mix their states with push/pop/stay only at C16 boundaries, retaining 256 KiB recurrent state. Status: TPU kernel qualification; native LM launch pending qualification.
- Source: `4437dfe31`. [Protocol and implementation](../../experiments/grug/moe_cplm/boundary_stack_design.md), [theory issue](https://github.com/WhenWen/astra-d8-research/issues/59).
