# Belay base16 on TPU v4

The TPU entry point freezes Larry's d512 base15 control and applies the three
base16 gate removals. It retains 512 routed experts, top-8 routing, 768-wide
shared experts, the 524,288-row bigram table, KDA/MLA/Inkling, MuonH/AdEMAMix,
the 16k tokenizer, global batch 128, sequence length 4096, and 2817 steps.
Final evaluation uses the original pure EMA settings (0.995 over the last
1000 steps), Paloma and uncheatable datasets.

Sources:

- [Recorded base15 control](https://wandb.ai/marin-community/marin_moe/runs/lc1-b205-ctrl),
  saved without host metadata in `experiments/grug/fast_track/belay_base15_d512.json`.
- [Base16 change](https://github.com/marin-community/marin/issues/9451#issuecomment-6105926801):
  disable `kda_write_gate`, `kda_erase_gate`, and `kda_onchip_erase_gate`.
- Source branch `ladderclimb_v1_batch14`, commit
  `358283f7da684d156d8bca095669cf8c023ef2f3`.
- [Port and validation record](https://github.com/WhenWen/astra-d8-research/issues/96).

## Execution differences

Use v4-32 (16 chips across four hosts) for the full model. The launcher uses
one expert-parallel group spanning the chips. It recomputes attention
intermediates and uses the existing XLA chunked KDA and TPU grouped matmuls.
Inkling attention runs in rematerialized query blocks of 128 tokens, preserving
the bias and packed-document masks. CUDA-only launch flags are not applied.

The default is one complete routing batch. `--microbatches` can lower memory
use, but changes the capacity/drop boundary and must be reported as an
experiment change. The TPU expert topology differs from the eight-GPU source;
cross-backend numerical and learning-curve equivalence remain separate checks.
This port does not include later experimental KDA state reshaping or OKLS.

## Data and launch

Prepare a JSON cache map with exactly the component names in the frozen
configuration. Every value must point to the corresponding cache in
`gs://marin-us-central2/`. Preserve tokenizer, token order, cache metadata and
validation coverage; a similarly named cache is not proof of equivalence.
No cache is copied or retokenized by this launcher.

Resolve the configuration without starting training:

```bash
uv run --package marin-levanter --extra tpu python -m experiments.grug.fast_track.belay_tpu \
  --run-id belay-base16-v4-s0 --cache-map cache-map.json \
  --checkpoint-path gs://marin-us-central2/USER/belay-base16-v4-s0/checkpoints \
  --output outputs/belay-base16-v4-s0
```

Submit from the repository root on a remote execution host. The cache map must
be included in the Iris workspace bundle. Use a unique run ID and output prefix
for each new run; automatic restart/resume is disabled.

```bash
uv run iris --cluster marin job run --enable-extra-resources \
  --tpu v4-32 --region us-central2 --no-preemptible --priority batch \
  --cpu 8 --memory 128GB --disk 16GB --timeout 14400 \
  --sync-package marin-levanter --extra tpu --job-name belay-base16-v4-s0 \
  -e PYTHONPATH .:lib/levanter/src:lib/haliax/src:lib/fray/src:lib/iris/src:lib/rigging/src:lib/finelog/src:lib/marin/src:lib/zephyr/src \
  -e JAX_PLATFORMS tpu,cpu -- \
  uv run --package marin-levanter --extra tpu python -m experiments.grug.fast_track.belay_tpu \
    --run-id belay-base16-v4-s0 --cache-map cache-map.json \
    --checkpoint-path gs://marin-us-central2/USER/belay-base16-v4-s0/checkpoints --run
```

The command has a four-hour operational ceiling, not a measured duration.
Reaching that ceiling is an interrupted result, not a completed comparison.
Each attempt preserves `resolved-config.yaml`, `provenance.json`, and per-rank
`metrics.jsonl` under `IRIS_OUTPUT_DIR`; checkpoints use the supplied GCS path.
Collect the attempt archive and plot the raw train/validation losses before
reporting an experiment. Preserve partial logs from failures and interruptions.

## Accelerator validation

`python -m experiments.grug.fast_track.validate_belay_tpu` checks the full
128-by-4096 d512 forward/backward with packed random tokens. It requires TPU v4,
writes per-rank numerical diagnostics, and performs no optimizer updates.
It does not establish optimizer-state memory use, throughput, or LM quality.

Run kernel tests on TPU with one pytest process (`-n 0`); pytest's repository
default starts multiple workers that contend for the TPU runtime:

```bash
uv run --package marin-levanter --extra tpu --group test --with pytest-timeout pytest \
  experiments/grug/moe/test_kda.py \
  experiments/grug/fast_track/test_kda_local.py \
  experiments/grug/fast_track/test_mla_num_heads.py \
  lib/levanter/tests/grug/test_attention.py -n 0 -q
```
