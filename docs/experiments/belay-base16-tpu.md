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

Muon uses batched JAX Newton–Schulz on TPU. Training requires
`XLA_FLAGS=--xla_allow_excess_precision=false`. All four tested sharded/unsharded
Muon cases passed with this setting; the two tall-matrix cases failed under the
default. This supports compiler rounding as the cause without establishing
parity for every shape. The worker checks the flag before training and records
it in provenance. This is independent of TPU VMEM settings.

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
  -e JAX_PLATFORMS tpu,cpu -e XLA_FLAGS --xla_allow_excess_precision=false -- \
  uv run --package marin-levanter --extra tpu python -m experiments.grug.fast_track.belay_tpu \
    --run-id belay-base16-v4-s0 --cache-map cache-map.json \
    --checkpoint-path gs://marin-us-central2/USER/belay-base16-v4-s0/checkpoints --run
```

The command has a four-hour operational ceiling, not a measured duration.
Reaching that ceiling is an interrupted result, not a completed comparison.
Each attempt writes `resolved-config.yaml`, `provenance.json`, and per-rank
`metrics.jsonl` under `IRIS_OUTPUT_DIR`; checkpoints use the supplied GCS path.
Collect the attempt archive when available and export the Iris job logs; archive
capture is not guaranteed on every worker or cancellation path. Plot the raw
train/validation losses before reporting an experiment. Preserve partial logs
from failures and interruptions.

## Accelerator validation

`python -m experiments.grug.fast_track.validate_belay_tpu` checks the full
128-by-4096 d512 forward/backward with packed random tokens. It requires TPU v4,
writes per-rank numerical diagnostics, and performs no optimizer updates.
It does not establish optimizer-state memory use, throughput, or LM quality.
`--batch-size 4` permits an architecture check on v4-8, with the same d512
geometry and 4096-token sequences. This does not validate the full batch's HBM use.

Run kernel tests on TPU with one pytest process (`-n 0`); pytest's repository
default starts multiple workers that contend for the TPU runtime:

```bash
uv run --package marin-levanter --extra tpu --group test --with pytest-timeout pytest \
  experiments/grug/moe/test_kda.py::test_chunk_kda_matches_recurrent \
  experiments/grug/moe/test_kda.py::test_chunk_kda_bf16_matmuls_match_recurrent \
  experiments/grug/moe/test_kda.py::test_chunk_kda_gradients_finite \
  experiments/grug/moe/test_kda.py::test_floor_decay_is_exact_with_16_token_chunks \
  experiments/grug/fast_track/test_kda_local.py \
  experiments/grug/fast_track/test_mla_num_heads.py \
  lib/levanter/tests/grug/test_attention.py -n 0 -q --timeout=600
```

The broader KDA file also exercises interpreted GPU Pallas kernels. On v4,
ten of those checks failed their original tolerances; the TPU model dispatches
to XLA `chunk_kda`, not those kernels. The original tolerances are unchanged.
See the linked validation record for the full test results and run identities.

On 2026-10-13, `/kaiyue/belay-base16-v4-8-full-model-v1` passed the full
d512 architecture forward/backward at diagnostic batch 4 and sequence 4096:
random-token loss 9.825521469, finite gradients, 227.44 seconds including
compilation. This is a numerical check, not a throughput or training-quality
measurement. The first multi-host validator attempt failed because its input
arrays were captured in a closure; the corrected validator passes them as
explicit JIT arguments.

`/kaiyue/belay-base16-tpu-model-tests-v2` passed all 18 selected model and
attention tests on TPU in 193.05 seconds. The 600-second per-test ceiling
allows compilation; numerical tolerances are unchanged. The earlier attempt
hit the default timeout while compiling a KDA gradient test.

`/kaiyue/belay-base16-v4-32-full-shape-v2` then passed on all four hosts
(16 TPU v4 chips) with the original batch 128 and sequence 4096. Every rank
reported loss 9.825540543 and finite gradients; compilation plus execution
took about 434.6 seconds. This validates forward/backward at the original
batch size. Optimizer-state memory, sustained throughput, real-data loss
curves, and equivalence to the GPU run remain unmeasured.

The entry point and numerical validation are implemented. Training is blocked
until verified us-central2 mirrors of the original 16k-token caches are
available. The remote execution host could not access the source S3 caches
without AWS credentials; no replacement data was used.

## Performance measurements

`python -m experiments.grug.fast_track.validate_belay_tpu` can time repeated
forward/backward calls and capture XPlane profiles with HLO metadata. Pass
`--training-objective --benchmark-steps 8 --profile-steps 2 --revision COMMIT`
and a unique `--artifact-prefix gs://marin-us-central2/USER/RUN`. It applies
the JAX settings and z-loss from `belay_base15_d512.json` and enables the model's
training loss terms. Reports record `jax_config` and `z_loss_weight`. It warms
up twice, then times each combined forward/backward call, waiting for all its
local outputs. Compare the slowest rank's timings separately from compilation. Each rank
uploads its report and profile directly to GCS. No optimizer updates run.

The first profiling attempt, `belay-base16-v4-32-profile-baseline-v1`, uses the
numerical objective without z-loss and `jax_softmax_custom_jvp=false`. Preserve it as a
diagnostic reference; it is not the training-objective performance baseline.
Full native train-step timings must additionally include optimizer work, router
bias updates, EMA and metrics before making training-throughput claims.

For that measurement, run this worker command on v4-32:

```bash
XLA_FLAGS=--xla_allow_excess_precision=false \
  uv run --package marin-levanter --extra tpu --with matplotlib \
  python -m experiments.grug.fast_track.benchmark_belay_train_tpu \
    --steps 4 --revision COMMIT \
    --artifact-prefix gs://marin-us-central2/USER/RUN
```

The timed region includes the native optimizer, router bias updates and inline
parameter/gradient statistics. It allocates EMA state, but these early steps
precede EMA activation at step 1817; active EMA timing remains a separate check.
Every completed update saves its random-token loss and partial curves. These
losses do not measure LM quality. Data loading, checkpoints and held-out
evaluation are excluded.

The first native benchmark failed during lowering on a CUDA-only QuACK import.
The JAX optimizer fallback passes the parity cases above, but successful native
step timings remain unverified.
