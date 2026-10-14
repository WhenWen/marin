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
intermediates and uses XLA chunked KDA and TPU grouped matmuls. The base16
configuration selects `kda_xla_scan_impl="parallel_stacked_split"`, which uses
stacked affine-scan reconstruction and independent stride-slice transposes.
Other model configurations retain the reference scan by default. TPU routing
acceptance uses inverse permutation positions, sharing that permutation with combine.
The selected KDA implementation passed two seven-update native runs at 1.2095
and 1.2112 seconds/update. Its 35-update check with active EMA passed at 1.2074
seconds/update. All recorded losses matched the prior stacked-scan references
exactly, and complete state was finite on every host. A seven-update run through
the launcher defaults also passed at 1.2050 seconds/update.
Inkling attention runs in rematerialized query blocks of 128 tokens, preserving
the bias and packed-document masks. Structured masks are built for each query
tile in global sequence coordinates; dense masks retain their slice path.
CUDA-only launch flags are not applied.
QB routing estimates quantile thresholds for expert bias updates.
On TPU v4 at the measured local routing shape (32,768 tokens, 512 experts, 10,000 bins),
QB histogram counts use binary matrix products with int32 accumulation,
64 experts per group and radix 64. Other shapes and devices retain the flat
histogram. Quantile interpolation and global count reduction are unchanged.

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

Prepare a JSON cache map with exactly the `data.components` names in
`experiments/grug/fast_track/belay_base15_d512.json`. Each entry maps a
component name to its cache URL. Every value must point to the corresponding cache in
`gs://marin-us-central2/`. Reproducing the original data requires preserving
tokenizer, token order, cache metadata and validation coverage; a similarly
named cache is not proof of equivalence. Fresh retokenization must record its
tokenizer files, source objects, document sampling/capping and shuffle settings.
Label any changed tokenizer or corpus as a separate data baseline.
No cache is copied or retokenized by this launcher.
The execution commands require an accessible, verified cache map; data
preparation is still pending as described below.

Resolve the configuration without starting training:

```bash
uv run --package marin-levanter --extra tpu python -m experiments.grug.fast_track.belay_tpu \
  --run-id belay-base16-v4-s0 --cache-map cache-map.json \
  --checkpoint-path gs://marin-us-central2/USER/belay-base16-v4-s0/checkpoints \
  --output outputs/belay-base16-v4-s0
```

Submit from the repository root on a remote execution host. Place
`cache-map.json` in that root and ensure it is not gitignored; Iris bundles
tracked and unignored untracked files. Use a unique run ID and output prefix
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
Find each attempt archive with `iris --cluster marin attempt describe /USER/JOB/0:0`
(replace the task index and attempt number for each worker), and export logs with
`iris --cluster marin job logs /USER/JOB --max-lines 1000000 > job-logs.txt`.
Archive capture is not guaranteed on every worker or cancellation path. Plot the raw
train/validation losses before reporting an experiment. Preserve partial logs
from failures and interruptions.

## Accelerator validation

`python -m experiments.grug.fast_track.validate_belay_tpu` checks the full
128-by-4096 d512 forward/backward with packed random tokens. It requires TPU v4,
writes per-rank numerical diagnostics, and performs no optimizer updates.
A single pass checks numerical execution; repeated timing requires the options
below. Neither establishes complete train-step throughput or LM quality.
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
batch size. Complete native train-step measurements are reported below;
real-data loss curves and equivalence to the GPU run remain unmeasured.

The entry point and numerical validation are implemented. Fresh-cache
retokenization is authorized, but the original tokenizer and training inputs
remain inaccessible. The [original data build](https://github.com/marin-community/marin/issues/9188#issuecomment-5707888411)
published `hero_bpe_tokenizers/v16384` and the document-capped model-training
corpus `hero_text_capped` under `s3://marin-us-east-02a/marin/datakit/`.
Neither path was found in the seven configured GCS regions. All 23 evaluation
source globs resolve in us-central2: 578 objects with sizes, generations and
MD5 hashes recorded in the [retokenization inventory](https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6158156421).
Raw-source availability does not establish tokenized-cache equivalence.

The remote execution host lacks credentials for the source
CoreWeave S3-compatible store. A metadata-only CPU job submitted to
`cw-us-east-02a` was rejected because this account is not admitted to that
cluster; its source listing never ran. The pending choice is accessible original
inputs or a separately labeled baseline with a new tokenizer and corpus.
No replacement data or tokenizer has been selected, and no retokenization job
has started.

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

The native benchmark uses model seed 0 and data seed 1, with a fixed random
batch of 128 sequences. Each 4096-token sequence contains four 1024-token
documents, with segment boundaries at positions 1024, 2048 and 3072.
The timed region includes the native optimizer, router bias updates and inline
parameter/gradient statistics. All hosts synchronize before the timer, and all
local outputs complete before it stops. Artifact uploads and plotting occur outside
the timer. The default `--ema-phase early` allocates EMA state but precedes its
activation at step 1817. `--ema-phase active` overrides the diagnostic activation
threshold to zero to measure the blend operation; model and optimizer state
still start at step zero. It does not simulate a trained checkpoint.
Every completed update saves its random-token loss and partial curves. These
losses do not measure LM quality. Data loading, checkpoints and held-out
evaluation are excluded.
The final parameters, optimizer buffers and EMA state are checked for
finiteness outside the timing region.

On 2026-10-13, changing aligned Inkling bias expansion from gather to pad/slice
reduced each host's numerical forward/backward median from 92.56 to 1.40 seconds
(about 66 times faster). The reported loss stayed 9.825540543 with finite
gradients. This comparison uses the numerical objective, eight timed calls
after two warmups, and no optimizer. Ten attention value/gradient cases passed
their unchanged tolerances. See the [measurement and parity record](https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6153215667).

The native benchmark at `1be1f0994`, `/kaiyue/belay-base16-native-profile-v3`,
completed seven updates on all four hosts: two warmups, one profiled update,
then four timed updates. The timed medians were 2.4858–2.4878 seconds. Dividing
the global 524,288 tokens per update by the slowest host's median gives about
211,000 tokens/second. All ranks reported
the same finite loss sequence, from 9.835014 to 5.998792, on the fixed random
batch. This includes EMA storage before activation. The compiler reported
9.57 GB temporary storage and 2.21 GB arguments per device; those figures do
not include all runtime allocations. Reports, raw losses, HLO, profiles and
rank-zero curves are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-profile-v3/`.

The first native attempt failed during lowering on a CUDA-only QuACK import;
the JAX fallback enabled the completed updates. The second attempt completed
but exposed host timing skew from reporting; the third added synchronization.
`/kaiyue/belay-base16-native-active-ema-v1` also completed seven updates at
`1be1f0994`, with the diagnostic EMA threshold set to zero. Its four host
medians were 2.4804–2.4813 seconds, and every reported loss matched the
early-EMA run exactly. This checks execution of the active blend at initial
optimizer state; it does not validate late-training behavior or EMA evaluation.
The [research record](https://github.com/WhenWen/astra-d8-research/issues/96)
preserves these attempts and their distinct measurement protocols.

The QB matrix-count implementation at `0fdda285a`,
`/kaiyue/belay-base16-native-qb-matmul-profile-v1`, reduced native host medians
to 1.4726–1.4739 seconds under the same seven-update protocol. Dividing
524,288 global tokens by the slowest host's median gives about 356,000
tokens/second, with 41% less time than the synchronized 2.4878-second
baseline above. Every loss matched that baseline exactly, and all four hosts passed
the final complete-state finiteness check. Compilation took 332.7–347.4
seconds; compiler cache conditions were not controlled. The compiler reported
9.57 GB temporary storage per device. Reports, losses, HLO, profiles and curves
are at `gs://marin-us-central2/kaiyue/belay-base16-tpu/native-qb-matmul-profile-v1/`.

The isolated histogram improved from 174.6 to 6.32 milliseconds. All 24
microbenchmark rows had exact per-bin count parity; the six cases in
`experiments/grug/fast_track/test_qb_balancing.py` and the tightened all-equal
quantile case passed. A grouped-scatter intermediate reached 1.862 seconds
for the slowest host's median under the same protocol before the matrix-count
implementation. These measurements use fixed random tokens and early EMA;
real-data training and evaluation remain unverified.

The matrix-count active-EMA run at `e8d111782`,
`/kaiyue/belay-base16-native-qb-matmul-active-ema-v1`, also passed exact parity
for all seven losses and complete-state finiteness on all four hosts. Host
medians were 1.4724–1.4748 seconds. As with the earlier active-EMA diagnostic,
this measures the blend operation from initial state, not late-training quality.

Building structured masks per query tile removes repeated full-sequence mask
construction inside the attention checkpoint. After 22 TPU mask/attention
cases passed, `/kaiyue/belay-base16-native-mask-tile-v1` at `e321355de` and
the independent `native-mask-tile-v2` repeat at `2ce7dba10` passed all seven
exact losses and complete-state finiteness on every host. Their slowest-host
medians were 1.4317 and 1.4321 seconds, respectively: 2.83–2.86% less time
than the 1.4739-second matrix-QB reference, or about 366,000 tokens/second.
Both runs used early EMA. The repeat's numerical source was unchanged;
intervening commits only added a separate benchmark and documentation.
Compiler temporary storage was 9.55 GB/device in the first run. Its compilation
took 337.0–345.6 seconds; the repeat took 40.1–55.8 seconds, with uncontrolled
cache conditions. Reports, raw losses, curves and profiles are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-mask-tile-v1/` and
`native-mask-tile-v2/`.
The first run's profile removed all full `8 × 4096 × 4096` mask operations
inside CausalSelfAttention. Aggregate attention self time fell from 1.403 to
1.049 million core-microseconds across eight cores, consistent with the measured
update reduction. Aggregate activity is not a critical-path decomposition.

The same mask path passed active EMA at `b97ec6384`,
`/kaiyue/belay-base16-native-mask-active-ema-v1`: host medians were
1.4256–1.4297 seconds, all seven losses matched exactly and every host's
complete state was finite. Compiler temporary storage remained 9.55 GB/device;
compilation took 346.7–358.2 seconds. Reports and the loss curve are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-mask-active-ema-v1/`.
This checks active EMA from initial optimizer state; real-data Belay acceptance
and late-training EMA quality remain unverified.

## Execution experiment history

Removing the inner attention checkpoint at `af0314a3f`,
`/kaiyue/belay-base16-native-no-inner-remat-v1`, reduced host medians to
1.3252–1.3344 seconds, with finite complete state. However, the seven losses
differed from the matrix-count baseline by up to 2.1935e-5, failing the frozen
exact-loss gate. Temporary compiler storage increased to 9.79 GB/device.
The production configuration retains its inner checkpoint. The benchmark's
`--attention-remat off` and `--attention-remat save-all` options are diagnostics;
the latter retains the checkpoint wrapper while saving its intermediates.
Neither option is a validated replacement for the default `on` setting.
Raw losses, curves and profiles are at
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-no-inner-remat-v1/`.

The `save-all` variant at `52a93099d`,
`/kaiyue/belay-base16-native-save-all-remat-v1`, also failed exact-loss parity:
the largest loss difference was 5.00679e-5. All hosts completed with finite
state and medians of 1.3826–1.3889 seconds, but compiler temporary storage
rose to 11.77 GB/device. Keeping the checkpoint wrapper did not recover the
baseline loss sequence. Reports, curves and profiles are at
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-save-all-remat-v1/`.

The row-gather microbenchmark at `b18b7fc91`, `/kaiyue/belay-row-gather-v1`,
passed bit-exact parity in all 40 cases. Ordinary bf16 indexing was fastest
at width 512: 0.838 ms for dispatch and 3.433 ms for permutation. Packing
pairs into uint32 or gathering output-row chunks slowed both operations;
neither was integrated. These isolated timings do not explain all gather
costs inside the native graph. Reports and HLO are at
`gs://marin-us-central2/kaiyue/belay-base16-tpu/row-gather-v1/`.

The expanded grid at `8ee76334e`, `/kaiyue/belay-row-gather-width256-v1`,
passed all 60 bit-exact cases. Ordinary indexing also won at width 256:
0.833 ms for dispatch and 0.834 ms for permutation. Packed pairs took
4.013 and 18.309 ms respectively; every chunked alternative was slower.
This covers the latent MoE dispatch width observed in the native profile.
Reports and width-qualified HLO files are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/row-gather-width256-v1/`.

The state-only KDA affine-scan diagnostic at `b97ec6384` retained exact
forward states but failed the unchanged 1e-5 absolute/relative gradient gate:
431 entries of the transition-matrix gradient differed, with maximum absolute
error 0.474609375. The state-increment gradient was exact. A localization run
at `63ac6ef25` found all failing entries in forced-reset chunks. These inputs
treat transition matrices as free parameters; parity through KDA's actual
document-masked parameterization requires the separate checks below. This
unconstrained-matrix failure remains part of the record. Reports are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/kda-scan-state-only-v1/` and
`kda-scan-error-locations-v1/`.

That separate KDA parameter check at `8ee76334e`,
`/kaiyue/belay-kda-parameter-scan-v1`, passed all 36 rows with zero observed
differences in outputs, final states, objectives and q/k/v/g/beta gradients.
It covers three shapes, seed 0, and unpacked, 1024-token and 83-token documents,
including resets inside chunks. At full shape `(8, 4, 4096, 128)`, the packed
value-and-gradient time fell from 83.4 to 68.8 ms; compiler temporary storage
fell from 6.85 to 6.18 GB. The native benchmark's `--kda-scan stacked` option
selects this reconstruction explicitly; `--kda-scan reference` restores the
reference scan for comparison, while the default follows the launcher config.
Reports and HLO are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/kda-parameter-scan-v1/`.

The first native stacked-scan run at `7cae3794f`,
`/kaiyue/belay-base16-native-kda-stacked-v1`, passed all seven exact reference
losses and complete-state finiteness on every host. Host medians were
1.2830–1.2933 seconds. The slowest median was 9.7% below the 1.4321-second
tiled-mask baseline, or about 405,000 tokens/second. Compiler temporary storage
fell from 9.55 to 8.85 GB/device; compilation took 335.0–346.7 seconds with
uncontrolled cache conditions. This fixed-random-token run uses seed 0 and
finishes before EMA updates begin. The independent repeat at `a3cc2a231`,
`/kaiyue/belay-base16-native-kda-stacked-v2`, also passed all seven exact losses
and finite state on every host. Its slowest median was 1.2915 seconds, a 9.82%
reduction; compiler temporary storage remained 8.85 GB/device. Compilation
took 32.23–32.32 seconds with uncontrolled cache conditions. The active-EMA
run at `3e3b33ed2`, `/kaiyue/belay-base16-native-kda-stacked-active-ema-v1`,
also passed all seven exact losses and finite state at 1.2904 seconds/update.
It enables EMA from initial state. The input-seed-1 and input-seed-2 grids at
`3e3b33ed2` passed all 72 rows with zero observed differences and unchanged
tolerances, bringing parameterization coverage to 108 rows across three seeds.
Reports, raw losses, curves and profiles are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-kda-stacked-v1/` and
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-kda-stacked-v2/`.
The corresponding rank-0 profile reduced aggregate KimiDeltaAttention self
time from 4.233 to 3.309 million core-microseconds across eight cores, relative
to `native-mask-tile-v1`. This measures aggregate activity, not the critical
path. The derived comparison is retained at
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-kda-stacked-v1/analysis/kda-profile-comparison.json`;
the additional seed grids are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/kda-parameter-scan-seed1-v1/` and
`gs://marin-us-central2/kaiyue/belay-base16-tpu/kda-parameter-scan-seed2-v1/`.

The isolated acceptance-mask grid at `a02ac1142`, `/kaiyue/belay-accepted-mask-v1`,
passed all 64 cases with exact masks and inverse positions. It covers two
shapes, partial/full/zero acceptance and concentrated assignments with invalid
expert IDs. At 262,144 assignments and 512 experts, computing acceptance from
inverse positions took 4.040–4.046 ms, compared with 7.351–7.375 ms for the
reference boolean scatter, when both construct and return the inverse positions
used by combine. Temporary compiler storage fell from 2.65 to 1.69 MB. The native
benchmark's `--accepted-mask reference` option restores the original scatter
for comparison; the default follows TPU production dispatch. Reports and HLO are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/accepted-mask-v1/`.

The first native inverse-mask run at `a3cc2a231`,
`/kaiyue/belay-base16-native-accepted-inverse-v1`, passed all seven exact
reference losses and complete-state finiteness on every host. Median step
times were 1.3839–1.3858 seconds across the four hosts. The slowest median was
3.23% below the 1.4321-second tiled-mask baseline. Compiler temporary storage
was 9.546 GB/device; compilation took 350.3–358.4 seconds with uncontrolled
cache conditions. This run uses the reference KDA scan and finishes before
EMA updates begin. Its independent repeat at `3e3b33ed2`,
`/kaiyue/belay-base16-native-accepted-inverse-v2`, passed the same exact-loss
and finiteness checks at 1.3977 seconds/update, a 2.40% reduction. Both exceed
the frozen 2% gate. The first run's profile removed 24 large boolean-scatter
HLO rows and reduced routing-sort rows from 36 to 24. Reports, raw losses,
curves and profiles are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-accepted-inverse-v1/` and
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-accepted-inverse-v2/`.

The first combined stacked-KDA and inverse-mask run at `80aff07e4`,
`/kaiyue/belay-base16-native-kda-accepted-v1`, matched all seven losses from
`native-mask-tile-v2` exactly. Parameters, optimizer buffers and EMA state
were finite on every host. The slowest host median was 1.2514 seconds,
3.10% below the faster KDA-only repeat. Compiler temporary storage was
8.854 GB/device. The candidate launcher path at `fcc2455cb` selects both
kernels directly; the native benchmark defaults exercise it without trace
patches. Its independent seven-update run, `native-production-v2`, passed
all exact reference losses and complete-state finiteness on every host.
The slowest median was 1.2460 seconds, 3.52% below the faster KDA-only repeat,
or about 421,000 tokens/second. Compiler temporary storage remained
8.854 GB/device. Compilation took 31.81–32.21 seconds with uncontrolled
cache conditions.

The same production path passed `native-production-active-35-v1`, with EMA
active from initial state: all 35 losses matched `native-reference-35-v1`
exactly, and complete state was finite on every host. The slowest median
was 1.2520 seconds over 32 timed updates after two warmups and one profiled
update. This is 13.41% below the reference's 1.4459 seconds, exceeding the
frozen 2% gate. Compiler temporary storage remained 8.854 GB/device;
compilation took 347.0–356.6 seconds with uncontrolled cache conditions.
The reference finishes before EMA starts; EMA does not feed the training
loss. This does not compare EMA buffer values or late-training quality.

Reports and curves use each `native-*` name above as a directory under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/`.

The integrated implementations at `0ed2832d8` passed all 10 selected TPU unit
cases in `/kaiyue/belay-integrated-kernels-v1`: eight comparisons of the two
KDA scan implementations against a sequential oracle and two head-pairing
checks. The same job passed all 36 KDA parameter-grid rows and 64 routing-mask
rows with zero observed differences. JUnit, the attempt archive and
microbenchmark reports are retained under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/integrated-kernels-v1/`.

Further KDA backward experiments targeted the interior padding used to transpose
strided slices. Two candidates at `b21db1bd3` and `783cfad37` combined the
paired-even and paired-odd cotangents; each failed three full-shape
parameter-gradient rows and was not advanced to native training. Their reports
remain under `kda-split-vjp-v1/` and `kda-split-vjp-v2/` in the GCS artifact
parent above.

The separate-slice candidate at `aa64d324a` preserves each slice's gradient
branch and inserts zeros through stack/reshape. It passed 54 parameter-grid
rows per input seed (0, 1 and 2), 162 rows total, with zero maximum absolute
output, final-state, objective or q/k/v/g/beta gradient differences against the
original JAX associative scan. Each seed job also passed 15 sequential-oracle
unit cases. At `(batch, heads, sequence, head_dim)=(8,4,4096,128)`, median
value-and-gradient time for 1024-token and 83-token documents was 19.1–19.3%
lower than the validated stacked scan in every seed. The seed-2 unpacked case,
with no document resets, showed no improvement: 116.9 ms for the candidate
versus 115.6 ms for the validated stacked scan. These are isolated-kernel results.

For the seed-0 1024-token case, compiler temporary storage fell from 6.176 to
6.046 GB; interior-pad HLO operations fell from 40 to zero, while total pad
operations rose from 98 to 141. Reports, HLOs and unit-test archives use prefixes
`kda-split-vjp-v3/`, `kda-stride-seed1-v1/` and `kda-stride-seed2-v1/`.
At source `60aec58c0`, `native-stride-v1` and `native-stride-v2` tested the
separate-slice implementation through normal model dispatch with
`--kda-scan stacked-split`. Each matched all seven losses from
`native-production-v2` exactly on all four hosts, with finite parameters,
optimizer buffers and EMA state. Their slowest host medians were 1.2095 and
1.2112 seconds/update, respectively 2.92% and 2.79% below the 1.2460-second
reference. Both passed the frozen 2% gate.

The same source passed `native-stride-active-35-v1`: all 35 losses matched
`native-production-active-35-v1` exactly, and complete state was finite on
every host. Both runs activate EMA blending from initial model/optimizer
state. The slowest median over 32 timed updates was 1.2074 seconds, 3.56%
below the 1.2520-second reference, or about 434,000 tokens/second. Final
random-token loss was 2.1129331589. EMA buffer equality and trained-checkpoint
quality were not tested.

All three native candidate runs reported 9.138 GB/device of compiler temporary
storage, 3.21% above the 8.854 GB reference. Their compilation times ranged
from 41.8 to 347.7 seconds with uncontrolled cache conditions. The first run's
profile attributed 339,253 core-microseconds to KDA padding, compared with
783,569 for `native-production-v2`: 3.60% versus 8.03% of aggregate operation
self-time across eight TPU cores. These shares do not measure wall-time savings.
Exported profile tables are under `native-stride-v1/analysis/`.

Commit `08fcbc4e4` selects this implementation in the base16 launcher.
`native-stride-default-v1` passed the
[frozen default-path check](https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6156914446)
without kernel-selection overrides: all seven losses matched
`native-production-v2` exactly and complete state was finite on all four hosts.
The slowest host median was 1.204977358 seconds/update, 3.29% below the
1.245962153-second reference. Compiler temporary storage remained 9.138 GB/device. See the
[three-seed replication result](https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6156530045)
and [native validation plan](https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6156421544).
Reports and loss curves retain each native run name under the GCS artifact
parent above.

The broadcast-select transpose candidate at `4a7741354` failed three of 72
parameter-grid rows in `kda-stride-select-v1`, all full-shape backward cases.
Maximum gradient differences were 0.00048828125 without resets and with
1024-token documents, and 0.000244140625 with 83-token documents. The k/g/beta
gradients failed atol=rtol=1e-5 against the original JAX associative scan;
objectives and q/v gradients matched exactly. All 20 sequential-oracle unit cases passed. The unpacked backward HLO
had 71 pad operations versus 140 for the validated separate-slice transpose,
but numerical failure prevented backward timing or native promotion. The candidate was
removed; its source commit, reports, HLOs and JUnit remain recorded under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/kda-stride-select-v1/`.

The outer-layer SAVE diagnostic at `c3cf5ccab`, `native-layer-save-v1`, completed
seven updates with finite parameter, optimizer and EMA arrays on all four hosts.
SAVE retains layer residuals for ordinary autodiff; RECOMPUTE reruns the layer
forward in its custom backward. Inner attention
checkpointing remained enabled. Its slowest host median was 0.96475 seconds,
19.94% below `native-stride-default-v1`, but updates 2–7 failed exact scalar-loss
parity; the maximum absolute difference was 3.719329833984375e-5. Compiler temporary
storage rose from 9.138 to 12.915 GB/device (+41.33%). This single trial failed
the frozen acceptance gate and was not advanced to repeat or active-EMA checks.
The launcher retains RECOMPUTE. Raw losses, curves, reports, HLO and profiles
remain under `gs://marin-us-central2/kaiyue/belay-base16-tpu/native-layer-save-v1/`.

The saved-pullback candidate at `d1a10c8d4`, `native-saved-vjp-v1`, retained the
RECOMPUTE path's explicit history-gradient additions and backward barriers while
saving forward residuals. All seven updates completed with finite parameter,
optimizer and EMA arrays on every host. Updates 1–2 matched the default reference
exactly, but updates 3–7 differed; maximum absolute scalar-loss difference was
1.430511474609375e-5. Its slowest host median was 0.97193 seconds, 19.34% below
`native-stride-default-v1`. Compiler temporary storage rose to 14.059 GB/device
(+53.85%). The candidate failed the frozen exact-loss gate and was removed.
Its source commit and artifacts remain available; reports, HLO, profiles and
loss curves use the literal object prefix
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-saved-vjp-v1//rank-{0,1,2,3}/`.

Adding an input optimization barrier before forming the saved pullback at
`888315a78` also failed exact-loss parity in `native-saved-barrier-v1`.
Scalar training losses at updates 1–2 matched `native-stride-default-v1`;
updates 3–7 differed, with maximum absolute difference 8.106231689453125e-6.
All seven updates completed, and parameter, optimizer and EMA arrays were finite
after the final update on every host. The maximum of
four host medians over four timed updates was 1.03653 seconds, 13.98% below the
reference. Host medians ranged from 0.96772 to 1.03653 seconds. Compiler temporary
storage was 14.991 GB/device (+64.06%); runtime peak memory was not measured.
These are single-trial timings. The failed candidate was removed. Its reports, HLO,
profiles and loss curves remain under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-saved-barrier-v1/`.

The QB histogram reduce-scatter diagnostic at `5bf6750d9`,
`native-qb-scatter-active35-v1`, matched all 35 scalar losses of
`native-stride-active-35-v1` exactly, with finite final parameters, optimizer
and EMA buffers on all four hosts. It partitions histogram counts by expert,
computes thresholds on each partition and gathers those thresholds.
The maximum host median over 32 timed updates was 1.20181 seconds versus
1.20744 seconds for the reference, a 0.47% reduction in this single trial.
Compiler temporary storage rose from 9.138 to 9.817 GB/device.
The candidate missed the [frozen 1% performance gate](https://github.com/WhenWen/astra-d8-research/issues/96#issuecomment-6158204146).
The benchmark-only override and helper extraction were removed, restoring the
original production all-reduce implementation. The conditional control required
both numerical and performance gates to pass, so it was not launched.
Reports, raw losses, curves, HLO and profiles remain under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/native-qb-scatter-active35-v1/`.

## Fixed-state gradient localization

Run `experiments.grug.fast_track.diagnose_belay_gradients_tpu` on v4-32 with
`--revision COMMIT --artifact-prefix gs://marin-us-central2/USER/RUN` and
`XLA_FLAGS=--xla_allow_excess_precision=false`. It uses the native loss and
quantile-based router-bias helpers at identical initial weights and packed
inputs, without optimizer updates. An exact, finite
reference repeat must pass before the saved-pullback candidate is evaluated.
The reference recomputes each layer during backward; the candidate saves its
pullback during forward. Both use the frozen base16 recipe defined above.
The worker records per-parameter and per-layer gradient comparisons and both
optimized HLOs. Materializing gradients changes the compilation boundary, so
this diagnostic cannot certify fused training-step parity.

At `ebc10497e`, `gradient-localization-v1` passed the reference control on all
four hosts. Reference and candidate scalar objectives both equaled
9.835014343261719, but 72 of 137 gradient tensors differed. All gradients were
finite and within `atol=rtol=1e-5`; the maximum absolute difference was
5.7220458984375e-6. Layers 4 and 5 (zero-based) had differences in MLP gated
normalization and short-convolution gradients; layer 5 also differed in its
attention gated-normalization up-projection. The output projection and final
normalization gradients matched exactly. These findings do not identify the
first divergent operation or resolve the native failure to match all seven
`native-stride-default-v1` update losses exactly. The launcher remains on
RECOMPUTE. Reports, objective evaluations and optimized HLOs are under
`gs://marin-us-central2/kaiyue/belay-base16-tpu/gradient-localization-v1/`.
