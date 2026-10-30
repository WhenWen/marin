# Original-scaling Muon versus MuonH on KMA d512

This experiment compares MuonH (norm-scaled updates projected to the parameter's
Frobenius sphere) with the supplied Markdown's original Muon:
`sqrt(max(1, fan_out/fan_in))`. The user confirmed this multiplier on October 28.
The original Muon arm preserves the reference Adam and AdamH groups, including
the AdamH output head. It removes MuonH norm scaling and sphere projection from
matrix updates and applies fixed decoupled weight decay 0.1.
The previously proposed unclamped spectral arm was a misinterpretation; it was
never launched and has been removed from the runner.

The [model source](https://github.com/marin-community/marin/tree/4eba59cafa27a1ea174d1b5d6a28d9d1660eba10/experiments/grug/fast_track)
is KMA d512 (Kimi Delta Attention, Multi-head Latent Attention, and Attention Residuals):
6 layers, 384 experts/top-8, 256 latent width, KDA/MLA/Inkling and AttnRes.
The data and tokenizer identities are extracted from [PR 10678](https://github.com/marin-community/marin/pull/10678) at
`caff3778edcb23d41a154ba962b67daf3d669fab`. The original 16K tokenizer is
replaced with the cached Llama-3.1 tokenizer (128256 vocabulary). Context
remains 4096. No CPLM, Plus head, Belay architecture, or bigram embeddings are added.

All arms use the same data order, initialization seed, TPU backend, context,
evaluation examples and token budget. TPU adaptation selects dropless scatter
experts, portable Newton–Schulz, and fused TPU vocabulary loss. AttnRes
backward recomputation changes memory use, without changing the model formula.

The first comparison uses B16 (global batch of 16 sequences) for 19,800 updates, the same 1,297,612,800 tokens
as B128 for 2,475 updates in the attachment. Learning rates are scaled by
`sqrt(16/128)`; beta2 is 0.999499874937461 and epsilon is 1.361533886761545e-15,
from the source revision's `MoeHeuristic` at this batch.
The first original-Muon comparison uses the batch-scaled reference Muon LR
without an additional LR multiplier. MuonH is rerun on these inputs, rather
than compared to the attachment's old loss.
All Muon matrices have fixed decoupled weight decay 0.1; the shape multiplier
applies only to their update direction. Router/gate decay is
`0.02 * max(1 - step/19800, 0)`. KDA `w_beta` uses twice its arm's matrix LR.

Primary metrics are final Paloma and Uncheatable macro BPB (equal-weight mean
over domains, evaluated on the parameters after update 19,800), with validation
loss and per-domain results retained. Validation uses the first 2048 sequences
per domain. The tokenizer change prevents comparing the old NLL directly.
A single-seed LR winner is exploratory; repeat the selected arm and MuonH
before declaring a stable improvement.

Every execution initializes online W&B before updates and stores its resolved
configuration, first-batch finite-gradient check and incremental JSON loss log.
Each process-group restart gets fresh W&B and checkpoint paths; it starts from
initialization and is not described as checkpoint resume. Iris preemption retry
defaults remain enabled. No queue, task, or training wall-clock limit is set.

The initial online logging check passed with zero training updates:
https://wandb.ai/marin-community/marin/runs/logcheck-4f8e6938601f4ec8847d388ae2b53cfc

Optimizer tests:

```bash
uv run --no-sync --with pytest --with pytest-timeout python -m pytest experiments/grug/muon_mup_tpu/test_optimizer.py -q
```

The optimizer checks include numerical direction/decay checks, Adam-group parity,
and a four-device expert-sharding regression. MuonH B16 passed TPU preflight
and was training with `reference` execution as of October 28, 23:15 PDT. B128 exceeded HBM before updates; B64 runs but has worse token
throughput than B16. The formal paired comparison uses B16 and 19,800 steps.
The completed comparison is reported below. Local submission state is under
`scratch/20261028-1105_monitoring_state.json`.

## TPU execution qualification

For d768, the execution adapter passes CE block sizes `(B=1024, H=128, V=256)`.
Despite the API's H setting, the TPU forward kernel loads all 768 hidden
dimensions and four vocabulary tiles;
the previous V=512 setting required 16.94 MiB VMEM on v4, above its 16 MiB limit.
This changes kernel tiling only. `--qualify-ce` checks loss and gradients of
input hidden states and output-head weights against XLA at the failing local
shape `(tokens=4096, hidden=768, vocabulary=128256)`, including masked tokens and
the final partially populated group of four vocabulary tiles. It writes
`ce-qualification.json` before the full-model finite-gradient preflight and
continues the requested training schedule only if both checks pass.

`--execution reference` retains the first TPU adaptation. `histogram` replaces
only the flat QB histogram with exact integer two-stage histograms; `tpu` also
ports the ordered FP32 KDA state pass and its reverse pass from the Belay
implementation at `fcad3cbe6`. The adapter uses KMA's existing chunk preparation,
16-token chunks, document boundaries, 384 experts and routing interpolation.
It does not import the Belay model, CPLM head or its optimizer.

`--qualify` uses the real first B16 batch and native online W&B initialization.
It advances one reference step, then supplies that resulting state to each requested
implementation. The requested modes share weights, pending QB betas and optimizer state, with
nonzero warmup LR.
After the initial TPU measurement, the user explicitly accepted the histogram's
rounding differences (gradient relative L2 7.7e-7, original Muon update relative
L2 8.1e-5). Its revised gate keeps exact loss and auxiliary metrics, finite values,
gradient relative L2 at most 1e-5 and update relative L2 at most 1e-3. The KDA and Inkling gate was fixed before their TPU measurements: loss difference at
most 2e-4, global gradient/update relative L2 at most 0.005, each gradient leaf
relative L2 at most 0.02. Relative L2 divides the difference norm by the reference
norm; the per-leaf denominator has a floor of `1e-8 * sqrt(leaf.size)`. Auxiliary metrics and per-leaf update differences are retained but do not gate
KDA or Inkling selection. Tolerances must not be weakened without user approval; the original diagnostic
reports retain their initial verdicts, with acceptance reviews stored separately.

Each mode then runs two warmups and eight synchronized complete optimizer steps
as independent evaluations from the same fixed state and batch; this measures
computation, without data loading or logging overhead. The diagnostic uses non-donating buffers to retain the common
input state. Production throughput must also be checked with the native donated
state. Only a numerically qualified mode with at least a 10% median step-time
improvement over `reference` in this diagnostic can be selected. `qualification.json` is persisted incrementally;
`qualification-complete.json` is distinct from a completed training run.
The result collector separates execution modes when matching comparisons.

Run `python -m experiments.grug.muon_mup_tpu.train` with the usual training
arguments and `--qualify`. The reports are written below the attempt UUID under
`--artifact-root` and `--output`; `qualification.json` contains `selected` and
per-mode errors, pass flags and timings. Pass its accepted mode as `--execution`
for training, then check production throughput on the running job. The ported source files are
`experiments/grug/moe/kda_state_tpu.py` and
`experiments/grug/fast_track/qb_balancing.py` in the Marin Belay checkout at
`fcad3cbe6`; the premeasurement limits are recorded in `qualify.py` and the report.

### Measured ports and remaining bottleneck

On October 28, the first full-model qualification measured these synchronized
median step times on global batch 16 × sequence length 4096 / TPU v4-32:

| Execution | Original Muon | MuonH |
| --- | ---: | ---: |
| Reference | 4.7944 s | 4.7945 s |
| Two-stage histogram | 4.7518 s | 4.7654 s |
| Histogram + KDA state kernel | 4.7234 s | 4.7261 s |

The user accepted the histogram rounding differences. Both arms retain identical
loss and routing statistics; gradient relative L2 is below 8e-7 and update relative
L2 below 9e-5. Acceptance reviews are stored beside the original reports, without
rewriting their initial bitwise-gate verdicts. The KDA port changed updates by
13.5% (Muon) and 10.8% (MuonH) in global relative L2 against reference, and was not selected. Neither port gave a useful
speed improvement at the original KMA geometry.

An eight-step original-Muon reference device profile located the bottleneck in dense Inkling bias
expansion: four elementwise gathers, two scatter transposes and two sorts consume
about 4.50 seconds of each 4.80-second device step (summed XLA operations on one TPU core). The profile and a compact
trace summary are under
`gs://marin-us-central2/kaiyuew/muon-mup-20261028/tpu-port-profile/1b360cfec4f1460a81bd692a11f2a80f/`.

`--execution inkling` ports aligned slice indexing from the existing TPU
`blocked_inkling_attention` implementation at Marin `1c64d46c541fe5c61492f580ac5a080915771f49`.
It replaces only dense bias expansion, preserving the original full QK contraction,
softmax, masks, KDA and optimizer. Six FP32/BF16 forward-and-transpose cases pass
bitwise CPU comparison. Full-model TPU qualification passed for both arms.
Loss and auxiliary metrics are identical; gradient relative L2 is 1.01e-6
for Muon and 8.34e-7 for MuonH, with update relative L2 1.91e-4 and 1.66e-4.
The synchronized complete-step medians are 0.3699 s for Muon and 0.3954 s
for MuonH, versus 4.7952 s and 4.7973 s for their respective references:
12.97× and 12.13× speedups. Both reports select `inkling`. These timings exclude
data loading and logging. Native production checks over logged steps 50–100
(51 steps, excluding compilation) measured median step times of 0.2815 s for
Muon and 0.2823 s for MuonH: 232,848 and 232,155 tokens/s. The native donated
state is faster than the diagnostic that retains its input state. Native step
timings also exclude data loading and callbacks. Elapsed W&B timestamps over
the same interval give 0.3647 s and 0.3777 s per step, including those costs:
179,684 and 173,529 tokens/s end to end. Reported MFU
is 3.283% and 3.273%, versus about 0.193% for the reference. Both arms passed
the finite-gradient preflight with initial loss 11.8044195 and reached over
150 updates by October 28, 23:23 PDT. Final validation was pending at that observation.

The qualification reports are under `inkling-qualification-muon/bdb740c5831b4535aeacee114813c61c`
and `inkling-qualification-muonh/13f4b979842f487f82fd15b7f1421285`, relative to
`gs://marin-us-central2/kaiyuew/muon-mup-20261028/`.
The full 19,800-update B16 comparison was submitted with `--execution inkling`
as Iris jobs `kma-muon-original-inkling-20261028-b16-s0-v1` and
`kma-muonh-inkling-20261028-b16-s0-v1`. Each starts from initialization with a
fresh online W&B run: [original Muon](https://wandb.ai/marin-community/marin/runs/kma-muon-original-inkling-d512-b16-s0-7545c3670287)
and [MuonH](https://wandb.ai/marin-community/marin/runs/kma-muonh-inkling-d512-b16-s0-b8c1d55ff590).
The older reference jobs remain running as of October 28, 23:15 PDT:
MuonH B16 has completed 8,392 updates and original Muon B16 has completed 1,142.

Use `--qualify --qualification-modes reference inkling` to reproduce it.
`--profile-steps 8 --execution reference`, in a separate invocation without
`--qualify`, captures eight warmed native optimizer steps with online W&B.
`experiments/grug/muon_mup_tpu/trace_summary.py` accepts the emitted
`.trace.json.gz` via `--source` and a GCS JSON destination via `--output`; it
streams the trace on a CPU worker in the storage region. The acceptance reviews
are `acceptance-review.json` under the qualification attempt directories
`tpu-port-qualification/130b5c270ceb41f49f91cf17fe5d3ad9` and
`tpu-port-qualification-muonh/7c100e26e3294216b3a5e27a9785a6e6`, relative to
`gs://marin-us-central2/kaiyuew/muon-mup-20261028/`.

## Completed baseline and matched matrix LR sweeps

Both Inkling runs completed 19,800 updates and final evaluation. Their comparison
keys match: the model, data, seed, precision, execution and evaluation protocol
are identical. Final macro BPB (lower is better):

| Optimizer | Paloma | Uncheatable |
| --- | ---: | ---: |
| Original Muon | 1.27324748 | 1.05293880 |
| MuonH | 1.26944342 | 1.04656432 |

MuonH has lower BPB in 22 of 23 domains. This is one seed at the reference learning
rates; it does not establish the ranking after tuning. The final per-domain
metrics and comparison keys are saved in
`gs://marin-us-central2/kaiyuew/muon-mup-20261029/lr-sweep/baseline-comparison.json`.

The October 29 sweeps use matrix LR multipliers 0.25, 0.5, 1, 2 and 4 for each
optimizer, reusing the completed 1x runs. All LR values below are peak schedule
values; warmup and linear decay remain unchanged. The original Muon base LR is
0.005606496627706948; the MuonH base LR is 0.007928783568302187. Every point keeps
head LR 0.007928783568302187, Adam LR 0.0018297192849928126, B16, seed 0 and
19,800 updates. KDA write-strength matrices retain twice the matrix LR.

`--lr-mult` now applies to either optimizer. The experiment-local
`MuonHSweepConfig` separates MuonH's head schedule from its matrix schedule.
Multi-step tests compare its updates to the original optimizer at 0.25x, 1x
and 4x: matrix groups match the corresponding native MuonH LR, while head and
Adam groups match the unchanged reference schedules, bitwise. The original Muon resolved
1x configuration is unchanged by this implementation change.

Use the existing TPU launch with `--optimizer muonh --lr-mult <multiplier>
--execution inkling --batch-size 16 --steps 19800 --seed 0`; selecting `muonh`
automatically uses `MuonHSweepConfig`. Selecting `muon` uses original-scaling
Muon. Submission identities and artifact roots are recorded in `manifest.json`
(Muon) and `muonh-manifest.json` under the same GCS sweep directory above.

## KMA d768 validation

`--size d768` selects the fast-track KMA rung: width 768, eight layers,
384 experts, top-8 routing, and latent/expert widths 384. The Llama 3.1 vocab
remains 128,256, giving 1,604,663,844 parameters. The data-matched ladder budget
is 3,633,315,840 tokens: 55,440 updates at batch 16 and sequence length 4,096.
The TPU Inkling execution mode preserves the qualified model math.

`--optimizer muonw` uses original-scaling Muon with matrix WD 0.1 and AdamW
for every fallback group, including the head. AdamW decays logical matrices;
layer-stacked norm/bias vectors and AttnRes queries remain exempt. Existing
`muon` retains the attachment's AdamH head and original fallback rules.

At d768's budget and batch, the heuristic sets MuonH/head LR to
0.0048275735852692635 and Adam LR to 0.0011140554427544456. MuonW's matrix
LR is 0.0034136100188209497, preserving the attachment's 1/sqrt(2) ratio to
the MuonH base. KDA beta retains its 2x matrix multiplier and AttnRes queries
retain their 0.1x fallback LR. These are transferred base settings, not
measured d768 optima.

```bash
uv run python -m experiments.grug.muon_mup_tpu.train \
  --size d768 --optimizer muonw --steps 55440 --batch-size 16 \
  --execution inkling --run-name kma-muonw-d768-b16-s0 \
  --artifact-root gs://marin-us-central2/kaiyuew/kma-d768-20261029/muonw
```
