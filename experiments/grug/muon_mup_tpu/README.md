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
and is training. B128 exceeded HBM before updates; B64 runs but has worse token
throughput than B16. The formal paired comparison uses B16 and 19,800 steps.
Final validation results are pending. Local submission state is under
`scratch/20261028-1105_monitoring_state.json`.

## TPU execution qualification

`--execution reference` retains the first TPU adaptation. `histogram` replaces
only the flat QB histogram with exact integer two-stage histograms; `tpu` also
ports the ordered FP32 KDA state pass and its reverse pass from the Belay
implementation at `fcad3cbe6`. The adapter uses KMA's existing chunk preparation,
16-token chunks, document boundaries, 384 experts and routing interpolation.
It does not import the Belay model, CPLM head or its optimizer.

`--qualify` uses the real first B16 batch and native online W&B initialization.
It advances one reference step, then supplies that resulting state to all three
implementations. They share weights, pending QB betas and optimizer state, with
nonzero warmup LR.
After the initial TPU measurement, the user explicitly accepted the histogram's
rounding differences (gradient relative L2 7.7e-7, original Muon update relative
L2 8.1e-5). Its revised gate keeps exact loss and auxiliary metrics, finite values,
gradient relative L2 at most 1e-5 and update relative L2 at most 1e-3. The KDA gate was fixed before TPU measurements: loss difference at
most 2e-4, global gradient/update relative L2 at most 0.005, each gradient leaf
relative L2 at most 0.02. Relative L2 divides the difference norm by the reference
norm; the per-leaf denominator has a floor of `1e-8 * sqrt(leaf.size)`. Auxiliary metrics and per-leaf update differences are retained but do not gate
KDA selection. Tolerances must not be weakened without user approval; the original diagnostic
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
for training after checking production throughput. The ported source files are
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
bitwise CPU comparison. Full-model TPU qualification is pending for both arms.
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
