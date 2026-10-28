# Muon spectral scaling on the original KMA d512

This experiment compares MuonH (norm-scaled updates projected to the parameter's
Frobenius sphere), original Muon (`sqrt(max(1, fan_out/fan_in))`), and spectral
Muon (`sqrt(fan_out/fan_in)`). It preserves [the supplied reference's](reference_configs.md) Adam
and AdamH groups, including the AdamH output head. It tests spectral Muon at
fixed width; it does not establish full-network muP or hyperparameter transfer.

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
Muon candidates scan 0.5, 1 and 2 times the batch-scaled reference Muon LR. MuonH
is rerun on these inputs, rather than compared to the attachment's old loss.
All Muon matrices have fixed decoupled weight decay 0.1; the shape multiplier
applies only to their update direction. Router/gate decay is
`0.02 * max(1 - step/19800, 0)`. KDA `w_beta` uses twice its arm's matrix LR.

Primary metrics are final Paloma and Uncheatable macro BPB (equal-weight mean
over domains, evaluated on the parameters after update 19,800), with validation
loss and per-domain results retained. Validation uses the first 2048 sequences
per domain. The tokenizer change prevents comparing the old NLL directly.
A single-seed LR winner is exploratory; repeat the selected arm and MuonH
before declaring a stable improvement. Width transfer requires a separate test.

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

The 12 numerical checks passed on CPU, including a four-device expert-sharding regression. TPU execution and final metrics are
pending. Local submission state is under `scratch/20261028-1105_monitoring_state.json`.
