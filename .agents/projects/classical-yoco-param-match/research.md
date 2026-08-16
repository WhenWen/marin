# Classical YOCO parameter matching

## Background Research Brief

- Effort: medium
- Stop rule: stop once the primary paper and current Marin implementation determine the shared-K/V boundary, removed parameter tensors, and a falsifiable d512 launch matrix.
- Date: 2026-08-16

### Question

Does classical YOCO, which projects one global K/V pair from the midpoint representation and reuses those projected tensors in every cross-decoder layer, improve the d512 MoE overtraining result? If it reduces total parameters relative to the July baseline, does reinvesting the saved parameters in expert capacity or query-head capacity change the result?

### Current Marin Context

- The existing fixed-YOCO variant keeps every layer's Q/K/V/O weights. Its second half reuses the midpoint activation only as the input to each layer's own K/V projections.
- The exact-July d512 model has six layers, four query heads, one K/V head, head dimension 128, 256 routed experts, and a midpoint boundary at layer 3.
- The fixed-YOCO d512 750-TPP run uses 118,620 steps, batch size 16, sequence length 8192, seed 0, and the exact July optimizer schedule.

### External Prior Art

- YOCO splits an L-layer model into an L/2-layer self-decoder and an L/2-layer cross-decoder. It computes a single global pair `K_hat = LN(X_mid) W_K`, `V_hat = LN(X_mid) W_V`, and all cross-decoder layers reuse those projected tensors while retaining layer-specific query projections. Source: [YOCO paper, Sections 2 and 2.2](https://arxiv.org/abs/2405.05254).
- The paper's self-decoder uses constant-memory attention such as gated retention or sliding-window attention, while the cross-decoder uses causal global cross-attention. The current Marin July architecture already alternates sliding-window and full-causal layers, so this experiment isolates projected K/V sharing without claiming to reproduce the paper's full self-decoder. Source: [YOCO paper, Sections 2.1-3](https://arxiv.org/abs/2405.05254).

### Negative / Failed Leads

- Reusing only the midpoint activation is not classical YOCO: it preserves all per-layer K/V matrices and recomputes projected K/V in every cross layer.
- Increasing the global expert count or global head count is not a close parameter match at d512. One full routed expert in every layer or one query head in every layer overshoots the K/V savings by several times. Matching must be localized and the residual mismatch reported explicitly.
- Reusing the boundary self-attention layer's already-projected K/V directly would remove one additional K/V pair, but it does not match Equation 2 of the paper, which specifies a dedicated shared projection from the midpoint representation.

### Evidence Map

#### Claim: classical YOCO removes per-cross-layer K/V projections after one shared pair

- Support:
  - YOCO paper Equation 2 defines one shared projected K/V pair.
  - YOCO paper Equation 3 gives each cross layer its own Q projection and reuses the shared K/V pair.
  - Marin `model.py` currently gives every block its own `w_k` and `w_v` and only shares `kv_input`.
- Contradictions:
  - This experiment retains Marin's existing self-decoder block types rather than reproducing YOCO gated retention.
- Directness to Marin: high for the cross-decoder mechanism; medium for claims about the complete YOCO architecture.
- Confidence: high.
- Action: implement a single projected midpoint K/V pair and omit K/V weights from later cross layers.

### Recommended Next Experiments

#### 1. Classical projected-K/V sharing

- Minimum experiment: d512 at the existing 750-TPP schedule.
- Baseline/control: the completed exact-July d512 control and fixed-YOCO runs.
- Expected signal: equal or better validation loss with fewer total parameters and one global K/V cache.
- Falsifier: worse validation loss than both existing comparisons at matched tokens.
- Cost/risk: one v5p-8 run; us-central1 capacity/preemption risk.
- Sources: YOCO Sections 2.1-2.2; current Marin fixed-YOCO implementation.

#### 2. Reinvest savings in expert capacity

- Minimum experiment: add localized expert capacity without changing top-k, data order, token horizon, or optimizer schedule.
- Baseline/control: classical projected-K/V sharing.
- Expected signal: recover any quality lost from deleting K/V matrices.
- Falsifier: no validation improvement relative to bare classical YOCO.
- Cost/risk: routing/stat-shape complexity; report exact total and active parameter deltas.

#### 3. Reinvest savings in query-head capacity

- Minimum experiment: add localized query heads at fixed head dimension without restoring extra K/V heads.
- Baseline/control: classical projected-K/V sharing.
- Expected signal: improve cross-decoder capacity while preserving a single shared K/V cache.
- Falsifier: no validation improvement or a throughput regression that dominates quality gains.
- Cost/risk: heterogeneous per-layer query-head shapes; report exact mismatch.

### Hypothesis Queue Update

- Add: classical projected-K/V sharing at d512/750 TPP.
- Add: parameter reinvestment via localized expert capacity.
- Add: parameter reinvestment via localized query-head capacity.
- Revise: call the existing implementation midpoint K/V-input reuse, not classical YOCO.
- Falsify / stop: do not use a global expert/head increase as a purported close match.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| YOCO paper | paper | https://arxiv.org/abs/2405.05254 | Shared projected K/V equations and decoder-decoder split | high | Primary source |
| Marin fixed-YOCO model | Marin code | `experiments/grug/moe_yoco_kv_reuse/model.py` | Current variant shares K/V input but retains all K/V matrices | high | Branch commit `53d8daa08` before this work |
| Marin exact-July recipe | Marin code | `experiments/grug/moe_yoco_kv_reuse/recipe.py` | d512 architecture and 750-TPP schedule | high | Existing monitored comparison |

### Handoff

- Suggested logbook entry: launch three d512 runs—classical, expert-capacity, and query-head-capacity—using the exact existing 750-TPP recipe.
- Open questions: whether localized added capacity should be spread across cross layers or concentrated at the first cross layer; parameter-count tests will choose the closest simple configuration and record the residual mismatch.
- Stop reason: the primary mechanism and experiment matrix are determined; implementation measurements are now more informative than additional literature search.

## 2026-08-16 Launch

- Snapshot: `29056c870`
- Parent: `/kaiyuew/moe-classical-yoco-overtrain-750tpp-8196`
- Monitoring state: `scratch/20260816-1402_classical_yoco_param_match_monitoring_state.json`
- Command:

```bash
/tmp/marin-iris-current-venv/bin/iris --config /tmp/marin-iris-compat.yaml job run \
  --no-wait --preemptible --region us-central1 \
  --job-name moe-classical-yoco-overtrain-750tpp-8196 \
  --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -- python -m experiments.grug.moe_yoco_kv_reuse.experiment_classical_yoco \
  --max_concurrent 3
```

The three runs keep the exact d512 July data order, seed, batch size, token horizon, optimizer schedule, and evaluation cadence. Relative to the exact-July baseline, bare classical YOCO has 262,144 fewer parameters, the added width-171 shared expert has 512 more parameters, and the two localized query heads have 1,024 more parameters.

Launch verification at 2026-08-16 21:04 UTC found the parent running and exactly the three requested v5p-8 children pending with zero failures. Iris reported ordinary us-central1 capacity pressure (`Insufficient TPUs`); no cross-region routing or duplicate submission was made.
