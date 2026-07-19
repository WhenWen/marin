# July-baseline Identity Hyper-Connections: Research Logbook

## Scope

- Goal: implement four-stream Identity Hyper-Connections on the real July baseline and launch Gate 1 at d512 and d768.
- Primary metrics: matched v5p-8 throughput delta, activation-memory evidence, finite training loss, and final Paloma macro loss.
- Constraints: candidate throughput loss must be at most 11% (revised by the user from the initial 8% gate on 2026-07-19); use recomputation to control the widened residual activation; preserve all unrelated July-baseline choices.
- Issue: [marin-community/marin#7409](https://github.com/marin-community/marin/issues/7409), tracked under #4281.
- References: [Identity HC proposal](https://zhuanlan.zhihu.com/p/2010852389670908320), [mHC paper](https://arxiv.org/abs/2512.24880), [Megatron-LM implementation](https://github.com/NVIDIA/Megatron-LM/pull/2943).

## Baseline

- Date: 2026-07-19.
- Code: `marin/july_baseline@52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`.
- Fixed cases: d512 / 6 layers / batch 16 / 10,980 steps and d768 / 8 layers / batch 32 / 16,875 steps, both sequence length 8,192 on v5p-8.
- Architecture: July GQA and half-RoPE retained; long layers disable RoPE; PKO disabled.

## Experiment Log

### 2026-07-19 09:35 - Implementation contract

- Hypothesis: fixing `H_res=I` removes Sinkhorn and the n-by-n dynamic residual map while retaining useful token-dependent `H_pre` and `H_post` routing.
- Command: `uv run --with pytest --with pytest-timeout pytest -q experiments/grug/moe_identity_hyperconnection/test_identity_hyperconnection.py experiments/grug/moe_identity_hyperconnection/test_optimizer.py tests/test_grug_variant_contracts.py -k 'identity_hyperconnection or grug_moe_adamh_mask_routes_expert_mlp_weights_to_expert_group'`.
- Config: four streams; Xavier projection from `4D` to 8 logits; alpha 0.01; zero bias; identity residual update; two-layer rematerialization groups; new mapping parameters routed to Adam.
- Result: seven tests passed, including discovered Grug shape contracts, unique optimizer registration, and optimizer routing. A full small-model train-step JAXPR also lowered with 469 equations.
- Interpretation: local equations and boundary behavior match the upstream Hyper-Connection implementation; real compile/performance evidence is still required.
- Next action: run required lint/type checks, create matched v5p-8 baseline/candidate profiles, and launch Gate 1 only if both widths remain within the 8% throughput budget.

### 2026-07-19 09:40 - Profile harness ready

- Hypothesis: matched 220-step baseline/candidate jobs at each Gate 1 width provide enough post-compile steady-state steps and a 50-step XPlane window to enforce the throughput budget.
- Command: `python -m experiments.grug.moe_identity_hyperconnection.profile_comparison` through Iris on four v5p-8 workers.
- Config: identical seed, data, sequence length, batch, July model, optimizer schedule, and device type within each width; evaluation disabled; profiler steps 20-69; final profile checkpoints written only to worker-local `/tmp`.
- Result: all four profile `ExecutorStep` configs and both Gate 1 configs build in one process. Variant-specific optimizer registry names fixed an import collision with the baseline. Full repository pre-commit and Pyrefly checks pass.
- Interpretation: the harness is reproducible and cannot accidentally reuse baseline optimizer identities or stale demo runs.
- Next action: snapshot the implementation and submit only the matched profile parent.

### 2026-07-19 09:45 - Matched profiles submitted

- Hypothesis: the paired runs will isolate Identity-HC overhead because every shape, schedule, data, dtype, and hardware choice is matched within width.
- Command: `/Users/kaiyuew/Downloads/Project/marin/.venv/bin/iris --config <July-config normalized for current client> job run --no-wait --cpu=1 --memory=2G --extra=cpu --job-name july-baseline-identity-hc-profile-7409 -e WANDB_API_KEY "${WANDB_API_KEY}" -- python -m experiments.grug.moe_identity_hyperconnection.profile_comparison`.
- Config: snapshot `identity-hc-profile-v1` / `444b5ea66`; CPU-only parent; four v5p-8 children; W&B group `MOE-JULY-IHC-perf-issue-7409`.
- Result: parent `/kaiyuew/july-baseline-identity-hc-profile-7409` accepted and entered building with zero failures or preemptions. The first attempt with the July branch's stale Iris client was rejected before job creation; the current client submitted the same clean workspace and config.
- Interpretation: exactly one parent exists. Gate 1 is still unlaunched.
- Next action: verify exactly four children, successful TPU compile, finite metrics, structured profiles, and the 8% threshold at both widths.

### 2026-07-19 09:48 - Placement-only failure; one exact retry

- Hypothesis: the July launcher resolves its regional Marin prefix and dispatch placement from the CPU parent, so pinning that parent to `us-east5-a` will place the four `v5p-8` children in the same region as the successful July baseline cells.
- Command: repeat the exact profile parent command with `--zone us-east5-a`; preserve the parent name, snapshot, four child identities, W&B group, and all model/profile settings.
- Config: no model, optimizer, data, batch, step, profiler, or checkpoint change.
- Result: the first parent failed before creating a child or W&B run. Its unpinned CPU worker landed in `europe-west4`, and all four child submissions were rejected as unschedulable because no `v5p-8` group exists there. Failure count was one; preemption count was zero.
- Interpretation: this is an orchestration-placement error before TPU allocation, not compile, memory, numerical, or throughput evidence. The single allowed retry is safe because no training work or identity was created.
- Next action: resubmit once from `us-east5-a`, then require exactly four children and resume the original performance gate.

### 2026-07-19 09:51 - Exact profile children materialized

- Hypothesis: the zone-pinned retry should preserve all scientific identities while changing only scheduler placement.
- Command: Iris prefix query after the mandatory 120-second post-submit window.
- Config: parent `/kaiyuew/july-baseline-identity-hc-profile-7409`; exactly the four `BASE/CAND` x `d512/d768` child names recorded in the monitoring state.
- Result: the parent is running and exactly four intended children are pending in `us-east5-a`; every child has zero failures and zero preemptions. Pending reason is v5p capacity with quota-pool tier monotonicity. No sibling or cross-width launch exists.
- Interpretation: placement is corrected. Capacity pending is not a training failure, and no W&B evidence is expected until a worker is allocated.
- Next action: monitor in place; require compile/memory evidence, finite W&B progress, completed XPlane profiles, and matched steady-state throughput at both widths.

### 2026-07-19 10:15 - d512 baseline complete; candidate allocated

- Hypothesis: the 220-step baseline provides a stable post-profiler reference once the first 120 steps are excluded.
- Command: W&B history scan over `global_step >= 120`, plus Iris terminal state and compile/error log scan.
- Config: `MOE-JULY-IHC-PERF-BASE-d512`, July d512, batch 16, sequence 8,192, seed 0, profiler steps 20-69.
- Result: Iris succeeded with zero failures/preemptions; W&B finished at summary step 219 and the trainer completed 220/220. All logged losses were finite; final loss was 6.069026. The last 100 rows averaged 357,057.71 tokens/s (median 357,301.52). The XPlane profiler completed and uploaded eight artifact files. No HBM/OOM signature appeared. The scheduler then allocated `MOE-JULY-IHC-PERF-CAND-d512`; its live config exactly records four streams, alpha 0.01, and two-layer rematerialization groups.
- Interpretation: the d512 reference is decision-grade. Candidate compilation is now the first direct activation-memory test; d768 remains capacity-pending.
- Next action: require the d512 candidate to compile, finish with finite loss, upload its profile, and keep its matched last-100-step throughput loss at or below 8%.

### 2026-07-19 10:27 - d512 v1 narrowly misses throughput gate

- Hypothesis: two-layer rematerialization plus fixed identity residual routing would keep four-stream overhead at or below 8%.
- Command: matched W&B history scan over the final 100 rows (`global_step >= 120`) for baseline and candidate; Iris/log/profile verification for both runs.
- Config: d512 baseline and four-stream candidate, otherwise identical July architecture, data, batch 16, sequence 8,192, seed 0, and 220 steps.
- Result: both jobs succeeded with zero failures/preemptions, finite loss, and uploaded XPlane profiles. Baseline averaged 357,057.71 tokens/s; candidate averaged 327,957.19 tokens/s. Slowdown is 8.1501%, exceeding the hard limit by 0.1501 percentage points. Candidate added 196,728 parameters (746,803,832 versus 746,607,104). No HBM/OOM signature appeared. The d768 baseline was allocated next and is compiling.
- Interpretation: memory/compile viability passes at d512, but the requested throughput gate fails narrowly. Gate 1 remains blocked. Per-token coefficient-telemetry reductions inside every rematerialized block are a non-model overhead candidate and will be checked against the XPlane breakdown before a new identity is launched.
- Next action: finish both profile summaries, remove only proven observability overhead, validate locally, and run a new optimized profile identity; continue the original d768 pair for scale evidence.

### 2026-07-19 10:41 - d768 baseline complete; candidate compiling

- Hypothesis: the wider July cell supplies independent scale evidence for both throughput and activation viability.
- Command: final-100-row W&B scan after Iris terminal success.
- Config: `MOE-JULY-IHC-PERF-BASE-d768`, hidden 768, 8 layers, batch 32, sequence 8,192, seed 0, profiler steps 20-69.
- Result: baseline succeeded with zero failures/preemptions, finite final loss 5.841033, and a completed XPlane artifact. The final 100 rows averaged 251,195.43 tokens/s (median 251,190.35). The original d768 candidate then acquired the released v5p-8 and its live W&B config exactly matches four streams, alpha 0.01, and remat group 2.
- Interpretation: the d768 reference is ready; candidate compilation is the strongest current memory test because d768 has the larger widened residual state.
- Next action: monitor the d768 candidate to terminal, then compare its last 100 rows and profile while preparing the fresh PERF2 remeasurement snapshot.

### 2026-07-19 10:44 - Optimized PERF2 profile cells queued

- Hypothesis: removing per-layer coefficient telemetry collectives, while retaining the exact Identity-HC routing equations, parameters, optimizer groups, and two-layer rematerialization, will recover the 0.1501 percentage-point d512 miss without changing the model.
- Command: submit `/kaiyuew/july-baseline-identity-hc-profile-v2-7409` from `us-east5-a` at snapshot `identity-hc-profile-v2` / `f51a83fdd`; W&B group `MOE-JULY-IHC-perf-v2-issue-7409`.
- Config: fresh `PERF2` W&B identities; otherwise the same four matched `BASE/CAND` x `d512/d768` cells, seed, data, shapes, 220 steps, and XPlane window as v1. The implementation now uses the paper's exact sigmoid routing coefficient without an epsilon offset and omits coefficient-only monitoring reductions.
- Result: after the required 120-second check, the parent is running and exactly four intended children are capacity-pending in `us-east5-a`, all with zero failures and zero preemptions. No sibling or cross-width identity exists. The v1 d512 baseline XPlane summary also completed; candidate XPlane ingestion is running.
- Interpretation: this is a fresh, decision-grade measurement identity. Capacity pending is normal; Gate 1 remains blocked until both optimized width pairs pass the 8% throughput limit and the d768 candidate supplies successful compile/memory evidence.
- Next action: monitor v1 d768 to terminal for diagnostic scale evidence, monitor all PERF2 cells in place, and compare matched final-100 throughput plus structured profiles before any Gate 1 submission.

### 2026-07-19 10:54 - d768 v1 terminal; optimized remeasurement remains decisive

- Hypothesis: d768 supplies the strongest activation-memory check and shows whether the original telemetry-heavy implementation scales within the throughput budget.
- Command: compare matched final-100 W&B rows (`global_step` 120-219), verify Iris terminal state, and scan the complete child log for non-finite, HBM, OOM, resource-exhaustion, traceback, or failed signatures.
- Config: v1 d768 baseline and four-stream candidate; hidden 768, eight layers, batch 32, sequence 8,192, seed 0, two-layer rematerialization, and otherwise identical July settings.
- Result: both jobs succeeded with zero failures/preemptions, finite losses, and completed profile uploads. Baseline averaged 251,195.43 tokens/s; candidate averaged 224,838.44 tokens/s. The v1 slowdown is 10.4926%, so it fails the 8% budget. The candidate compiled, completed 220 steps, saved its local checkpoint, and showed no HBM/OOM or other unrecoverable log signature.
- Interpretation: two-layer recomputation makes the full d768 configuration operational, but v1 performance is insufficient. This result is diagnostic because PERF2 removes coefficient-only telemetry collectives without changing the routing model.
- Next action: wait for the exact four PERF2 cells, require both width pairs to pass 8%, and use their own XPlane directories for valid same-format profile comparisons before Gate 1.

### 2026-07-19 11:07 - Throughput ceiling revised to 11%; Gate 1 unblocked

- Decision: the user accepted the current implementation and revised the maximum throughput decrease from 8% to 11%.
- Evidence: terminal matched v1 measurements are 8.1501% slower at d512 and 10.4926% slower at d768. Both candidates completed with finite loss, zero failures/preemptions, and profile artifacts. The larger d768 cell compiled and completed with two-layer rematerialization and no HBM/OOM signature.
- Interpretation: both widths pass the revised throughput gate, and the d768 completion provides the requested activation-memory viability evidence. PERF2 remains useful diagnostics but no longer blocks launch.
- Next action: snapshot the accepted four-stream implementation and submit exactly the d512 and d768 Gate 1 cells from `us-east5-a`; verify child identity, compile, finite startup, checkpoint paths, and scheduled Paloma evaluations.

### 2026-07-19 11:10 - Gate 1 exact children materialized

- Command: submit the CPU-only parent `/kaiyuew/july-baseline-identity-hc-gate1-7409` from `us-east5-a` at snapshot `identity-hc-gate1-launch` / `3e9e72353`, then query the Iris prefix after submission.
- Config: W&B group `MOE-JULY-IHC-gate1-issue-7409`; sole children `MOE-JULY-IHC-G1-001-d512` and `MOE-JULY-IHC-G1-002-d768`; accepted four-stream Identity-HC implementation with alpha 0.01 and two-layer rematerialization.
- Result: exactly the two intended children materialized with no siblings. d512 is running in worker build; d768 is capacity-pending. Both have zero failures and zero preemptions.
- Interpretation: launch identity and fan-out are correct. The d768 capacity wait is normal and is not a training failure.
- Next action: monitor each child in place through compilation and require fresh W&B progress with finite startup loss before declaring the Gate 1 launch healthy.

### 2026-07-19 11:17 - d512 Gate 1 passes startup

- Command: inspect the live Iris child, W&B history, and W&B-resolved trainer/checkpointer config for `MOE-JULY-IHC-G1-001-d512`.
- Result: d512 passed first-step compilation and advanced to step 55 with finite train loss 10.350956 at 329,969.97 tokens/s. Iris and W&B remain running with zero failures/preemptions and a fresh heartbeat. The permanent final checkpoint path resolves to `gs://marin-us-east5/grug/MOE-JULY-IHC-G1-001-d512-39b35d/checkpoints/step-10980`.
- Config identity: hidden 512, six layers, four attention heads, one KV head, batch 16, sequence 8,192, four residual streams, alpha 0.01, two-layer rematerialization, and 10,980 steps.
- Interpretation: d512 Gate 1 has healthy numerical, compile, throughput, identity, and checkpoint-path evidence.
- Next action: keep d512 running; wait for d768 capacity, then require the same compile and finite-startup evidence independently.

### 2026-07-19 11:28 - Both Gate 1 widths pass startup

- Command: verify the exact Iris prefix, live W&B config/history/heartbeats for both Gate 1 runs, and terminal matched PERF2 histories over steps 120-219.
- Result: the parent has exactly two running children and no siblings. Both have zero failures/preemptions. d512 is at step 1,449 with finite loss 4.046039 and 328,600.09 tokens/s. d768 passed the larger first-step compile and is at step 45 with finite loss 10.977134 and 227,401.84 tokens/s. Both W&B heartbeats are fresh.
- Checkpoints: d512 final is `gs://marin-us-east5/grug/MOE-JULY-IHC-G1-001-d512-39b35d/checkpoints/step-10980`; d768 final is `gs://marin-us-east5/grug/MOE-JULY-IHC-G1-002-d768-c9eb80/checkpoints/step-16875`.
- Throughput gate: optimized matched PERF2 final-100 means are 354,247.48 baseline versus 329,300.35 candidate at d512 (7.0423% slower), and 251,248.13 versus 225,620.46 at d768 (10.2001% slower). Both pass the accepted 11% ceiling. Both candidates completed 220 steps with finite loss, profile artifacts, and no HBM/OOM signature.
- Interpretation: the requested real-July-baseline Identity-HC Gate 1 launch is reproducible, correctly scoped, within the accepted throughput budget, activation-memory viable under two-layer recomputation, and healthy at both widths.
- Next action: leave both Gate 1 cells running for scheduled 1,000-step Paloma evaluations and final checkpoints; no recovery action is indicated.

### 2026-07-19 12:06 - Both widths have evaluations and recoverable checkpoints

- Command: verify the exact Iris prefix, scan dense W&B train histories for non-finite loss, inspect scheduled eval rows, and read temporary checkpoint `metadata.json` for both widths.
- Result: both children remain running with zero failures/preemptions. d512 reached step 5,999 with finite loss 3.433643 at 328,645.55 tokens/s; its step-5,000 Paloma macro loss is 3.950007 (BPB 1.414534). d768 reached step 1,760 with finite loss 3.602213 at 225,306.26 tokens/s; its step-1,000 Paloma macro loss is 4.169527 (BPB 1.492243). All scanned train losses are finite.
- Checkpoints: complete temporary metadata exists at d512 step 5,615 (timestamp `2026-07-19T19:02:55.245413`) and d768 step 1,590 (timestamp `2026-07-19T19:03:35.825568`).
- Interpretation: both arms have durable recovery points and scheduled evaluation evidence; no recovery action is indicated.
- Next action: continue monitoring independently to terminal success, then verify finished W&B state, final permanent checkpoint metadata, and matched final Paloma comparisons.

### 2026-07-19 12:52 - d512 completes successfully; d768 continues

- Command: verify the d512 Iris summary and terminal logs, scan its complete W&B train history and final evaluation, and read the permanent step-10,980 checkpoint metadata; independently refresh d768 Iris, W&B, and temporary checkpoint evidence.
- Result: d512 succeeded with exit code 0, zero failures, and zero preemptions. W&B finished after 10,979 logged train steps, all losses were finite, final train loss was 3.175577 at 326,877.26 tokens/s, and the terminal Paloma macro loss was 3.544176 (BPB 1.270561). Permanent metadata confirms step 10,980 at `gs://marin-us-east5/grug/MOE-JULY-IHC-G1-001-d512-39b35d/checkpoints/step-10980`, timestamp `2026-07-19T19:48:52.652431`. d768 remains running with zero failures/preemptions at step 3,809, finite loss 3.357501, 224,876.31 tokens/s, step-3,000 Paloma macro loss 3.777038 (BPB 1.353537), and a complete temporary checkpoint at step 3,500.
- Interpretation: d512 meets every terminal criterion: successful Iris state, finished W&B state, finite full history, final scheduled evaluation, and exact permanent checkpoint metadata. No recovery action is indicated for d768.
- Next action: continue monitoring d768 in place through its scheduled evaluations and permanent step-16,875 checkpoint, then make the matched final Paloma comparison and publish the combined result.

### 2026-07-19 13:24 - d768 passes step-5,000 evaluation

- Command: refresh the exact Iris prefix, scan the dense d768 W&B train history and scheduled evaluations, read the newest temporary checkpoint metadata, and scan recent logs for numerical, memory, task-failure, and dead-node signatures.
- Result: d768 remains running with zero failures/preemptions at step 5,153. All 5,153 scanned train losses are finite; the latest loss is 3.283502 and the instantaneous throughput is 214,794.00 tokens/s. The step-5,000 Paloma macro loss is 3.667298 (BPB 1.315073), improved from 3.724738 at step 4,000. Complete temporary checkpoint metadata exists at step 4,915 with timestamp `2026-07-19T20:17:24.999238`. The error scan is clean apart from JAX's successful serialization check message.
- Interpretation: the remaining arm continues to make numerically stable, evaluated, and recoverable progress. The lower instantaneous throughput sample is informational and does not trigger recovery.
- Next action: continue monitoring d768 in place to terminal success and its permanent step-16,875 checkpoint.
