# July-baseline Identity Hyper-Connections: Research Logbook

## Scope

- Goal: implement four-stream Identity Hyper-Connections on the real July baseline and launch Gate 1 at d512 and d768.
- Primary metrics: matched v5p-8 throughput delta, activation-memory evidence, finite training loss, and final Paloma macro loss.
- Constraints: candidate throughput loss must be at most 8%; use recomputation to control the widened residual activation; preserve all unrelated July-baseline choices.
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
