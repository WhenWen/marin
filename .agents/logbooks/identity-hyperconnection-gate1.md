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

### 2026-07-19 09:55 - Profile harness ready

- Hypothesis: matched 220-step baseline/candidate jobs at each Gate 1 width provide enough post-compile steady-state steps and a 50-step XPlane window to enforce the throughput budget.
- Command: `python -m experiments.grug.moe_identity_hyperconnection.profile_comparison` through Iris on four v5p-8 workers.
- Config: identical seed, data, sequence length, batch, July model, optimizer schedule, and device type within each width; evaluation disabled; profiler steps 20-69; final profile checkpoints written only to worker-local `/tmp`.
- Result: all four profile `ExecutorStep` configs and both Gate 1 configs build in one process. Variant-specific optimizer registry names fixed an import collision with the baseline. Full repository pre-commit and Pyrefly checks pass.
- Interpretation: the harness is reproducible and cannot accidentally reuse baseline optimizer identities or stale demo runs.
- Next action: snapshot the implementation and submit only the matched profile parent.
