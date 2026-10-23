# Boundary head mixing in native Belay CPLM Plus

Protocol frozen 2026-10-23 before qualification and training. Theory
[#59](https://github.com/WhenWen/astra-d8-research/issues/59); motivating task
scoped_binding@1 [#58](https://github.com/WhenWen/astra-d8-research/issues/58)
is not evaluated here. The user explicitly selected boundary head mixing,
retaining all four native head updates, and authorized direct native LM tests
without synthetic gates. The top-only protected-frame proposal is rejected.

## Fixed architecture and algorithm

All four heads run unchanged native KDA inside each 16-token chunk. Their
updated states U are mixed once, after emitting that chunk's outputs:

    S_next = stay*[U0,U1,U2,U3]
           + push*[0,U0,U1,U2]
           + pop *[U1,U2,U3,0].

Actions are softmax(x_last W+b), using the last token's normalized KDA layer
input. Softmax coordinates are [push, pop, stay]. They affect only the following
chunk, so no future token influences an earlier output. W is zero initialized,
b=[0,0,6], giving approximately [0.002467,0.002467,0.995067]. Native q/k/v, decay,
beta, convolutions, output norm/gate/projection, MLA, MoE, AttnRes and the
CPLM Plus copy head remain unchanged. The four KDA layers each add 1,539
parameters, totaling 6,156. All heads continue writing; lower slots are not
immutable parent snapshots. Hard stay recovers native KDA.

State remains four FP32 128-by-128 matrices: 65,536 floats or 256 KiB per
sequence/layer. All four heads must reside on one device (model axis size 1).
Document segmentation uses native per-token masks, including resets inside
a chunk. The boundary action applies to the final document's updated states.
This training/prefill interface requires complete chunks; partial-chunk decode
would additionally track phase and defer the boundary action.

Preparation reuses native _chunk_prep_xla with four independent C-by-C systems.
One Pallas stream owns four states in VMEM. It invokes the native per-head
state update/readout and then performs the three-way mixture on chip.
Let M=stay I+push L+pop R, where L/R are zero-filled shifts. Backward uses
dU=M^T dS_next, then the native per-head reverse formulas. Each action gradient
is <dS_next, corresponding shifted U>. Per-value partials are reduced by the
wrapper. The head arithmetic and BF16/FP32 policy match native KDA.

Native preparation and state GEMM FLOPs are unchanged. Boundary routing and
its adjoint add O(4RV) arithmetic per chunk, or O(4RV/C) per token, plus three
inner-product reductions. The ordered boundary depth remains T/C. Head-level
stream parallelism is reduced because a stream owns all four heads; sequence
groups remain independent. Boundary routing requires no network communication
under model-axis size 1. The controller adds a 512-by-3 projection and a
three-way softmax per boundary/layer; its 6,156 FP32 parameter gradients add
24,624 bytes of gradient payload to distributed synchronization per update
(wire traffic depends on the collective algorithm). Speed is measured.

The first implementation saves U for action gradients as well as native
incoming states and innovations. This adds T/C times 256 KiB of saved
activations (128 MiB at T8192), not recurrent state. Saving U avoids four
extra reconstruction GEMMs per chunk in backward. Report temporary and
residual memory separately from recurrent state.

## Qualification and training

Use FP32 literal token recurrence plus boundary mixing as the independent
oracle. The candidate's FP32 preparation and fused state kernel must match
outputs, final state and all continuous input/state gradients at atol=1e-5,
rtol=1e-4; check hard stay, hard routing, document resets and chunk continuation.
The deployed mixed fused path must match the executed native-prepared scan
and its VJP at the same strict tolerances. Separately, each mixed-versus-literal
FP32 output/gradient leaf must have relative L2 error <=5% or maximum absolute
error <=1e-4; report all errors, including those below these bounds. Disable XLA
excess-precision elision consistently; do not loosen FP32 thresholds.

Measure native and candidate on TPU v4, B1,T8192,H4,R128,V128,C16, using the
same normalized random fixture, segment ids and initial states. Report seven
drained repetitions after warmup for forward, saved-residual backward and
combined forward/backward, compilation and memory. No tile sweep is planned.

After qualification, start one fresh seed-0 native d512 LM run for 10,676
updates, global batch 16*8192 on TPU v4-32 (16 physical chips). Reuse baseline
4434036161f49ae783d3, native identity 1e589eb42c014e2497196c73e553b6c0:
[baseline W&B](https://wandb.ai/marin-community/auto-understanding-arch/runs/plus-d512-s0-v1-1e589eb42c014e2497196c73e553b6c0).
Freeze belay_recipe.json, july_data.json, tokenizer_identity.json and native
copy-head settings. No baseline retraining. Compare equal tokens, report the
additional controller parameters and any throughput/memory change. Optimizer
schedule is build_config(width=512,steps=10676,batch_size=16); new action
weights use native KDA Adam routing, not MuonH.

The candidate is plus-boundary-stack-kda, initialized fresh with native seed 0;
all shared parameter initialization and the data order, copy-head settings and
validation schedule inherit train_belay.build_config from the reused baseline.
Frozen SHA256 identities are belay_recipe.json
1c20b82a2fd4a37b7e30bc0de7e78354d3c3ed5ca132359ade1f4cf47e976fb5,
july_data.json f6a1300f821eabcdd670556fc886d7d930f9a3a9ae64ff175568d7041cd3b57b,
and tokenizer_identity.json c64ccac3589298ea8f0d8ba05e6f6e1c7c3ce7a969dd2a3b1c4374c373b27f17.
The launch contract saves source hashes for the actual wrapper, kernel,
training entrypoint and copy-head implementation; resolved-config.json records
the complete model/optimizer/data/evaluation settings before updates.

Use online W&B, a non-training logging preflight, incremental regional loss
logs, the inherited validation/checkpoint schedule and fresh run/checkpoint
identities after each process restart. This entrypoint does not resume: a restart
initializes model, optimizer, RNG, data position and step count afresh, and runs
10,676 updates in that new trajectory. Interrupted attempts remain separate
curves and cannot be counted as resumed progress. Preserve Iris default preemption and
retry behavior; impose no queue or experiment wall-clock limit. A full-model
zero-update preflight checks finite loss/gradients and nonzero copy/action
gradients before training. Present partial loss curves when reporting launch
progress; this launch request does not establish completed LM quality.
