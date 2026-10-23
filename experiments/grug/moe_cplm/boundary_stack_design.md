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
b=[0,0,6] in the original v1 run, giving approximately
[0.002467,0.002467,0.995067]. Native q/k/v, decay,
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

## 2026-10-23 qualification amendment and measured result

The first two TPU attempts failed the strict gradient check against the TPU
token oracle: eight entries exceeded tolerance, with maximum violating absolute
error 2.26e-5. Fused and unfused chunk paths agreed much more closely, indicating
the discrepancy was shared by chunk preparation versus token recurrence.
The independent oracle now executes the same literal equations in CPU FP64,
avoiding compounded TPU exponential/rounding errors. Thresholds stay unchanged.
CPU numerical checks perform zero training updates. A separate rank-128 CPU
mixed interpreter diagnostic had one value-gradient mismatch of 5.52e-4 against
CPU AD; this is retained as a CPU mixed-numerics limitation. The target TPU mixed
path passes the strict AD comparison, with maximum relative L2 5.51e-7.

TPU job boundary-stack-kernel-20261023-v3 succeeded with source c6455b6fd and
artifact identity abba6662cf8e448292a587a330808f64. Its FP32 candidate passes the
FP64 oracle at atol=1e-5, rtol=1e-4. Mixed-versus-FP64 relative L2 is at most
0.00373 across all output/gradient leaves. Hard-stay output and final state
match native mixed KDA exactly.

Seven-repetition medians on one TPU v4 chip, B1,T8192,H4,R=V=128,C16:

| Path | Native KDA | Boundary mixing | Ratio |
| --- | ---: | ---: | ---: |
| Forward | 1.525 ms | 1.738 ms | 1.140 |
| Saved-residual backward | 3.075 ms | 3.774 ms | 1.227 |
| Combined forward/backward | 3.774 ms | 4.722 ms | 1.251 |

Recurrent state is 262,144 bytes in both. Backward residuals are 442,114,056
versus 594,157,576 bytes (an extra 145 MiB including wrapper/preparation
residuals); combined temporary allocation is 343,953,920 versus 489,293,312
bytes. Compilation takes 1.442/4.013/4.235 seconds for native and
1.454/4.664/3.678 seconds for boundary forward/backward/combined. These are
operator measurements, not end-to-end LM throughput or quality results.

## 2026-10-23 initialization diagnosis and bias sweep

The bias-6 run is active on v4-32 with process identity
13aa3067db7d4c99838d99c61da50fab. Through optimizer step 332, its maximum
absolute training-NLL difference from the reused seed-0 baseline is 0.0102062.
Initial losses are 11.86815071 versus 11.86813641; step-332 losses are
4.20034504 versus 4.19646454. All four controller weight gradients are nonzero
before training and all four zero-initialized matrices change after two updates.
Mean stay probabilities at step 332 are 0.9981729, 0.9946595, 0.99931455 and
0.99999857 for layers 0, 1, 2 and 4. These measurements establish a connected,
trainable controller that remains strongly stay-dominated, not useful stack
behavior or completed quality evidence.

The user requested bias tuning. Freeze three fresh seed-0 conditions with
initial action bias [0,0,b], b in {0,2,4}; reuse the running b=6 condition and
completed native baseline. Initial stay probabilities are respectively
0.333333, 0.786986, 0.964663 and 0.995067. Keep zero controller weights and
identical native parameters, copy head, data ordering, optimizer, mixed
precision and state size. Both the controller weights and bias remain trainable.
Bias only changes the initial controller distribution;
the previously qualified operator and its gradients are unchanged.

Run exactly 1,000 updates per new condition (131,072,000 tokens), preserving
the original 10,676-update optimizer/LR schedule through the native
stop_after_steps field. Native final callbacks retain checkpoint and validation
results at this prefix endpoint. Use three independent, concurrently submitted
v4-32 jobs for b=0,2,4, as requested by the user. Levanter's
process-0 exit handler completes its Iris job, so each condition needs a separate
job identity. The submit-host launcher submits all conditions before waiting
for completion. Each condition and each restart receives a fresh W&B and
checkpoint identity. A job restart repeats only its own bias condition from
initialization; it does not resume any earlier trajectory. Keep default
Iris preemption/retries and no wall-clock limits. The existing b=6 run continues.

Compare Paloma macro validation NLL after 1,000 updates as the primary tuning
criterion among b=0,2,4,6; the native baseline is a reference, not a bias candidate.
Use the reused baseline and b=6 validation at the same 1,000-update endpoint
(native metric step 999). Also report Uncheatable validation NLL,
last-100-update mean training NLL, each layer's mean push/pop probabilities
averaged over the same last 100 training updates (each step averages its batch
and chunk boundaries), and throughput. Report every
condition, including divergence and near-stay collapse. This is a single-seed
early-budget comparison; its winner is provisional and does not establish the
best full-budget bias. The expected tradeoff is more controller sensitivity at
smaller bias versus more destructive initial mixing; no winner is predicted.

## 2026-10-23 measured prefix results and full-budget extension

All three concurrent prefixes completed 1,000 updates. Measured seed-0
validation NLL at native metric step 999:

| Condition | Paloma | Uncheatable | Mean train NLL, last 100 updates |
| --- | ---: | ---: | ---: |
| Native baseline | 3.986214 | 3.890764 | 3.803172 |
| Bias 6 | 3.976407 | 3.880238 | 3.796629 |
| Bias 0 | 3.972532 | 3.874654 | 3.792122 |
| Bias 2 | 3.971663 | 3.873104 | 3.791436 |
| Bias 4 | 3.967944 | 3.867916 | 3.790402 |

Bias 4 has the lowest prefix Paloma NLL and becomes the provisional initializer
default. This single-seed result does not establish a full-budget winner.
Every comparison launch specifies its bias explicitly.

The user subsequently authorized completing all four biases to 10,676 updates.
Continue bias 0/2/4 concurrently on independent v4-32 jobs from their committed
step-1000 checkpoints; retain the already running bias-6 trajectory. Restore
the full native training state with partial restore disabled and require step
1000 before the first optimizer update. Native data loading uses the restored
step; the original optimizer/LR schedule remains unchanged. Each continuation
gets a fresh W&B segment linked by its parent checkpoint, preserving its prefix
curve. Default Iris retries/preemption remain enabled without wall-clock caps;
a retry reloads the specified parent checkpoint into a new segment. Record
restoration and subsequent progress before claiming a successful continuation.
