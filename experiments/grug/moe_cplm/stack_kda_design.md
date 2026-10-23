# Depth-four KDA inside Belay CPLM Plus

Protocol frozen 2026-10-23, before implementation or accelerator execution.
Theory: https://github.com/WhenWen/astra-d8-research/issues/59.
Motivating task: scoped_binding@1, issue #58; not evaluated by this LM trial.
The user requested depth four and previously explicitly waived synthetic gates
for direct native LM testing. This protocol tests learned native-text use, not
complete synthetic-task representability or learned symbolic-stack correctness.

Start from Marin commit 9324f95fb4c4cf07ddcb3f0510a1153b0aa94db2 and native
Belay CPLM Plus d512, six layers. Replace the memory recurrence inside KDA
layers 0, 1, 2 and 4. Retain all four heads, each with key/value dimension 128.
Each head has four 128-by-128 frames, initially zero. MLA layers, convolutions,
query/key/value projections, learned decay/beta, output normalization/gating,
MoE, AttnRes and CPLM Plus head keep their original computation. The previous
final-residual Frame KDA branch is disabled. This increases recurrent state from
256 KiB to 1 MiB per sequence per KDA layer, not counting training intermediates.

For U(M)=(I-beta*k*k^T)diag(exp(g))M+beta*k*v^T, the actions on incoming
frames [M0,M1,M2,M3] are push=[U(M0),M0,M1,M2],
pop=[M1,M2,M3,0], stay=[U(M0),M1,M2,M3], and
replace=[beta*k*v^T,M1,M2,M3]. Their softmax-weighted sum is the new state.
Read q^T M0 after the action. Top-only reading retains the native KDA read
shape and gives saved frames the role of inactive contexts. All controls depend
only on the causal input to that KDA layer. Packed-document starts reset all
frames before the current write. States do not cross training batches.

New per-layer action logits are x W_actions+b_actions, reshaped by head.
W_actions is zero-initialized and b_actions=[0,0,6,0] for each head in
push/pop/stay/replace order. Stay initially has probability about 0.99262.
All new parameters are learned by the same next-token NLL, with no auxiliary
loss, oracle actions or warm-start checkpoint. The initializer preserves every
inherited random draw. Soft actions change the initial function slightly;
exact initial loss equality with Plus is not required. The four modified layers
add 32832 parameters. Hard stay is the native KDA algebraic limit, which is
tested numerically; finite softmax logits do not attain that limit exactly.

Hard push saves the parent without changing it; a later matching hard pop
restores it if no action discarded it and no depth overflow occurred. Soft
mixtures can dilute memories. A four-frame stack does not represent unbounded
nesting. These properties do not prove that language-model training learns
useful actions.

## Exact structured chunk algorithm

Flatten key and depth to N=128*4. A token transition is
S'=E S+u(g_write^T-f^T S). E is block-diagonal over key channels, with
tridiagonal 4-by-4 depth blocks. The correction has rank one. For each chunk,
propagate E and write impulses to construct the strict-lower interaction K,
incoming-state erase Q, transported writes U, and read coefficients. Solve
(I+K)Z=G-Q S_in. The outgoing state is F S_in+U Z, where F is the
product of the E operators. No E inverse is used, including at document resets.

Chunk preparation runs across all chunks. Propagate boundary states sequentially
using the small depth blocks and rank-C correction; then emit all chunk outputs
in parallel. Start at C=32; non-training TPU comparisons may choose C=16 or 64
without changing the mathematical recurrence. There is no dense N-by-N prefix
scan and no claim of logarithmic full-sequence depth. Backward propagates
boundary cotangents in reverse and rematerializes local chunks in bounded
batches. Verify all input and initial-state adjoints, including the action
controls. Control projection, local gradients and memory traffic count toward
the measured cost.

Per group, chunk preparation has O(C^2 R D+C R D^2) transport work, plus
the triangular solve and its matrix products. State propagation costs
O(R D^2 V+C R D V) per chunk. Fixed C,D,R,V give linear work in sequence
length, but the critical path retains T/C boundary steps. Retaining all incoming
states costs O((T/C)H D R V); rematerialization avoids storing all token states.
No new sequence/head communication is introduced at the native EP1/model1
layout. Action weights require the ordinary replicated gradient reduction.
Generation cache integration is not added; kernel streaming continuation must
match full prefill, while full-model generation remains unsupported as in Plus.

## Checks and native comparison

First run remote CPU non-training output and gradient comparisons against
independent explicit branch enumeration, hard parent restoration, resets,
causality and streaming. Then run TPU reference comparisons and drained
forward/backward timings at B1,T8192,H4,R128,V128,D4. Preserve native JAX/XLA
settings. Record compilation separately, temporary memory and raw repetitions.
Numerical tolerances are atol=1e-5, rtol=1e-4 for normalized FP32 checks;
report BF16 model-boundary checks separately. No optimizer steps occur in
kernel qualification. Full-model first-batch loss and all gradients must be
finite, including nonzero action-weight gradients.

The native comparison is one fresh seed-0 run, 10676 updates, global batch
16*8192, on 16 physical TPU v4 chips. It uses the same frozen data/tokenizer,
optimizer, validation schedule and Plus head as the historical seed-0 baseline
4434036161f49ae783d3, native identity 1e589eb42c014e2497196c73e553b6c0.
Do not retrain that baseline. Keep online W&B, native incremental metrics,
checkpoints and all eleven periodic/final validations. No queue, job or
watchdog time limit is imposed; preserve Iris default preemption retries.
Each process restart has a fresh W&B identity; do not claim resume without
verified checkpoint restoration. No GPU allocation is acquired or released.

Prediction: protected per-layer contexts may lower final validation NLL; this
is unmeasured. Report token-weighted NLL, all-domain/Paloma/Uncheatable macro
NLL, each domain, matched-step throughput, memory and full loss curves.
There is no promised 10% overhead bound. Kernel timings must not be presented
as whole-model overhead. One historical seed comparison is a discovery pilot,
not a significance test or a claim of task #58 success.

## Definitions for reproduction

The entrypoint is `experiments.grug.moe_cplm.train_belay`, with its frozen
`belay_recipe.json`, `july_data.json`, and `tokenizer_identity.json` at the
source parent above. `build_config(width=512, steps=10676, batch_size=16)`
resolves the inherited overrides. The historical W&B run is
https://wandb.ai/marin-community/auto-understanding-arch/runs/plus-d512-s0-v1-1e589eb42c014e2497196c73e553b6c0.
CPLM Plus is the Copy-Paste Language Model prediction head with predecessor
keys and normalized case-class emissions. MLA is multi-head latent attention;
AttnRes is the inherited attention-based residual aggregation. These remain
unchanged. The previous experiment added a separate memory only before the
prediction head; that branch is absent here.

In the equations, H=4 counts independent heads (groups), D=4 is stack depth,
R=128 is key dimension, V=128 is value dimension, T is token length, and C is
chunk length. Each head's state is FP32 with layout [R,D,V]. The action input x
is the width-512 normalized attention input passed to KimiDeltaAttention,
before its q/k/v short convolutions. W_actions has shape [512,16] and its bias
has shape [4,4]. Native short convolutions and SiLU still form q/k/v; keys and
queries use the native L2 normalization, and queries also scale by 1/sqrt(R).
Values retain native scaling. g has R channels per head and beta is scalar.

Write p,pop,s,r for push/pop/stay/replace probabilities, a=p+s, c=s+r,
and w=p+s+r. For channel i, E_i is a D-by-D matrix with top row
[a*exp(g_i),pop,0,0]. Rows j>0 have p on column j-1, c on column j,
and pop on column j+1 when it exists. Other entries are zero. Let e0 denote
the top-depth basis vector. Then u=k tensor e0, f=(a*beta*exp(g)*k) tensor e0,
and g_write=beta*w*v. A reset multiplies E and f by zero before the write.

For chunk indices 1..C, F(b:a)=E_b...E_a and an empty product is identity.
K[t,i]=f_t^T F(t-1:i+1)u_i for i<t, zero otherwise;
Q[t,:]=f_t^T F(t-1:1); G[t,:]=g_write_t^T;
U[:,i]=F(C:i+1)u_i. With read vector r_t=q_t tensor e0,
Q_read[t,:]=r_t^T F(t:1), and J[t,i]=r_t^T F(t:i+1)u_i for i<=t.
The outputs are Q_read S_in+J Z. F is stored as [R,D,D], Q and Q_read as
[C,R*D], U as [R*D,C], K and J as [C,C], and G and Z as [C,V].

After the banded transport work, the existing recursive triangular inverse
uses O(C^3 log C) arithmetic, and multiplying its result into [Q,G] costs
O(C^2(RD+V)). Prepared tensors need O(CRD+C^2+RD^2+CV) storage per chunk.
Backward reconstructs preparation and computes local VJPs in bounded batches;
a local differentiated transport can retain O(C^2RD+CRD^2) intermediates per
processed chunk. This is not a constant-memory training claim. Native batch
sharding keeps the complete sequence and each head's four frames on one chip;
the model and expert sharding degrees are one.

Numerical checks use the same native unit q/k normalization, finite random
values, nonzero initial states and document resets both inside and between
chunks. Compare outputs, final states and every input/initial-state adjoint,
including input sequences split into streaming segments. Full-model checks
require finite BF16-boundary loss and gradients and a nonzero norm for each
layer's action-weight tensor; no BF16 baseline-output equality is claimed.
TPU timing uses five post-compilation repetitions, waiting for all outputs and
gradients each time. Final validation occurs after updates 1000,2000,...,10000
and 10676. NLL is summed masked token loss divided by token count; macro NLL
averages domain NLLs uniformly, over all 23 domains or within Paloma/Uncheatable.
