# Four native KDA heads as one stack

## Current amendment: three actions and matched precision paths

Interface amendment on 2026-10-23: expose chunk_kda(q,k,v,g,beta,*,actions,...)
with native head-major layout, q/k normalization and scaling, log-decay input,
initial/final state, segment ids and matmul_dtype selection. The only required
additional input is one shared three-action vector per token. Both benchmark
operators receive identical g directly. The internal preparation exponentiates
g once; remove the decay-to-log roundtrip from the comparison. The earlier
roundtrip on TPU v4 changed decay by up to 7.82e-5 and caused the two failed
hard-stay checks. Matching effective decay restored strict equivalence in v3
(output max error 1.49e-8, final state 1.49e-7). No tolerance was changed.

2026-10-23, before the revised TPU submission: remove replace following user
feedback. The current actions are push, pop and stay, with softmax bias [0,0,6].
Their definitions below are unchanged; the replace branch is absent. Action
parameters now total 512*3+3=1,539 per layer, or 6,156 across four KDA layers.
State remains 256 KiB per sequence/layer. Earlier four-action measurements
belong to their immutable source and do not evaluate this revision.

Add a mixed-precision path alongside the strict FP32 path. Keep transport,
state, gates and reductions in FP32; use BF16 operands/results for triangular
inversion products, transformed erase/value products and intra-chunk readout.
Cross-chunk state GEMMs use FP32 with Precision.HIGH, matching native TPU KDA.
The strict reference retains Precision.HIGHEST throughout. Both new paths use
automatic reverse differentiation of their actual chunk graph, including all
casts, with rematerialized preparation. This replaces the previous bounded
sequential-replay VJP and may change training intermediate memory; report it.
It avoids using the exact recurrence's adjoints for a rounded BF16 forward.
Gradients follow JAX's dtype-conversion differentiation rules.

Benchmark both precisions at C16 and C32 with the same backward algorithm.
Retain strict FP32-versus-recurrence tolerances. Separately report mixed-versus-
FP32 output/state and every adjoint's maximum absolute and relative L2 errors;
do not reinterpret approximation error as passing the strict equivalence test.
Mixed precision is a numerical approximation, not an established LM quality
equivalence. The model wrapper defaults to mixed precision and has an explicit
FP32 option for the native-limit check. The remainder preserves the original
four-action protocol as historical context.

Protocol frozen on 2026-10-23 before implementation and TPU measurement.
Theory: https://github.com/WhenWen/astra-d8-research/issues/59.
Motivating task: scoped_binding@1 (#58), not evaluated here. This request is
limited to implementation, numerical validation and kernel timing; zero
optimizer updates. The user previously waived synthetic-task gates.

## State and recurrence

The original Belay CPLM Plus d512 KDA has four FP32 states of shape 128 by
128. Treat these existing states as slots S0,...,S3 of one stack. Both native
and candidate therefore retain 65,536 floats, or 256 KiB per sequence per KDA
layer. This supersedes the expanded-state candidate in stack_kda_design.md,
which allocated four stacks and remains a historical implementation.

For each slot h, let U_h(S)=D_h S + beta_h k_h (v_h-k_h^T D_h S)^T,
where D_h=diag(exp(g_h)). Reset all slots at a document boundary before
applying this recurrence. Four shared action probabilities (push, pop, stay,
replace) produce the convex mixture of:

- push: [U_0(S0), S0, S1, S2]; discard the old bottom.
- pop: [S1, S2, S3, 0].
- stay: [U_0(S0), U_1(S1), U_2(S2), U_3(S3)].
- replace: [beta_0 k_0 v_0^T, S1, S2, S3].

All four queries read their corresponding updated slots: y_h=q_h^T S'_h.
Thus hard stay reproduces the four native KDA recurrences, and all native
q/k/v, decay and beta projections remain active. Hard push followed by hard
pop recovers the previous top three slots; the discarded bottom cannot be
recovered. Lower slots may change during stay, so this is not an immutable
parent stack during arbitrary intervening operations. Soft mixtures do not
guarantee symbolic stack recovery.

One per-layer action projection has shape [512,4] plus four biases, adding
2,052 parameters per KDA layer and 8,208 across layers 0,1,2,4. Use zero
weights and bias [0,0,6,0] for near-stay initialization. Retain native query/key
normalization, query scale, convolution, output norm/gate/projection, MLA,
MoE, AttnRes and CPLM Plus prediction head. Near-stay is not exact baseline
initialization. No new LM training is included in this benchmark.
Apply softmax to x W+b, where x is the same normalized layer input used by
the native KDA projections. The operator accepts probabilities; the model
wrapper differentiates through softmax.

## Parallel implementation

Flatten slot and key axes to N=4R=512. Each token transition is a key-wise
4-by-4 tridiagonal transport E minus four erase outer products. The diagonal
of E is (push+stay)*decay_0 on the top and stay*decay_h+replace below;
its lower/upper diagonals are push/pop. Document keep multiplies E and erase.
Each erase row touches only its own slot. Injection columns are e_h tensor k_h.
The top write coefficient is beta_0*(push+stay+replace); lower coefficients
are beta_h*stay.

For C tokens, prepare the transport F, transported injection columns W,
incoming-state erase rows Q, read rows R and strictly block-lower correction
K. There are L=4C innovations. Solve (I+K)Z=G-Q S_in. Propagate a boundary by
S_out=F S_in+W Z; read all slots using R S_in+A Z. The four innovations within
one token have no mutual dependency. Prepare chunks in parallel, scan chunk
boundaries, then read chunks in parallel. Backward scans boundary adjoints and
replays bounded batches of C-token chunks for local input gradients.

Persistent recurrent state is unchanged. Prepared erase/write/read tensors
cost O(T H D R), boundary states O((T/C) D R V), with H=D=4. Compared with
native diagonal transport, depth mixing costs extra vector operations and
the triangular system grows from C per independent head to 4C per stack.
Boundary multiplication costs O((T/C) D^2 R V + T H D R V). These are
implementation costs, not a predicted speedup. The complete four-slot stack
must reside on one device; the frozen model-axis size is one, so no extra
communication is needed in this configuration. Head-sharded execution is
unsupported pending a separate communication design.

Here T is sequence length, C is tokens per chunk, H=D=4 counts heads/slots,
R is key width, and V is value width. In the algorithm above, call the read
matrix R_read to distinguish it from key width. At a chunk start F=I and
W=0. At token t, erase row (t,h) is f_h^T F_before and correction row
(t,h) is f_h^T W_before. Then update F=E_t F and W=E_t W, injecting
e_h tensor k_h into column (t,h). Read row (t,h) is
(e_h tensor q_h)^T F, with attention row (t,h) given by the same read of W.
The value row G_(t,h) is the write coefficient times v_h. Thus top erase
uses beta_0*(push+stay) while its value row includes replace as well.
F is represented by R matrices of shape D by D; W has shape N by L;
Q and R_read have shape L by N; G has shape L by V; K and A have shape L by L.
Q and G in the state pass mean their triangular-solved values.

The boundary scan is ordered, not an associative prefix scan. Its affine map
is S_out=(F-W (I+K)^-1 Q_raw)S_in+W (I+K)^-1 G_raw.
K and A storage is O((T/C)L^2); their preparation requires O(T H^2 C R)
reductions. Dense triangular inversion and solves add O((T/C)(L^3+L^2(N+V)))
arithmetic. This first implementation prioritizes exact numerical equivalence;
it makes no promise of native-kernel speed at equal state size.

## Validation and timing

Check against a literal sequential reference, all input and initial-state
adjoints, nonzero initial states, document resets, hard push/pop, hard stay
versus native KDA and prefix/chunk continuation. Retain atol=1e-5, rtol=1e-4.
Use native-size R=V=128 on TPU v4 for numerical qualification at T=128, then
timing at B=1,T=8192. Compare identical normalized q/k/v/decay/beta/reset and
initial state with native KDA's existing Pallas state pass, mixed precision.
The candidate uses highest precision initially; disclose this difference.

Report independently compiled forward, backward with saved forward residuals,
and combined forward/backward; synchronize every repetition and separate
compilation. Record raw repetitions, temporary bytes, exact sources and device.
Benchmark C=16 and C=32 with bounded local backward batches 16 and 64. Count
state bytes directly. Candidate-vs-native equality is expected only for hard
stay; general soft actions are checked against the independent reference.
Operator timing does not establish whole-model throughput or LM quality.

The baseline source is the native kda.py, kda_state_tpu.py and train_belay.py
at Marin commit 0e5182f48. Native prepared state is FP32; its intra-chunk
products use BF16 as in the inherited implementation. Candidate arrays are
FP32 and dot precision is jax.lax.Precision.HIGHEST. Persist exact source
SHA256 hashes with each run. Inputs use NumPy seed 261023+T: Gaussian q/k
normalized with epsilon 1e-6 (q additionally divided by sqrt(R)), standard
Gaussian values, exp(-0.1*softplus(N(0,1))) decay, sigmoid(N(0,1)) beta and
0.1*N(0,1) initial states. Keep is zero at T//2 and one elsewhere. The timing
actions are softmax(0.2*N(0,1)+[0,0,6,0]); correctness also tests unbiased
softmax actions and hard branches. Tolerances apply separately to every
output, final state, objective and gradient leaf. Keep is binary in execution;
its continuous relaxation is also differentiated for the numerical audit.
