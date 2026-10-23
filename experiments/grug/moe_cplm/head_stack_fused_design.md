# Fused tree preparation for the four-head stack

2026-10-23 algorithm freeze, theory [#59](https://github.com/WhenWen/astra-d8-research/issues/59).
This is an exact-arithmetic rearrangement of the three-action recurrence in
[the architecture specification](head_stack_kda_design.md), not a new model.
Belay's four existing KDA states remain the four stack slots. Its projections,
copy head, soft action controller and native operator interface remain fixed.
No training or chunk-size search is part of this implementation step.

## Interval representation

Let D=4, R=128 key channels, V=128 value channels, N=DR=512, and C=16 tokens
per chunk. Flatten slot and key axes in the state S, giving N by V.
A token has bare transport E: one D-by-D matrix per key channel. Its diagonal
is (push+stay)*exp(g_0) for slot zero and stay*exp(g_h) for other slots; its
lower and upper diagonals are push and pop. Multiply E by document keep.
Let a_h=stay+push*1[h=0]. Define injection U[:,h]=e_h tensor k_h,
erase Q[h,:]=e_h tensor (a_h*beta_h*exp(g_h)*keep*k_h), and
G[h,:]=a_h*beta_h*v_h. Then S'=E S+U Z, Z=G-Q S.
Reading gives y=R_read S+A Z, where R_read is the block query times E and
A is diagonal with entries q_h dot k_h. No inverse of E is needed.

An interval of m tokens stores F (D,D,R), W (N,Dm), Q and R_read (Dm,N),
K and A (Dm,Dm), and G (Dm,V). These are defined by

    Z = G - Q S_in - K Z
    S_out = F S_in + W Z
    Y = R_read S_in + A Z.

K is strictly lower in token blocks. The same-token D-by-D block is zero.
For adjacent intervals L then R, construct their parent's arrays by

    F = F_R F_L
    W = [F_R W_L, W_R]
    Q = [Q_L; Q_R F_L]
    R_read = [R_L; R_R F_L]
    K = [[K_L, 0], [Q_R W_L, K_R]]
    A = [[A_L, 0], [R_R W_L, A_R]]
    G = [G_L; G_R].

F products act independently on each key channel. The identities follow by
substituting the left output into the right input, before solving for Z.
They also hold for singular E, hard push/pop, and document resets. They avoid
the unstable inverse-prefix trick and the full N-by-N transition matrix.

Merge in a balanced binary tree, then compute T=(I+K)^-1,
Q_hat=T Q and G_hat=T G. The state pass consumes
F,W,Q_hat,G_hat,R_read,A in the same form as the existing chunk implementation.
The tree has log2(C) dependent levels. The interval state pass remains ordered
across T_length/C chunks; this is not a logarithmic-depth full-sequence scan.

Elimination follows the same tree, starting with four-by-four identity
blocks. Each merge forms T_parent=[[T_L,0],[-mm(mm(T_R,K_RL),T_L),T_R]].
Only compact child blocks enter these products; do not multiply masked full
L-by-L arrays at every level, which would add a log(C) work factor.
T remains FP32; every mm casts both operands and the result to BF16 in mixed
mode and uses Precision.HIGHEST FP32 otherwise. Q_hat and G_hat use one
concatenated mm. The backward differentiates this exact sequence of casts.
K is a mathematical description: the kernel consumes each cross block
immediately when building T and does not materialize the complete K.

## Fusion and backward

One preparation program owns one sequence/chunk. It loads q,k,v,g,beta,
actions and keep, builds the tree and performs the triangular elimination in
VMEM, and writes only the final prepared arrays. Parallel chunks have no
dependencies. On TPU, this independence permits batching and core partitioning;
it does not imply that every Pallas grid element executes simultaneously.

The initial kernel uses static tree merges with dense MXU cross products.
State transport is four-by-four per key and uses vector operations.
Keep transport, gates and state in FP32. The mixed path rounds intra-chunk
GEMM inputs and outputs to BF16, accumulating products in FP32; FP32 remains
the strict numerical reference. Different operation order means mixed values
need not equal the old mixed implementation bitwise.

The preparation custom VJP reloads token inputs, rebuilds each small tree
on chip and differentiates the executed fixed-size array computation inside
the Pallas program. It writes only input gradients, never a sequence-wide
tree tape. This includes the dtype-conversion adjoints used by JAX. It does
not substitute the exact inverse derivative for a rounded inverse algorithm.
Compiler support and VMEM allocation for this schedule require validation.

In the reverse preparation, retain the fixed tree's transport/row arrays
and rounded dot operands from recomputation, reverse elimination, then reverse
merges from root to leaves. For C16, all levels' three L-by-N row arrays
total at most 1.875 MiB; transports under 0.25 MiB; K/A under 0.063 MiB;
G under 0.157 MiB. Budget another copy for adjoints, 0.5 MiB for elimination
values/adjoints and 2 MiB for GEMM and compiler temporaries: an 8 MiB target.
This is a conservative planned budget, not a measured compiler allocation.
Compilation must confirm VMEM fit; do not claim the absence of spills yet.

The state kernel should extend the native TPU Pallas stream: retain S in
VMEM, compute Z=G_hat-Q_hat S, emit Y=R_read S+A Z and update F S+W Z.
Save only incoming chunk states and innovations. Its reverse stream uses

    dZ = A^T dY + W^T dS_out
    dS_in = F^T dS_out + R_read^T dY - Q_hat^T dZ.

Prepared-input gradients are outer products with S and Z; dF reduces over
value channels for each source/destination slot and key. The reverse stream
also retains its state adjoint in VMEM. Preparation gradients are independent
across chunks after this reverse stream. No token-state replay or automatic
differentiation through a sequence-length preparation scan is required.

## Costs and acceptance

For L=DC innovations per chunk, exact tree cross products cost
O(L^2 N), transport transformations O(L D N log C), and the final dense
elimination O(L^3+L^2(N+V)). These are arithmetic counts, not runtime claims.
Static small merges can underutilize TPU MXUs; packing independent merges is
an implementation option after measuring the first fused kernel.
Forward plus backward use the same orders of work, with recomputation and
matrix-product adjoints adding constant factors. The tree replaces the old
O(C L D R) sequential transport work with O(L D N log C) work and fewer
dependent levels; this benefit depends on C versus D log C and is not uniform.

Final preparation traffic is O(LN+L^2+LV+D^2R) per chunk; tree intermediates
stay on chip. A conservative retained full tree uses O(log C) times that
storage on chip. State is still exactly 256 KiB per sequence/layer; saved
chunk states and preparation arrays are separate training memory. No network
communication is added when all four heads remain on the same device.

State-pass GEMMs cost 6LNV+2L^2V forward and 12LNV+4L^2V backward FLOPs
per chunk (multiply-add counts as two), plus O(D^2RV) transport/reductions.
Tree cross products cost 2N(L^2-C D^2); compact inversion costs less than
(2/3)L^3, and the final solved-row product costs 2L^2(N+V). These counts
exclude vector operations, input/output projections and hardware padding.
Recomputing preparation for its VJP adds one preparation forward plus its
reverse products. TPU padding and small-dot launch overhead can dominate
these arithmetic counts and remain an implementation risk.

Cross-head mixing still makes a dense D*C correction system instead of D
independent C systems. Its elimination has D^2 more leading cubic work and
state GEMMs D times more leading work than independent native heads. Fusion
can remove HBM traffic and poor scheduling, not this algebraic cost. There is
no claim of native speed, lower-bound optimality, or proven LM benefit.

First verify tree preparation, full outputs and all differentiable inputs
against the literal FP32 recurrence at atol=1e-5, rtol=1e-4, including resets,
nonzero state and hard actions. Check the fused VJP against differentiation
of the same tree and precision path (the same strict tolerance for FP32;
report mixed relative L2/max error separately and require finite results).
Mixed-versus-literal errors are reported, not treated as strict equivalence.
Then measure the fixed C16 fused forward
and backward on the native B1,T8192 shape, preserving raw timing and memory.
Any unsupported lowering or spill is a failed implementation check, not
evidence that the mathematical algorithm passed hardware qualification.
