# Protected-frame KDA on Belay CPLM Plus

This experiment adds a content-addressed memory to each frame of a soft stack.
The starting model is native Marin Belay CPLM Plus, d512, six layers, seed 0,
batch 16 and sequence length 8192. The existing KDA/MLA/Inkling layers and Plus
copy head remain unchanged. A new residual reads the frame memories immediately
before the Plus head. Its zero-initialized gain preserves the initial head input.

## Token operator

Each of H heads stores D frames, each an R by V matrix. The frozen configuration
is H=4, D=8, R=4, V=64. Query, key, value, decay, write strength, four action
probabilities and depth address are functions of the current backbone hidden
state. Queries and keys are L2-normalized; values are RMS-normalized. Beta uses
sigmoid; R-vector decay uses exp(-softplus(g-4)). Action and depth probabilities
use separate per-head softmax distributions. Controls
never depend on the recurrent frame state, which makes the recurrence affine.

For top memory M, define U(M)=(I-beta k k^T) diag(decay) M + beta k v^T.
The four branch outcomes are:

- Push: [U(M0), M0, M1, ..., M(D-2)]. The child inherits the parent before its update.
- Pop: [M1, M2, ..., 0]. The restored parent receives no decay or write.
- Stay: [U(M0), M1, ..., M(D-1)].
- Replace: [beta k v^T, M1, ..., M(D-1)].

Softmax action probabilities form a convex mixture of these outcomes. A packed
document boundary zeros all incoming frames before the current write. Training
starts every sequence from zero and carries no state between batches. The read
uses the state after the current action mixture: the depth-weighted sum of q^T Md.
A projection returns the read to the
backbone width, and an elementwise learned gain scales the residual.

One-hot actions provide an exact parent snapshot until depth overflow. Learned
soft actions can mix or lose snapshots; native text provides no explicit scope
labels. The experiment tests whether the model can learn useful controls. It
does not establish exact symbolic stack behavior or generalization beyond D.

## Parallel forward and backward

Flatten depth and key rank to N=DR. One token is M'=L M+B, with sparse depth
shifts and a diagonal-plus-rank-one update in the top frame. For chunk length
C=64, process [I_N | 0] through C structured token operators independently in
every chunk, with an N by V zero block and writes applied only to that block.
This produces its affine summary (G,b). Compose only these T/C
summaries using (G2,b2) o (G1,b1)=(G2 G1,G2 b1+b2). Prefix summaries supply
the initial state for each chunk. Replay all chunks independently to emit reads.

The custom backward first propagates output cotangents through C transposed
structured operators with zero terminal cotangent. These local sources and
G^T define reverse affine summaries. A reverse prefix scan supplies each chunk's
terminal cotangent, including any loss on the final state. Local reverse-mode
replays then compute all input and initial-state gradients. This avoids
differentiating the augmented summary construction.

Parallel depth is O(C+log(T/C)). Per head, summary construction takes
O(T(N^2+NV+R(N+V))) work; boundary composition takes
O((T/C)(N^3+N^2V)); each local replay takes O(T(NV+RV)). Backward has the same
orders. Projections add O(T d H(3R+2V+D+5)) work. These are arithmetic bounds;
TPU efficiency also depends on fusion, padding and sequential work within C.

The recurrent state is 8192 FP32 values (32 KiB) per sequence. Stored boundary
matrices and incoming states occupy O((T/C)H(N^2+NV)); projected controls and
reads occupy O(TH(R+V+D)). Local autodiff may materialize O(THNV) replay
intermediates, so boundary storage is not a total training-memory bound. The
branch adds 313872 parameters at d512. Sequence shards are batch-local: the
memory recurrence adds no cross-device collectives. Replicated branch weights
still require ordinary data-parallel gradient reductions.

## Native experiment contract

The source parent is Marin 73279f54be6785b7a4a1b89acfca30d820bf352c. The exact
configuration is `train_belay.build_config`, based on `belay_recipe.json`, with
variant `plus-frame-kda`, width 512 and steps 10676. KDA runs in layers 0, 1, 2,
4 and MLA in layers 3, 5. The historical
Plus d512 comparison is run 4434036161f49ae783d3, seed 0, 10676 updates, with
baseline identity ecab1940b8165cecc1ba10267ab50b5fbf8de7f901e4eaddf511df0af42cde4f.
The candidate uses the same 10676-update schedule and frozen July data/tokenizer
manifests `july_data.json` and `tokenizer_identity.json` in this directory.
This is an equal-token comparison; the candidate
has additional arithmetic and must report its measured throughput and memory.
Both runs start from initialization; there is no restored pretrained checkpoint.
The historical run plan records Marin 0c079e8fbb1d80bf051e5d56a4d5e7a44b780aa7.
Historical source versions differ, so equivalence of inherited execution changes
must be checked before attributing a quality difference solely to this branch.

The user explicitly authorized native LM testing without the synthetic gates.
[Theory #59](https://github.com/WhenWen/astra-d8-research/issues/59) and
[task #58](https://github.com/WhenWen/astra-d8-research/issues/58) motivate parent
restoration; this trial does not evaluate that synthetic task or claim its gates
passed. The native entrypoint retains online W&B, raw incremental losses,
first-batch finite-gradient checks, checkpoints and validation every 1000 steps.
The new device preflight compares the chunked recurrence and adjoints to a
serial recurrence, then measures forward/backward at the native local shape.
That shape is B=1, T=8192, H=4, D=8, R=4, V=64 on one TPU v4 chip. The device
comparison uses T=256 with absolute tolerance 1e-5 and relative tolerance 1e-4
for its scalar objective and every input/state gradient; failure stops training.
Independent branch-enumeration tests additionally cover hard pop restoration,
packed-document isolation and streaming state continuation. Full-model decoding
and generation caches are outside this experiment.
