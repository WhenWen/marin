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
branch dimensions frozen in `frame_kda.FRAME_CONFIG` and recorded in W&B. Use
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

## Device qualification, 2026-10-21

The native experiment runs source `50baefeea7e5c5a1a47561aed80878a98077ca09`.
Eight remote CPU tests in [test_frame_kda.py](test_frame_kda.py) passed.
On TPU v4, the T=256 objective/input/state-adjoint
comparison had maximum absolute error 2.3842e-6. Five drained forward/backward
measurements at B1×T8192 gave median 44.851 ms (44.826–44.947 ms). XLA reported
810741760 temporary bytes, about 773 MiB, for this standalone memory kernel.
These timings include the benchmark objective and exclude projections, the
backbone, optimizer and cross-device synchronization. They are not whole-model
training overhead measurements.
The objective is the sum of sine applied to every read and final-state element.
Each timed sample waits for all returned values with `block_until_ready`.

The full native model passed its zero-update finite-gradient check. Initial
loss was 11.868136405944824, identical to the historical Plus first-batch loss.
The new gain gradient norm was 4.91277e-5. The resolved optimizer configurations
match exactly. The only backbone configuration difference is an explicitly false
`qb_defer_update` field, preserving the old immediate-update path. Data differs
only in the spelling of the local tokenizer path; the checked files match.
Source inspection found unchanged copy-head implementation, optimizer, recipe,
data manifest and tokenizer identity. The inherited QB refactor with
`qb_defer_update=False` calls the same immediate histogram update; accumulation
hooks are inactive at accumulation factor one. The inherited loss-scaling
argument defaults to one, as used here. Both checkpoint initialization paths
start at restored step zero, without restored weights. These checks cover the
identified inherited changes but are not a proof of identical full-run dynamics.
This supports comparison under the recorded source changes; it does not imply
bitwise equality of later mixed-precision training trajectories.

The full seed-0 run is
[tracked in W&B](https://wandb.ai/marin-community/auto-understanding-arch/runs/frame-kda-d512-s0-v1-85822ec3537e493db4e5bf2f91beb9a2).
Iris job `/kaiyuew@stanford.edu/frame-kda-d512-s0-20261021-v1` uses four hosts
and 16 physical TPU v4 chips, batch priority, the default preemption retries,
and no imposed experiment wall-clock limit. Regional evidence is under
`gs://marin-us-central2/kaiyuew/frame-local-kda-20261021/results/85822ec3537e493db4e5bf2f91beb9a2/`.
The historical comparison is
[Plus seed 0](https://wandb.ai/marin-community/auto-understanding-arch/runs/plus-d512-s0-v1-1e589eb42c014e2497196c73e553b6c0).
These qualification checks establish implementation compatibility. Training
progress and validation quality require the run's measured losses.

## Early training snapshot: steps 0–224

The native run has performed optimizer updates on TPU. At logged step 224,
training NLL is 4.61545 nats/token versus 4.65907 for historical Plus at the
same step. The unsmoothed mean over steps 125–224 is 5.00040 versus 5.06197.
This single-seed early training comparison is not validation-quality evidence.
The candidate has not reached its first scheduled validation at this snapshot.

Over the matched interval 100–224, median native training throughput is
364740 tokens/s versus 415415 tokens/s for Plus, a 12.2% throughput reduction.
This early window exceeds the 10% sustained-throughput reduction target in the
[dated pre-training prediction](https://github.com/WhenWen/astra-d8-research/issues/59#issuecomment-6311975558);
a later steady-state window is still needed to judge that target.
The field is `throughput/tokens_per_second`, based on the native step duration
from dispatch or previous step completion to the loss becoming host-readable.
It waits for the loss but does not add separate data-loading, logging,
checkpoint or evaluation durations. Pipelined host work may overlap device
execution, so this is not whole-run wall-clock throughput.
The hardware class and global token batch match; these historical runs
are not a simultaneous controlled timing experiment. No MoE assignments were
dropped in the candidate observations. The full 10676-step run remains active;
this snapshot neither ends its schedule nor establishes a quality improvement.

The [evidence bundle](https://console.cloud.google.com/storage/browser/marin-us-central2/kaiyuew/frame-local-kda-20261021/analysis/step-224)
contains the frozen run manifest, numerical checks, filtered candidate metrics,
progress summary, and unsmoothed `train-prefix` and `train-full` curves in PNG,
PDF and SVG. Each curve includes its exact plotted-data CSV and summary JSON.
`train-prefix` compares steps 0–224; `train-full` retains all available baseline
steps and the candidate's partial curve. The reproducible renderer is
[plot_frame_kda.py](plot_frame_kda.py). Raw native metrics remain in the regional
run directory above and in the historical run's artifact directory.

## First validation: 1000 updates, logged step 999

The first scheduled validation completed on all 23 evaluation domains. This
snapshot does not show a quality gain. Lower NLL is better:

| Metric (nats/token) | Historical Plus | Frame KDA + Plus | Difference |
| --- | ---: | ---: | ---: |
| Validation token-weighted NLL | 3.730108 | 3.734967 | +0.004859 |
| All-domain macro NLL | 3.957164 | 3.964257 | +0.007093 |
| Paloma macro NLL | 3.986214 | 3.992540 | +0.006326 |
| Uncheatable macro NLL | 3.890764 | 3.899609 | +0.008845 |

Eight of the 23 domain losses are lower; these correlated domain comparisons
are descriptive, not independent trials or a significance test. Training NLL
at this step is 3.734013 versus 3.733811. Over steps 100–999, the native
throughput medians are 364730 and 415390 tokens/s respectively, a 12.2%
reduction across 900 observations per run. No candidate MoE assignments were
dropped. The complete 10676-step experiment and its final comparison remain
pending; this early checkpoint does not settle the quality hypothesis.

The [step-999 evidence bundle](https://console.cloud.google.com/storage/browser/marin-us-central2/kaiyuew/frame-local-kda-20261021/analysis/step-999)
contains all domain losses, matched timing statistics, raw filtered metrics,
and training, Paloma and Uncheatable curves. Each curve has both a matched
prefix and a full available-range view, PNG/PDF/SVG exports and exact CSVs.
Sparse validation observations are shown as points, including this first one.
The plot source is `79002deb1`; the training source remains `50baefeea`.
