# July-baseline Identity Hyper-Connections

This Grug variant keeps the real July-baseline attention, MoE, optimizer, data,
and compute-optimal cells, but replaces each scalar residual stream with four
Identity Hyper-Connection streams.

## Identity-HC residual path

For each attention and MoE sublayer, the four streams are flattened and used to
compute token-dependent read and write coefficients:

```text
H_pre  = sigmoid(alpha_pre  * RMS(x) @ W_pre  + b_pre)
H_post = 2 * sigmoid(alpha_post * RMS(x) @ W_post + b_post)
z      = sum_i H_pre[i] * x[i]
x_next = x + H_post * F(z)
```

`H_res` is exactly the identity, so no Sinkhorn projection or residual-stream
mixing is present. The mapping projection uses Xavier-uniform initialization,
the two learned alpha values start at `0.01`, and the bias starts at zero. All
Identity-HC parameters use Adam rather than MuonH.

The model replicates the embedding stream four ways at transformer entry and
averages streams at transformer exit, matching the upstream Hyper-Connection
boundary contract. Projection and coefficient math accumulate in float32 while
the residual streams remain in the model compute dtype.

## Memory control

The widened residual state is rematerialized in two-layer groups. For the July
depths this is the rounded paper-optimal group size:

```text
L_r = floor(sqrt(n * L / (n + 2))) = 2
```

Grouping avoids saving the four-stream boundary at every layer. Gate 1 is
blocked until matched v5p-8 baseline/candidate profiles show no more than 8%
throughput loss and acceptable activation-memory behavior.

## Preserved July baseline

- 4:1 GQA with 128-dimensional heads.
- Half-RoPE on short layers; every fourth and final long layer disables RoPE.
- PKO disabled.
- 256 routed experts, four active experts per token, and one shared expert.
- May-recipe MuonH optimizer and the exact d512/d768 compute-optimal schedules.

The long-running experiment record is in
[`identity-hyperconnection-gate1.md`](../../../.agents/logbooks/identity-hyperconnection-gate1.md).
