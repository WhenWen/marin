# Cross-encoder-decoder (CED) MoE experiment

CED splits the transformer at its midpoint. The first half is a causal encoder
that produces fixed memory `z`. The second half is a causal cross-decoder: Q
consumes the evolving decoder residual, while every decoder layer's own K and V
projections consume `z`. This preserves every attention parameter.

For encoder `E` and decoder blocks `D_l`, ordinary CED is

```text
z = E(x)
p_0 = z
p_{l+1} = D_l(p_l; K_l(z), V_l(z)).
```

The pause-embedding variant follows *Pause Tokens Strictly Improve Language
Modeling* ([arXiv:2609.03807](https://arxiv.org/abs/2609.03807)) and initializes
every decoder token from one shared learned vector instead:

```text
z = E(x)
p_0[b, t, :] = e_pause
p_{l+1} = D_l(p_l; K_l(z), V_l(z)).
```

`e_pause` is initialized with the model's standard weight initializer and is
optimized as a one-dimensional Adam parameter. Relative to ordinary CED this
adds exactly `hidden_dim` parameters. It does not add a layer, attention call,
FFN, projected K/V tensor, or decoder pass, so its main-model FLOPs are unchanged.
Unlike ordinary CED, it removes the direct residual path from `z`; encoder
information reaches the decoder residual only through cross-attention.

The boundary is derived from model depth:

| model | layers | standard layers | shared K/V source | reuse layers |
| --- | ---: | --- | --- | --- |
| d512 | 6 | 0--2 | output of layer 2 | 3--5 |
| d768 | 8 | 0--3 | output of layer 3 | 4--7 |
| d1024 | 11 | 0--5 | output of layer 5 | 6--10 |
| d1280 | 13 | 0--6 | output of layer 6 | 7--12 |

Each decoder layer retains its own Q, K, V, O, RMSNorm, and GatedNorm parameters.
The cached source is passed through that target layer's attention RMSNorm and
GatedNorm before its K/V projections, matching where the current residual enters
those matrices in the July pre-norm architecture.

Run the d512 gate cell with:

```bash
uv run python -m experiments.grug.moe_yoco_kv_reuse.experiment \
  --run_only '["grug/moe_ced_july_d512"]'
```

Run the pause-embedding d512 gate cell with:

```bash
uv run python -m experiments.grug.moe_yoco_kv_reuse.experiment_pause \
  --run_only '["grug/moe_ced_pause_july_d512"]'
```

The d768, d1024, and d1280 cells are defined by the same depth-derived recipe.
For odd depths, the middle layer remains standard and its output becomes the
K/V source for the later `floor(num_layers / 2)` layers. The d1280 cell uses a
v5p-16; the smaller cells use v5p-8. These resources are pinned to us-central1,
where both requested TPU topology groups are configured.

The overtraining benchmark uses Marin's 750-token-per-active-parameter setting.
For exact-July d512 this is 15.55B tokens (batch 16, 118,620 steps). It defines
a fresh unchanged control and a CED arm under the same code snapshot:

```bash
uv run python -m experiments.grug.moe_yoco_kv_reuse.experiment_overtrain
```
