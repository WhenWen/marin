# Midpoint K/V reuse MoE experiment

This exact-July variant keeps every attention parameter but changes the input
to the K and V projections in the second half of the transformer. Q continues
to consume the current layer activation. K and V consume one cached activation:
the output of the last layer in the first half.

The boundary is derived from model depth:

| model | layers | standard layers | shared K/V source | reuse layers |
| --- | ---: | --- | --- | --- |
| d512 | 6 | 0--2 | output of layer 2 | 3--5 |
| d768 | 8 | 0--3 | output of layer 3 | 4--7 |
| d1024 | 11 | 0--5 | output of layer 5 | 6--10 |

Each reuse layer retains its own Q, K, V, O, RMSNorm, and GatedNorm parameters.
The cached source is passed through that target layer's attention RMSNorm and
GatedNorm before its K/V projections, matching where the current residual enters
those matrices in the July pre-norm architecture.

Run the d512 gate cell with:

```bash
uv run python -m experiments.grug.moe_yoco_kv_reuse.experiment \
  --run_only '["grug/moe_yoco_kv_reuse_july_d512"]'
```

The d768 and d1024 cells are defined by the same depth-derived recipe. For odd
depths, the middle layer remains standard and its output becomes the K/V source
for the later `floor(num_layers / 2)` layers.
