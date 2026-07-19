# July baseline half-RoPE plus learned relative attention

This variant starts from upstream commit
`52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d` on `marin/july_baseline`.
It preserves that baseline's GQA, PKO-disabled policy, and half-RoPE behavior:
short layers rotate the first half of each Q/K head, while the designated long
layers skip RoPE when `disable_long_rope=True`.

The attention score adds an Inkling-style learned relative term to the existing
RoPE content score. For query position `i` and causal key position `j`:

```text
score(i, j) = <RoPE(q_i), RoPE(k_j)> / D + (x_i W_r) E[:, i - j]
```

The Gate 1 recipe uses head dimension `D=128`, relative rank `R=16`, relative
extent `E=1024`, direct distance order, learned Q/K RMSNorm gains initialized
to one, and independent untruncated `N(0, 0.02)` initialization for `W_r`
and the relative table. The learned gains, `W_r`, and relative table use Adam;
the July baseline's remaining MuonH/AdamH/Adam groups are unchanged.

Launch both fresh Gate 1 cells from this worktree with:

```bash
uv run python -m experiments.grug.moe_rope_relative_position.launch_gate1
```

The launcher contains only the d512 and d768 cells and records them in W&B group
`MOE-JULY-ROPE-RPE-INKP3-gate1-issue-7208`.
