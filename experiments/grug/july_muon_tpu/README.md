# July MoE MuonW/MuonH sweep

This variant compares original-scaling Muon with AdamW fallback groups against
July MuonH with its original AdamH head and Adam fallback groups. Both arms use
the frozen July d512 model: six layers, 256 experts, top-4 routing, 8192-token
sequences, batch 16, seed 0, and 10,980 updates (1,439,170,560 tokens).

`source_manifest.json` identifies the historical recipe and source hashes.
`model.py`, `train.py`, `reference_optimizer.py`, `heuristic.py`, and `adamh.py`
were copied from that recipe, with imports, registration names and the dropped
assignment API adapted to this checkout. The model architecture is unchanged.
The shared library runtime differs from the historical baseline, so both arms
must be measured again here. `recipe.json` fixes the original regional caches,
mixture weights and shuffle configuration.

The matrix LR grid is 0.25, 0.5, 1, 2, and 4 times 0.009804100203726365.
The head peak LR stays 0.009804100203726365 and the other fallback peak LR stays
0.002262484662398392. July warmup and decay schedules are preserved.
MuonW uses `sqrt(max(1, fan_out/fan_in))` scaling and decoupled weight decay 0.1
on Muon matrices. Its AdamW groups decay matrices by 0.1 and exclude vectors
(norms and biases). MuonH retains the historical update rules.

Evaluation preserves logical batches of 256 and 2,048 examples per dataset,
executing model calls in batches of 16 as in the historical worker. Runs keep
online W&B, incremental local/GCS metrics, resolved configurations, source
hashes and checkpoints. A fresh process group gets a new W&B ID and artifact
prefix; process restarts do not claim checkpoint resumption. Completion is
recorded only after all updates and final evaluation return successfully.

```bash
uv run python -m experiments.grug.july_muon_tpu.run \
  --optimizer muonw --lr-mult 1 --run-name july-muonw-lr1 \
  --artifact-root gs://marin-us-central2/kaiyuew/july-muonw-muonh-20261029/muonw-lr1
```

Run this command inside a v4-32 Iris task with the `tpu` extra and authorized
Llama 3.1 tokenizer files at `tokenizer/`. Add `--logging-check` to test online
tracking without training, or `--dry-run` to print the resolved configuration.
