# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real LM updates: uninterrupted versus durable checkpoint continuation."""

import argparse
import dataclasses
import json
import os
import uuid
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.data.loader import DataLoader
from levanter.layers.attention import AttentionMask
from levanter.trainer_state import TrainerState
from rigging.secrets import require_secret

from experiments.hybrid_transfer.model import Residual, TransferConfig
from experiments.hybrid_transfer.optimizer import TransferOptimizer
from experiments.hybrid_transfer.recovery import write_json
from experiments.hybrid_transfer.storage import configure_storage
from experiments.hybrid_transfer.train import GROUP


def loss(model, batch, key):
    # Random input masking makes the equality check sensitive to restored RNG.
    masked = jnp.where(jax.random.bernoulli(key, 0.1, batch.shape), 0, batch)
    tokens = hax.named(masked, (hax.Axis("batch", batch.shape[0]), hax.Axis("position", batch.shape[1])))
    logits = model(tokens, AttentionMask.causal()).rearrange(("batch", "position", "vocab")).array
    labels = jnp.roll(batch, -1, axis=1)
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1]), labels[:, :-1, None], axis=-1))


@eqx.filter_jit
def train_step(state, batch):
    key, next_key = jax.random.split(state.training_key)
    value, grad = eqx.filter_value_and_grad(loss)(state.model, batch, key)
    return state.take_step(grad, key=next_key)[0], value


def check(method, optimizer_name, root):
    config = TransferConfig(
        hidden_dim=32,
        intermediate_dim=64,
        num_heads=1,
        num_kv_heads=1,
        num_layers=2,
        base_width=32,
        base_depth=1,
        max_seq_len=8,
        residual=method,
        attn_backend="vanilla",
    )
    model = config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(11))
    optimizer = TransferOptimizer(
        variant=optimizer_name,
        learning_rate=0.02,
        adam_lr=0.001,
        depth_alpha=config.depth_alpha,
        depth_multiplier=config.depth_multiplier,
        hidden_lr_multiplier=config.hidden_lr_multiplier,
        warmup=2,
        weight_decay=0.1 if optimizer_name == "muonw" else 0,
    ).build(6)
    initial = TrainerState.init(optimizer, model, key=jax.random.PRNGKey(19))
    dataset = ListAsyncDataset([np.random.default_rng(i).integers(0, 32, size=8, dtype=np.int32) for i in range(12)])
    dataset = dataset.shuffle(jax.random.PRNGKey(42), perm_type="linear")
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("replica", "data", "model"))
    evidence = {"method": str(method), "optimizer": optimizer_name, "runs": []}
    checkpoint = f"{root}/{method}-{optimizer_name}-{uuid.uuid4().hex[:8]}/step-2"
    result = None
    with jax.set_mesh(mesh), hax.axis_mapping({"batch": "data"}):
        loader = DataLoader(dataset, 2, mesh=mesh, max_buffered_batches=0)
        all_batches = list(loader)
        for phase in ("uninterrupted", "interrupted", "restored"):
            run = wandb.init(
                entity="marin-community",
                project="Hyperball",
                group=GROUP,
                name=f"resume-check-{method}-{optimizer_name}-{phase}",
                mode="online",
                reinit="finish_previous",
                config={
                    "validation_only": True,
                    "phase": phase,
                    "seed": 11,
                    "model": dataclasses.asdict(config),
                    "optimizer": optimizer_name,
                    "checkpoint": checkpoint if phase == "restored" else None,
                },
            )
            evidence["runs"].append(run.url)
            if phase == "restored":
                state = load_checkpoint(initial, checkpoint, allow_partial=False)
                assert int(state.step) == 3
                batches = list(loader.iter_from_step(int(state.step)))
                for actual, expected in zip(batches, all_batches[3:], strict=True):
                    np.testing.assert_array_equal(actual, expected)
            else:
                state = initial
                batches = all_batches[:3] if phase == "interrupted" else all_batches
            # Match Trainer's explicit device placement after checkpoint loading.
            state = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, PartitionSpec())), state)
            for batch in batches:
                state, value = train_step(state, batch)
                metric = float(value)
                assert np.isfinite(metric)
                val = float(loss(state.model, all_batches[0], jax.random.PRNGKey(123)))
                metrics = {"train/loss": metric, "validation/loss": val, "next_step": int(state.step)}
                run.log(metrics)
                with Path(run.dir, "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
            if phase == "uninterrupted":
                result = state
            elif phase == "interrupted":
                save_checkpoint(state.saveable_state, step=2, checkpoint_path=checkpoint)
            else:
                max_difference = 0.0
                for actual, expected in zip(jax.tree.leaves(state), jax.tree.leaves(result), strict=True):
                    np.testing.assert_array_equal(actual, expected)
                    max_difference = max(
                        max_difference, float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
                    )
                run.summary["resume/max_absolute_difference"] = max_difference
                evidence["max_absolute_difference"] = max_difference
                evidence["status"] = "passed_exact_full_state_equality"
            run.finish()
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method", choices=[str(r) for r in Residual])
    args = parser.parse_args()
    os.environ["WANDB_API_KEY"] = require_secret("WANDB_API_KEY")
    configure_storage()
    results = []
    for method in ([Residual(args.method)] if args.method else Residual):
        for optimizer in ("muonw", "muonh"):
            results.append(check(method, optimizer, args.output_root))
            Path("resume-validation.json").write_text(json.dumps(results, indent=2))
            write_json(f"{args.output_root}/results.json", {"results": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
