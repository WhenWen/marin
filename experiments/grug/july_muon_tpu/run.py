# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen July MoE: matched MuonW and MuonH matrix LR sweeps."""

import argparse
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import wandb
from fray.cluster import ResourceConfig
from jax.experimental.multihost_utils import broadcast_one_to_all
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import BlockShuffleConfig, DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.distributed import DistributedConfig
from levanter.models.lm_model import LmExample
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.july_muon_tpu import train
from experiments.grug.july_muon_tpu.heuristic import MoeHeuristic
from experiments.grug.july_muon_tpu.model import GrugModelConfig
from experiments.grug.july_muon_tpu.optimizer import JulyMuonHConfig, JulyMuonWConfig

HERE = Path(__file__).parent
SOURCE_COMMIT = "444b5ea66bda75b6dfc7746ff20c9dfec6ba159a"
SEQUENCE_LENGTH = 8192
STEPS = 10980


def build_config(args, run_id, checkpoint_root):
    values = json.loads((HERE / "recipe.json").read_text())
    model = GrugModelConfig(**values["model"])
    base = MoeHeuristic().build_optimizer_config(16, STEPS * 16 * SEQUENCE_LENGTH, 512, seq_len=8192)
    optimizer_args = asdict(base)
    optimizer_args["learning_rate"] *= args.lr_mult
    optimizer_class = JulyMuonWConfig if args.optimizer == "muonw" else JulyMuonHConfig
    optimizer = optimizer_class(**optimizer_args, head_lr=base.learning_rate)
    components = {
        name: DatasetComponent(
            cache_dir=value["cache_dir"],
            tags=value.get("tags"),
            source=UrlDatasetSourceConfig(),
            format=TextLmDatasetFormat(text_key=value.get("text_key", "text")),
        )
        for name, value in values["data_components"].items()
    }
    data = LmDataConfig(
        tokenizer=args.tokenizer,
        components=components,
        train_weights={name: values["train_weights"].get(name, 0.0) for name in components},
        auto_build_caches=False,
        cache_dir=None,
        shuffle=BlockShuffleConfig(**values["shuffle"]),
        permutation_type="feistel",
        block_cross_document_attention=True,
        mixture_block_size=values["mixture_block_size"],
    )
    tracker = WandbConfig(
        entity=args.entity,
        project=args.project,
        name=run_id,
        id=run_id,
        group="july-muonw-muonh-20261029",
        tags=[args.optimizer, "july-moe-d512", "tpu"],
        mode="online",
        resume="never",
        background=False,
        save_code=False,
    )
    trainer = TrainerConfig(
        id=run_id,
        seed=args.seed,
        train_batch_size=args.batch_size,
        num_train_steps=args.steps,
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=(JsonLoggerConfig(), tracker),
        log_dir=args.output / "logs",
        log_jaxprs=False,
        log_xla_hlo=False,
        progress_watchdog=ProgressWatchdogConfig(),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        load_checkpoint=False,
        checkpointer=CheckpointerConfig(
            base_path=checkpoint_root,
            append_run_id_to_base_path=False,
            temporary_base_path=checkpoint_root + "/temporary",
            keep=None,
            save_interval=timedelta(minutes=10),
        ),
        # Mosaic grouped matmul expects native bf16 products for bf16 inputs.
        jax_config={"jax_enable_pgle": False, "jax_default_matmul_precision": "default"},
    )
    return train.GrugRunConfig(
        model=model,
        optimizer=optimizer,
        data=data,
        resources=ResourceConfig.with_tpu("v4-32"),
        trainer=train.GrugTrainerConfig(
            trainer=trainer,
            expert_axis_size=1,
            z_loss_weight=0.0,
        ),
        eval=train.GrugEvalConfig(
            eval_batch_size=256,
            steps_per_eval=1000,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
            compute_bpb=True,
        ),
    )


class MetricsFileHandler(logging.FileHandler):
    def __init__(self, path, destination):
        super().__init__(path)
        self.destination = StoragePath(destination)
        self.last_upload = 0.0

    def upload(self):
        self.flush()
        self.destination.write_bytes(Path(self.baseFilename).read_bytes())
        self.last_upload = time.monotonic()

    def emit(self, record):
        super().emit(record)
        if time.monotonic() - self.last_upload >= 60:
            self.upload()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("muonh", "muonw"), required=True)
    parser.add_argument("--lr-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS, choices=(STEPS,))
    parser.add_argument("--batch-size", type=int, default=16, choices=(16,))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--entity", default="marin-community")
    parser.add_argument("--project", default="marin")
    parser.add_argument("--output", type=Path, default=Path("/tmp/july-muon-results"))
    parser.add_argument("--logging-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    identities = json.loads((HERE / "tokenizer_identity.json").read_text())
    for name, expected in identities.items():
        if hashlib.sha256((Path(args.tokenizer) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Tokenizer identity mismatch: {name}")
    if args.logging_check:
        tracker = WandbConfig(
            entity=args.entity,
            project=args.project,
            name=args.run_name + "-logging-check",
            id="logcheck-" + uuid.uuid4().hex,
            mode="online",
            resume="never",
            background=False,
            save_code=False,
        ).init(None)
        tracker.log_hyperparameters({"training_updates": 0, "seed": args.seed, "source_commit": SOURCE_COMMIT})
        tracker.log({"preflight/ok": 1}, step=0)
        tracker.finish()
        return
    if args.dry_run:
        config = build_config(args, args.run_name, args.artifact_root + "/checkpoints")
        print(json.dumps(asdict(config), default=str, indent=2))
        return
    DistributedConfig().initialize()
    identity = broadcast_one_to_all(np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8).copy())
    attempt = bytes(identity).hex()
    run_id = args.run_name + "-" + attempt[:12]
    root = args.artifact_root.rstrip("/") + "/" + attempt
    args.output = args.output / attempt
    args.output.mkdir(parents=True, exist_ok=True)
    config = build_config(args, run_id, root + "/checkpoints")

    def persist(name, value):
        if jax.process_index() == 0:
            content = json.dumps(value, default=str, indent=2).encode()
            (args.output / name).write_bytes(content)
            StoragePath(root + "/" + name).write_bytes(content)

    persist("resolved-config.json", asdict(config))
    persist(
        "identity.json",
        {
            "run_id": run_id,
            "execution": "july-native",
            "execution_sources": {
                name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                for name in ("run.py", "train.py", "model.py", "optimizer.py", "reference_optimizer.py", "adamh.py")
            },
            "source_commit": SOURCE_COMMIT,
            "data_sha256": hashlib.sha256((HERE / "recipe.json").read_bytes()).hexdigest(),
            "seed": args.seed,
            "steps": args.steps,
            "tokens": args.steps * args.batch_size * SEQUENCE_LENGTH,
            "restart_semantics": "fresh initialization; separate W&B curve and checkpoints per process group",
            "wandb_url": f"https://wandb.ai/{args.entity}/{args.project}/runs/{run_id}",
        },
    )
    handler = None
    if jax.process_index() == 0:
        handler = MetricsFileHandler(args.output / "metrics.jsonl", root + "/metrics.jsonl")
        logging.getLogger("levanter.json_logger").addHandler(handler)
    native_factory = train._make_train_step
    evaluator = train.TaggedEvaluator

    def bounded_evaluator(**kwargs):
        assert kwargs["max_examples_per_dataset"] == 2048 and kwargs["EvalBatch"].size == 256
        loss_fn = kwargs["loss_fn"]

        def evaluate(model, batch):
            if isinstance(batch, LmExample):
                batch = grug_lm_example_from_named(batch)

            def body(index):
                micro = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, index * 16, 16, axis=0), batch)
                return loss_fn(model, micro)

            values = jax.lax.map(body, jnp.arange(batch.tokens.shape[0] // 16))
            spec = jax.sharding.PartitionSpec(("replica_dcn", "data", "expert"), None)
            return jax.tree.map(
                lambda x: jax.lax.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]), out_sharding=spec), values
            )

        kwargs["loss_fn"] = evaluate
        return evaluator(**kwargs)

    def checked_factory(*params, **kwargs):
        native = native_factory(*params, **kwargs)
        first = True

        @jax.jit
        def preflight(state, batch):
            def loss_fn(model):
                return config.trainer.trainer.mp.cast_to_compute(model).next_token_loss(
                    batch.tokens,
                    batch.loss_weight,
                    mask=batch.attn_mask,
                    reduction="mean",
                    logsumexp_weight=None,
                    return_router_metrics=True,
                )

            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train._apply_qb_betas(state.params, state.pending_qb_betas)
            )
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]))
            return loss, finite

        def step(state, batch, **step_kwargs):
            nonlocal first
            if first:
                assert sum(x.size for x in jax.tree.leaves(state.params)) == 746607104
                if jax.process_index() == 0 and (wandb.run is None or wandb.run.offline or wandb.run.disabled):
                    raise RuntimeError("Online W&B must initialize before optimizer steps")
                loss, finite = jax.device_get(preflight(state, batch))
                if not bool(finite) or not np.isfinite(loss):
                    raise ValueError("Nonfinite full-model forward/backward preflight")
                persist(
                    "preflight.json", {"loss": float(loss), "finite_gradients": bool(finite), "step": int(state.step)}
                )
                preflight.clear_cache()
                first = False
            result = native(state, batch, **step_kwargs)
            if not np.isfinite(float(result[1]["train/loss"])):
                raise ValueError("Nonfinite training loss")
            if int(result[0].step) == args.steps:
                persist(
                    "training-endpoint.json",
                    {
                        "step": args.steps,
                        "train_loss": float(result[1]["train/loss"]),
                        "final_evaluation_complete": False,
                    },
                )
            return result

        return step

    try:
        with (
            patch.object(train, "_make_train_step", checked_factory),
            patch.object(train, "TaggedEvaluator", bounded_evaluator),
        ):
            train._run_grug_local(config)
        if jax.process_index() == 0 and not (args.output / "training-endpoint.json").exists():
            raise RuntimeError("Training returned before requested endpoint")
        persist("complete.json", {"steps": args.steps, "final_evaluation_complete": True})
    finally:
        if handler is not None:
            handler.upload()


if __name__ == "__main__":
    main()
