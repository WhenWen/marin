# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original KMA d512 on PR 10678's frozen regional caches."""

import argparse
import hashlib
import json
import logging
import math
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
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import BlockShuffleConfig, DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.distributed import DistributedConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.fast_track import train
from experiments.grug.fast_track.heuristic import MoeHeuristic
from experiments.grug.fast_track.model import AttnResLayerBackward, GrugModelConfig, LocalMixer
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.muon_mup_tpu.optimizer import MuonComparisonConfig

HERE = Path(__file__).parent
SOURCE_COMMIT = "4eba59cafa27a1ea174d1b5d6a28d9d1660eba10"
DATA_COMMIT = "caff3778edcb23d41a154ba962b67daf3d669fab"
REFERENCE_STEPS = 2475
REFERENCE_BATCH = 128
SEQUENCE_LENGTH = 4096
REFERENCE_MUON_LR = 0.015857567136604373
REFERENCE_HEAD_LR = 0.02242598651082779
REFERENCE_ADAM_LR = 0.005175227656344875


def build_config(args, run_id, checkpoint_root):
    model = GrugModelConfig(
        vocab_size=128256,
        hidden_dim=512,
        num_layers=6,
        initializer_std=0.5 / math.sqrt(512),
        local_mixer=LocalMixer.KDA,
        mla=True,
        inkling_relpos=True,
        qk_norm=False,
        qk_mult=1.0,
        attn_res=True,
        attn_res_num_blocks=8,
        sconv_sites=("attn", "mlp"),
        attn_res_layer_backward=AttnResLayerBackward.RECOMPUTE,
        kda_save_chunk_states=False,
    )
    scaling = math.sqrt(args.batch_size / REFERENCE_BATCH)
    heuristic = MoeHeuristic().build_optimizer_config(
        num_train_steps=args.steps, batch_size=args.batch_size, hidden_dim=512, seq_len=SEQUENCE_LENGTH
    )
    optimizer_args = dict(
        learning_rate=REFERENCE_HEAD_LR * scaling,
        adam_lr=REFERENCE_ADAM_LR * scaling,
        warmup=0.01,
        min_lr_ratio=0.05,
        lr_schedule="linear",
        beta1=heuristic.beta1,
        beta2=heuristic.beta2,
        epsilon=heuristic.epsilon,
    )
    optimizer = GrugMoeMuonHConfig(**optimizer_args)
    if args.optimizer != "muonh":
        optimizer_args["learning_rate"] = REFERENCE_MUON_LR * scaling * args.lr_mult
        optimizer = MuonComparisonConfig(
            **optimizer_args, spectral=args.optimizer == "muon-mup", head_lr=REFERENCE_HEAD_LR * scaling
        )
    values = json.loads((HERE / "july_data.json").read_text())
    components = {
        name: DatasetComponent(
            cache_dir=value["cache_dir"],
            tags=value.get("tags"),
            source=UrlDatasetSourceConfig(),
            format=TextLmDatasetFormat(text_key="text"),
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
        group="muon-mup-kma-july-20261028",
        tags=[args.optimizer, "kma-d512", "tpu"],
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
        watch=WatchConfig(interval=0),
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
            save_interval=timedelta(minutes=20),
        ),
        # Mosaic grouped matmul expects native bf16 products for bf16 inputs.
        jax_config={"jax_enable_pgle": False, "jax_default_matmul_precision": "default"},
    )
    return train.GrugRunConfig(
        model=model,
        optimizer=optimizer,
        data=data,
        resources=ResourceConfig.with_tpu("v4-32"),
        tensorstore_cache_bytes=1024**3,
        trainer=train.GrugTrainerConfig(
            trainer=trainer,
            data_seed=0,
            expert_axis_size=1,
            replica_axis_size=1,
            save_checkpoints=True,
        ),
        eval=train.GrugEvalConfig(
            eval_batch_size=16,
            steps_per_eval=args.steps // 2,
            compute_bpb=True,
            dropless_eval=True,
            dropless_eval_moe_implementation="scatter",
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
    parser.add_argument("--optimizer", choices=("muonh", "muon", "muon-mup"), required=True)
    parser.add_argument("--lr-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--entity", default="marin-community")
    parser.add_argument("--project", default="marin")
    parser.add_argument("--output", type=Path, default=Path("/tmp/muon-mup-results"))
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
            "source_commit": SOURCE_COMMIT,
            "data_commit": DATA_COMMIT,
            "data_sha256": hashlib.sha256((HERE / "july_data.json").read_bytes()).hexdigest(),
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
        kwargs["max_examples_per_dataset"] = 2048
        return evaluator(**kwargs)

    def checked_factory(*params, **kwargs):
        native = native_factory(*params, **kwargs)
        first = True

        @jax.jit
        def preflight(state, batch):
            (loss, _), grads = train._loss_and_grads(
                train._apply_qb_betas(state.params, state.pending_qb_betas),
                batch,
                config.trainer.trainer.mp,
                config.trainer.z_loss_weight,
            )
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]))
            return loss, finite

        def step(state, batch):
            nonlocal first
            if first:
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
            result = native(state, batch)
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
            patch.object(train, "_TRAIN_LOADER_BUFFER_SIZE", 4),
        ):
            train._run_grug_local(config)
        persist("complete.json", {"steps": args.steps, "final_evaluation_complete": True})
    finally:
        if handler is not None:
            handler.upload()


if __name__ == "__main__":
    main()
