# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one immutable HybridNorm transfer configuration with full-state recovery."""

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import uuid
from datetime import timedelta
from pathlib import Path

import draccus
import fsspec
import jax
import jmp
from levanter.checkpoint import CheckpointerConfig, discover_checkpoint_candidates
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.distributed import DistributedConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.main.train_lm import main as train_lm
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig
from rigging.secrets import require_secret

from experiments.hybrid_transfer.model import Residual, TransferConfig
from experiments.hybrid_transfer.optimizer import TransferOptimizer
from experiments.hybrid_transfer.recovery import prepare_recovery, require_final_checkpoint, write_json
from experiments.hybrid_transfer.storage import configure_storage

GROUP = "hybrid-transfer-h100-20261116"


@dataclasses.dataclass(frozen=True)
class RunSpec:
    width: int
    depth: int
    residual: str
    optimizer: str
    lr: float
    seed: int = 0

    @property
    def name(self):
        return f"{self.residual}-{self.optimizer}-w{self.width}-d{self.depth}-lr{self.lr:g}-s{self.seed}"


def configuration(spec, run_id, data_prefix, output_prefix, devices=1, microbatch=8):
    manifest = json.loads(Path(__file__).with_name("data_manifest.json").read_text())
    selected = ("fineweb-edu-10B", "paloma/c4_en")
    components = {
        name: DatasetComponent(cache_dir=f"{data_prefix}/{manifest[name]}", source=UrlDatasetSourceConfig())
        for name in selected
    }
    steps = math.ceil(1_000_000_000 / (128 * 1024))
    model = TransferConfig(
        implementation_digest=hashlib.sha256(
            b"".join(
                Path(__file__).with_name(name).read_bytes()
                for name in ("model.py", "optimizer.py", "train.py", "recovery.py", "storage.py")
            )
        ).hexdigest(),
        hidden_dim=spec.width,
        intermediate_dim=spec.width * 7 // 2,
        num_layers=spec.depth,
        num_heads=spec.width // 64,
        num_kv_heads=spec.width // 64,
        max_seq_len=1024,
        residual=Residual(spec.residual),
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        gradient_checkpointing=True,
        scan_layers=True,
    )
    optimizer = TransferOptimizer(
        variant=spec.optimizer,
        learning_rate=spec.lr,
        adam_lr=0.008,
        width_multiplier=model.width_multiplier,
        depth_multiplier=model.depth_multiplier,
        depth_alpha=model.depth_alpha,
        hidden_lr_multiplier=model.hidden_lr_multiplier,
        weight_decay=0.1 if spec.optimizer == "muonw" else 0.0,
        warmup=0.1,
        lr_schedule="cosine",
        min_lr_ratio=0.0,
    )
    output = f"{output_prefix}/{GROUP}"
    return TrainLmConfig(
        data=LmDataConfig(
            tokenizer="marin-community/marin-tokenizer",
            cache_dir=None,
            auto_build_caches=False,
            shuffle=True,
            permutation_type="linear",
            block_cross_document_attention=True,
            components=components,
            train_weights={name: float(name == "fineweb-edu-10B") for name in selected},
        ),
        model=model,
        optimizer=optimizer,
        trainer=TrainerConfig(
            id=run_id,
            seed=spec.seed,
            mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
            train_batch_size=128,
            per_device_parallelism=microbatch,
            per_device_eval_parallelism=microbatch,
            num_train_steps=steps,
            steps_per_eval=1000,
            max_eval_batches=None,
            mesh=MeshConfig(axes={"data": devices, "replica": 1, "model": 1}, param_mapping={}),
            distributed=DistributedConfig(initialize_jax_distributed=False),
            load_checkpoint=None,
            checkpointer=CheckpointerConfig(
                base_path=f"{output}/state/{spec.name}/checkpoints",
                save_interval=timedelta(minutes=2),
                keep=[{"every": 1000, "until": None}],
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=2,
            ),
            log_dir=Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / run_id,
            log_jaxprs=False,
            log_xla_hlo=False,
            tracker=(
                WandbConfig(
                    entity="marin-community",
                    project="Hyperball",
                    group=GROUP,
                    name=f"{spec.name}-{run_id}",
                    id=run_id,
                    mode="online",
                    resume="never",
                    background=False,
                    save_code=False,
                    tags=[GROUP, spec.residual, spec.optimizer, "hybrid-norm", "mup-init"],
                    replicate_path=f"{output}/tracking/{run_id}",
                ),
                JsonLoggerConfig(logger_name="hybrid_transfer.losses"),
            ),
        ),
        train_seq_len=1024,
        data_seed=42,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="JSON RunSpec")
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--logging-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight-steps", type=int, default=0, help="Validation-only stop point; retain the production LR schedule"
    )
    args = parser.parse_args()
    spec = RunSpec(**json.loads(args.spec))
    run_id = uuid.uuid4().hex[:12]
    config = configuration(spec, run_id, args.data_prefix, args.output_prefix, args.devices, args.microbatch)
    if args.preflight_steps:
        preflight_root = (
            f"{args.output_prefix}/{GROUP}/preflight/{config.model.implementation_digest[:12]}/"
            f"gpu-{args.devices}-mb{args.microbatch}-{spec.name}"
        )
        config = dataclasses.replace(
            config,
            trainer=dataclasses.replace(
                config.trainer,
                checkpointer=dataclasses.replace(config.trainer.checkpointer, base_path=preflight_root),
                max_eval_batches=4,
                tracker=(
                    dataclasses.replace(config.trainer.tracker[0], tags=[GROUP, "gpu-preflight"]),
                    config.trainer.tracker[1],
                ),
            ),
        )
    if args.dry_run:
        print(json.dumps(draccus.encode(config), indent=2))
        return
    require_secret("WANDB_API_KEY")
    configure_storage()
    config.trainer.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hybrid_transfer.losses")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(config.trainer.log_dir / "metrics.jsonl")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    if args.logging_check:
        tracker = dataclasses.replace(config.trainer.tracker[0], tags=[GROUP, "logging-check"]).init(run_id)
        tracker.log_hyperparameters({"resolved_config": draccus.encode(config), "non_training_check": True})
        tracker.log({"preflight/logging_check": 1}, step=0)
        tracker.finish()
        print(f"LOGGING_CHECK_OK https://wandb.ai/marin-community/Hyperball/runs/{run_id}", flush=True)
        return
    if len(jax.devices()) != args.devices or any(d.platform != "gpu" for d in jax.devices()):
        raise RuntimeError("Visible GPUs differ from this configuration's resource request")
    marker = f"{args.output_prefix}/{GROUP}/completed/{spec.name}.json"
    fs, path = fsspec.core.url_to_fs(marker)
    if fs.exists(path):
        print(f"ALREADY_COMPLETED {spec.name}", flush=True)
        return
    config, lineage = prepare_recovery(config, {})
    if args.preflight_steps and lineage["expected_next_step"] >= args.preflight_steps:
        # The preceding attempt committed the validation target before it was
        # interrupted. No optimizer steps or restoration are needed here.
        write_json(
            f"{args.output_prefix}/{GROUP}/preflight/completed/{run_id}.json",
            {
                "spec": dataclasses.asdict(spec),
                "devices": args.devices,
                "checkpoint": lineage["requested_checkpoint"],
                "steps": args.preflight_steps,
                "validation_only": True,
                "status": "already_complete_checkpoint_reused_without_training",
            },
        )
        return
    write_json(f"{args.output_prefix}/{GROUP}/attempts/{run_id}/config.json", draccus.encode(config))
    (config.trainer.log_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))
    print(f"RECOVERY_PLAN {json.dumps({k: v for k, v in lineage.items() if k != 'scientific_config'})}", flush=True)
    original_steps = Trainer.training_steps

    def observed_steps(trainer, state, loader):
        actual_step = int(state.step)
        if actual_step != lineage["expected_next_step"]:
            raise RuntimeError("Restored trainer step differs from the selected complete checkpoint")
        lineage["restoration_status"] = "verified_restored" if lineage["requested_checkpoint"] else "verified_fresh"
        lineage["observed_next_step"] = actual_step
        write_json(f"{args.output_prefix}/{GROUP}/attempts/{run_id}/lineage.json", lineage)
        print(f"VERIFIED_INITIAL_STATE step={actual_step} status={lineage['restoration_status']}", flush=True)
        for info in original_steps(trainer, state, loader):
            yield info
            if args.preflight_steps and int(info.state.step) >= args.preflight_steps:
                break

    Trainer.training_steps = observed_steps
    try:
        train_lm(config)
    finally:
        Trainer.training_steps = original_steps
    if args.preflight_steps:
        candidates = discover_checkpoint_candidates(config.trainer.checkpointer.expanded_path(run_id))
        if not candidates or candidates[-1].step != args.preflight_steps - 1:
            raise RuntimeError("GPU preflight failed to commit its final full checkpoint")
        write_json(
            f"{args.output_prefix}/{GROUP}/preflight/completed/{run_id}.json",
            {
                "spec": dataclasses.asdict(spec),
                "devices": args.devices,
                "checkpoint": candidates[-1].path,
                "wandb_url": lineage["wandb_url"],
                "steps": args.preflight_steps,
                "validation_only": True,
            },
        )
        return
    checkpoint = require_final_checkpoint(config)
    write_json(
        marker,
        {
            "spec": dataclasses.asdict(spec),
            "run_id": run_id,
            "checkpoint": checkpoint,
            "wandb_url": lineage["wandb_url"],
            "completed_steps": config.trainer.num_train_steps,
        },
    )


if __name__ == "__main__":
    main()
