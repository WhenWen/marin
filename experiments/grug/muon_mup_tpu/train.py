# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KMA d512 and d768 on PR 10678's frozen regional caches."""

import argparse
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jmp
import levanter.tracker
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

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.fast_track import train
from experiments.grug.fast_track.heuristic import MoeHeuristic
from experiments.grug.fast_track.model import AttnResLayerBackward, GrugModelConfig, LocalMixer
from experiments.grug.muon_mup_tpu.data_guard import (
    CHECK_INTERVAL,
    CacheIntegrityError,
    batch_statistics,
    check_caches,
    required_caches,
    validate_batch_statistics,
)
from experiments.grug.muon_mup_tpu.optimizer import (
    MuonComparisonConfig,
    MuonHSweepConfig,
    MuonMDConfig,
    MuonWComparisonConfig,
    MuownConfig,
    NorMuonHConfig,
    RowNormalizedMuonHConfig,
)
from experiments.grug.muon_mup_tpu.qualify import (
    QualificationComplete,
    profile_steps,
    qualify,
    qualify_d768_cross_entropy,
)
from experiments.grug.muon_mup_tpu.recovery import batch_fingerprint, prepare_recovery, restore_or_initialize
from experiments.grug.muon_mup_tpu.tpu_execution import execution_mode

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
    hidden = int(args.size[1:])
    model = GrugModelConfig(
        vocab_size=16384 if args.ladder16k else 128256,
        hidden_dim=hidden,
        num_layers=6 if hidden == 512 else 8,
        num_heads=hidden // 128,
        intermediate_dim=hidden // 2,
        shared_expert_intermediate_dim=hidden // 2,
        latent_dim=hidden // 2,
        initializer_std=0.5 / math.sqrt(hidden),
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
    reference_batch = args.optimizer_reference_batch_size or args.batch_size
    reference_steps = args.optimizer_reference_steps or args.steps
    scaling = math.sqrt(reference_batch / REFERENCE_BATCH)
    heuristic = MoeHeuristic().build_optimizer_config(
        num_train_steps=reference_steps, batch_size=reference_batch, hidden_dim=hidden, seq_len=SEQUENCE_LENGTH
    )
    head_lr = REFERENCE_HEAD_LR * scaling if hidden == 512 else heuristic.learning_rate
    adam_lr = REFERENCE_ADAM_LR * scaling if hidden == 512 else heuristic.adam_lr
    muon_lr = REFERENCE_MUON_LR * scaling if hidden == 512 else head_lr / math.sqrt(2)
    optimizer_args = dict(
        learning_rate=head_lr,
        adam_lr=adam_lr,
        # Match the reference's integer warmup boundary in tokens, including its
        # rounding, rather than independently rounding the new step count.
        warmup=(
            int(0.01 * reference_steps) * reference_batch // args.batch_size
            if args.optimizer_reference_steps is not None
            else 0.01
        ),
        min_lr_ratio=0.05,
        lr_schedule="linear",
        beta1=heuristic.beta1,
        beta2=heuristic.beta2,
        epsilon=heuristic.epsilon,
    )
    optimizer_args["learning_rate"] *= args.lr_mult
    optimizer = MuonHSweepConfig(**optimizer_args, head_lr=head_lr)
    if args.optimizer in ("normuonh", "row_muonh"):
        cls = NorMuonHConfig if args.optimizer == "normuonh" else RowNormalizedMuonHConfig
        optimizer = cls(**optimizer_args, head_lr=head_lr)
    if args.optimizer in ("muon", "muonw"):
        optimizer_args["learning_rate"] = muon_lr * args.lr_mult
        cls = MuonWComparisonConfig if args.optimizer == "muonw" else MuonComparisonConfig
        optimizer = cls(**optimizer_args, head_lr=head_lr)
    if args.optimizer in ("muown", "muon_md"):
        optimizer_args["learning_rate"] = (0.001 if args.optimizer == "muown" else 0.01) * args.lr_mult
        cls = MuownConfig if args.optimizer == "muown" else MuonMDConfig
        optimizer = cls(**optimizer_args, head_lr=head_lr)
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
    if args.ladder16k:
        values = json.loads((HERE / "ladder16k_data.json").read_text())["data"]
        data = LmDataConfig(
            tokenizer=args.tokenizer,
            components={
                name: DatasetComponent(
                    cache_dir=value["cache_dir"],
                    flat_cache=value["flat_cache"],
                    tags=value["tags"],
                    source=UrlDatasetSourceConfig(),
                    format=TextLmDatasetFormat(text_key="text"),
                )
                for name, value in values["components"].items()
            },
            train_weights=values["train_weights"],
            auto_build_caches=False,
            cache_dir=None,
            shuffle=BlockShuffleConfig(**values["shuffle"]),
            permutation_type=values["permutation_type"],
            block_cross_document_attention=values["block_cross_document_attention"],
            mixture_block_size=values["mixture_block_size"],
        )
    tracker = WandbConfig(
        entity=args.entity,
        project=args.project,
        name=run_id,
        id=run_id,
        group=args.wandb_group
        or ("muon-mup-kma-larry16k-b128-20261107" if args.ladder16k else "muon-mup-kma-july-20261028"),
        tags=[args.optimizer, "kma-" + args.size, "h100" if args.h100 else "tpu"],
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
        allow_partial_checkpoint=False,
        checkpointer=CheckpointerConfig(
            base_path=checkpoint_root,
            append_run_id_to_base_path=False,
            temporary_base_path=checkpoint_root + "/temporary",
            keep=[dict(every=250)],
            save_interval=timedelta(minutes=5),
            keep_last_temporary_checkpoints=2,
        ),
        # Mosaic grouped matmul expects native bf16 products for bf16 inputs.
        jax_config={"jax_enable_pgle": False, "jax_default_matmul_precision": "default"},
    )
    return train.GrugRunConfig(
        model=model,
        optimizer=optimizer,
        data=data,
        resources=(
            ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="600g", disk="900g", replicas=1)
            if args.h100
            else ResourceConfig.with_tpu(args.tpu)
        ),
        processes_per_task=8 if args.h100 else 1,
        tensorstore_cache_bytes=1024**3,
        trainer=train.GrugTrainerConfig(
            trainer=trainer,
            data_seed=None if args.ladder16k else 0,
            expert_axis_size=8 if args.h100 else 1,
            replica_axis_size=1,
            save_checkpoints=True,
        ),
        eval=train.GrugEvalConfig(
            eval_batch_size=8 if args.h100 else 16,
            steps_per_eval=args.steps // 2,
            compute_bpb=True,
            dropless_eval=True,
            dropless_eval_moe_implementation="sonic" if args.h100 else "scatter",
        ),
    )


def persist_json_artifact(output: Path, destination: str, name: str, value) -> None:
    """Write identical JSON evidence locally and to the artifact store."""
    content = json.dumps(value, default=str, indent=2).encode()
    local_path = output / name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(content)
    (StoragePath(destination) / name).write_bytes(content)


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
    parser.add_argument(
        "--optimizer", choices=("muonh", "muon", "muonw", "muown", "muon_md", "normuonh", "row_muonh"), required=True
    )
    parser.add_argument("--size", choices=("d512", "d768"), default="d512")
    parser.add_argument("--tpu", choices=("v4-8", "v4-32"), default="v4-32")
    parser.add_argument("--h100", action="store_true")
    parser.add_argument("--ladder16k", action="store_true")
    parser.add_argument("--lr-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--optimizer-reference-batch-size", type=int)
    parser.add_argument("--optimizer-reference-steps", type=int)
    parser.add_argument("--wandb-group")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--entity", default="marin-community")
    parser.add_argument("--project", default="marin")
    parser.add_argument("--output", type=Path, default=Path("/tmp/muon-mup-results"))
    parser.add_argument("--logging-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execution", choices=("reference", "histogram", "tpu", "inkling"), default="reference")
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--qualify-ce", action="store_true")
    parser.add_argument("--qualification-modes", nargs="+", default=["reference", "histogram", "tpu"])
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--compare-checkpoint")
    args = parser.parse_args()
    if (args.optimizer_reference_steps is None) != (args.optimizer_reference_batch_size is None):
        parser.error("Optimizer reference batch size and steps must be supplied together")
    if args.optimizer_reference_steps is not None:
        if args.optimizer_reference_steps * args.optimizer_reference_batch_size != args.steps * args.batch_size:
            parser.error("A fixed-optimizer batch ablation must preserve the reference token budget")
        if int(0.01 * args.optimizer_reference_steps) * args.optimizer_reference_batch_size % args.batch_size:
            parser.error("The reference warmup token boundary must align with the new batch size")
    args.output.mkdir(parents=True, exist_ok=True)
    identity_file = "ladder16k_tokenizer_identity.json" if args.ladder16k else "tokenizer_identity.json"
    identities = json.loads((HERE / identity_file).read_text())
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
    config = build_config(args, run_id, args.artifact_root.rstrip("/") + "/checkpoints")
    requires_restore = prepare_recovery(config, args.artifact_root, attempt)
    config = replace(
        config,
        stop_after_steps=args.stop_after_steps,
        trainer=replace(config.trainer, trainer=replace(config.trainer.trainer, load_checkpoint=requires_restore)),
    )

    def persist(name, value):
        if jax.process_index() == 0:
            persist_json_artifact(args.output, root, name, value)

    persist("resolved-config.json", asdict(config))
    persist(
        "identity.json",
        {
            "run_id": run_id,
            "size": args.size,
            "accelerator": "H100x8" if args.h100 else args.tpu,
            "tpu": None if args.h100 else args.tpu,
            "data_recipe": "larry-ladder16k" if args.ladder16k else "july",
            "execution": args.execution,
            "execution_sources": {
                name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                for name in (
                    "train.py",
                    "optimizer.py",
                    "decoupled.py",
                    "row_normalized.py",
                    "recovery.py",
                    "tpu_execution.py",
                    "kda_state_tpu.py",
                    "qualify.py",
                    "inkling_tpu.py",
                    "data_guard.py",
                )
            },
            "qualification": args.qualify,
            "ce_qualification": args.qualify_ce,
            "profile_steps": args.profile_steps,
            "source_commit": SOURCE_COMMIT,
            "data_commit": None if args.ladder16k else DATA_COMMIT,
            "data_sha256": (
                hashlib.sha256(
                    (HERE / ("ladder16k_data.json" if args.ladder16k else "july_data.json")).read_bytes()
                ).hexdigest()
            ),
            "seed": args.seed,
            "optimizer_reference_batch_size": args.optimizer_reference_batch_size,
            "optimizer_reference_steps": args.optimizer_reference_steps,
            "steps": args.steps,
            "tokens": args.steps * args.batch_size * SEQUENCE_LENGTH,
            "restart_semantics": "shared compatible complete checkpoints; fresh W&B run per process group",
            "requires_restore": requires_restore,
            "wandb_url": f"https://wandb.ai/{args.entity}/{args.project}/runs/{run_id}",
        },
    )

    def restore_state(state, **kwargs):
        restored = restore_or_initialize(state, config=config, persist=persist, **kwargs)
        if int(restored.step) == args.steps:
            persist(
                "training-endpoint.json",
                {"step": args.steps, "restored_at_endpoint": True, "final_evaluation_complete": False},
            )
        return restored

    handler = None
    if jax.process_index() == 0:
        handler = MetricsFileHandler(args.output / "metrics.jsonl", root + "/metrics.jsonl")
        logging.getLogger("levanter.json_logger").addHandler(handler)
    native_factory = train._make_train_step
    evaluator = train.TaggedEvaluator
    cache_paths = required_caches(config.data)
    cache_baseline = None

    def check_data(step, *, tracker_active=True):
        nonlocal cache_baseline
        paths = cache_paths
        try:
            snapshot = check_caches(paths, cache_baseline)
        except CacheIntegrityError as exc:
            persist(f"data-integrity/step-{step}.json", {"step": step, "passed": False, "caches": exc.report})
            if tracker_active:
                levanter.tracker.log({"data_integrity/passed": 0}, step=step)
            raise
        if cache_baseline is None:
            cache_baseline = snapshot
        persist(f"data-integrity/step-{step}.json", {"step": step, "passed": True, "caches": snapshot})
        if tracker_active:
            levanter.tracker.log({"data_integrity/passed": 1, "data_integrity/cache_count": len(paths)}, step=step)

    def bounded_evaluator(**kwargs):
        kwargs["max_examples_per_dataset"] = None if args.ladder16k else 2048
        return evaluator(**kwargs)

    def checked_factory(*params, **kwargs):
        # Trainer initializes online tracking before constructing this function,
        # which runs before model initialization or dataset cache loading.
        check_data(0)
        if args.profile_steps:

            def profile(state, batch):
                profile_steps(
                    state,
                    batch,
                    *params,
                    z_loss_weight=kwargs["z_loss_weight"],
                    native_factory=native_factory,
                    directory=args.output / "profile",
                    count=args.profile_steps,
                    persist=persist,
                )

            return profile
        if args.qualify:

            def diagnostic(state, batch):
                qualify(
                    state,
                    batch,
                    *params,
                    z_loss_weight=kwargs["z_loss_weight"],
                    native_factory=native_factory,
                    persist=persist,
                    modes=tuple(args.qualification_modes),
                )

            return diagnostic
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
            step_index = int(state.step)
            statistics = {
                name: value.item()
                for name, value in jax.device_get(batch_statistics(batch.tokens, batch.loss_weight)).items()
            }
            try:
                validate_batch_statistics(statistics)
            except ValueError:
                persist(f"batch-diagnostics/step-{step_index}.json", {"step": step_index, **statistics})
                levanter.tracker.log({"data_integrity/batch_passed": 0}, step=step_index)
                raise
            if step_index % CHECK_INTERVAL == 0:
                if step_index > 0:
                    check_data(step_index)
                persist(f"batch-diagnostics/step-{step_index}.json", {"step": step_index, **statistics})
                levanter.tracker.log(
                    {f"data_batch/{name}": value for name, value in statistics.items()}, step=step_index
                )
            if first:
                if jax.process_index() == 0 and (wandb.run is None or wandb.run.offline or wandb.run.disabled):
                    raise RuntimeError("Online W&B must initialize before optimizer steps")
                if args.qualify_ce:
                    qualify_d768_cross_entropy(persist)
                loss, finite = jax.device_get(preflight(state, batch))
                if not bool(finite) or not np.isfinite(loss):
                    raise ValueError("Nonfinite full-model forward/backward preflight")
                persist(
                    "preflight.json", {"loss": float(loss), "finite_gradients": bool(finite), "step": int(state.step)}
                )
                preflight.clear_cache()
                first = False
            if args.compare_checkpoint or args.stop_after_steps:
                persist(
                    f"continuation/batch-{step_index}.json",
                    {"step": step_index, "token_fingerprint": batch_fingerprint(batch.tokens)},
                )
            result = native(state, batch)
            if args.compare_checkpoint and int(result[0].step) == args.stop_after_steps:
                reference = restore_grug_state_from_checkpoint(
                    result[0],
                    checkpoint_search_paths=[args.compare_checkpoint],
                    load_checkpoint_setting=True,
                    mesh=jax.sharding.get_mesh(),
                    allow_partial=False,
                )
                deviations = [
                    jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))
                    for a, b in zip(jax.tree.leaves(result[0]), jax.tree.leaves(reference), strict=True)
                ]
                max_error = float(jax.device_get(jnp.max(jnp.stack(deviations))))
                persist(
                    "continuation-parity.json",
                    {"step": int(result[0].step), "max_absolute_error": max_error, "passed": max_error == 0.0},
                )
                if max_error != 0.0:
                    raise ValueError(f"Interrupted continuation differs from uninterrupted state: {max_error}")
            if not np.isfinite(float(result[1]["train/loss"])):
                raise ValueError("Nonfinite training loss")
            if int(result[0].step) == args.steps:
                check_data(args.steps - 1)
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
            patch.object(train, "restore_grug_state_from_checkpoint", restore_state),
            patch.object(train, "TaggedEvaluator", bounded_evaluator),
            patch.object(train, "_TRAIN_LOADER_BUFFER_SIZE", 4),
            execution_mode(args.execution),
        ):
            train._run_grug_local(config)
        if args.stop_after_steps is not None and args.stop_after_steps < args.steps:
            persist("prefix-complete.json", {"step": args.stop_after_steps, "full_training_run": False})
            return
        if jax.process_index() == 0 and not (args.output / "training-endpoint.json").exists():
            raise RuntimeError("Training returned before requested endpoint")
        # The trainer has finished W&B; validate again after final evaluation
        # before recording completion, with local/GCS evidence only.
        check_data(args.steps, tracker_active=False)
        persist("complete.json", {"steps": args.steps, "final_evaluation_complete": True})
    except QualificationComplete:
        levanter.tracker.current_tracker().finish()
        if args.profile_steps:
            for file in (args.output / "profile").rglob("*"):
                if file.is_file():
                    destination = (
                        root + f"/profile/host-{jax.process_index()}/" + str(file.relative_to(args.output / "profile"))
                    )
                    StoragePath(destination).write_bytes(file.read_bytes())
            persist("profile-complete.json", {"full_training_run": False, "profile_steps": args.profile_steps})
        else:
            persist("qualification-complete.json", {"full_training_run": False})
    finally:
        if handler is not None:
            handler.upload()


if __name__ == "__main__":
    main()
