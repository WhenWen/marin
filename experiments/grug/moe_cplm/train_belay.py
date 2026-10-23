# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay / Belay+CPLM / Belay-CPLM-plus at frozen TPU ladder budgets."""

import argparse
import hashlib
import json
import logging
import math
import os
import tarfile
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import draccus
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from fray.cluster import ResourceConfig
from jax.experimental.multihost_utils import broadcast_one_to_all, process_allgather, sync_global_devices
from jax.sharding import PartitionSpec as P
from levanter.data.text.datasets import BlockShuffleConfig, DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.distributed import DistributedConfig
from levanter.models.lm_model import LmExample
from levanter.optim.config import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.fast_track import model, qb_balancing, train
from experiments.grug.fast_track.column_sharded_embedding import column_sharded_embedding
from experiments.grug.fast_track.heuristic import MoeHeuristic
from experiments.grug.fast_track.router_lookup import unrolled_router_logits
from experiments.grug.moe import kda
from experiments.grug.moe.kda_state_tpu import state_pass
from experiments.grug.moe_cplm import accumulation, case_classes, plus
from experiments.grug.moe_cplm.belay import Belay, CopyPasteBelay
from experiments.grug.moe_cplm.checkpoint_io import python_zarr_checkpoint_writes
from experiments.grug.moe_cplm.frame_kda import FRAME_CONFIG
from experiments.grug.moe_cplm.frame_kda_model import FrameKDAPlus
from experiments.grug.moe_cplm.frame_kda_validation import validate_frame_kda
from experiments.grug.moe_cplm.head_validation import validate_head, validate_histogram
from experiments.grug.moe_cplm.stack_kda_model import StackKimiDeltaAttention
from experiments.grug.moe_cplm.stack_kda_validation import validate_stack_kda

RECIPE = Path(__file__).with_name("belay_recipe.json")
DATA = Path(__file__).with_name("july_data.json")
REFERENCE_PROJECTION = model._proj
MODELS = {"belay": Belay, "cplm": CopyPasteBelay, "plus": plus.BelayCPLMPlus, "plus-frame-kda": FrameKDAPlus}
MODELS["plus-stack4-kda"] = plus.BelayCPLMPlus


def gathered_projection(cfg, x, weight, out_sharding=None):
    return REFERENCE_PROJECTION(cfg, x, jax.sharding.reshard(weight, P(None, None)), out_sharding)


def tpu_recurrence(prep, state, mm, lead, dk, dv, *, scan_impl):
    """Run the qualified state recurrence inside the KDA local shard."""
    chunks, chunk_size = prep.kw.shape[-3:-1]

    def flatten(x):
        return x.reshape(-1, chunks, *x.shape[-2:])

    output, final = state_pass(
        flatten(prep.q_inflate),
        flatten(prep.attn),
        flatten(prep.kw),
        flatten(prep.k_cumdecay),
        flatten(prep.v_pseudo),
        prep.decay.reshape(-1, chunks, 1, dk),
        state.reshape(-1, dk, dv),
        precision=jax.lax.Precision.HIGH,
    )
    return output.reshape(*lead, chunks, chunk_size, dv), final.reshape(*lead, dk, dv)


@contextmanager
def execution(variant="cplm", tables=None):
    """Bind TPU implementations for this single-run process, including eval."""
    with ExitStack() as stack:
        if variant == "plus-stack4-kda":
            stack.enter_context(patch.object(model, "KimiDeltaAttention", StackKimiDeltaAttention))
        for module, name, implementation in (
            (train, "Transformer", MODELS[variant]),
            (train, "_accumulated_loss_and_grads", accumulation.accumulated_loss_and_grads),
            (plus, "CASE_TABLES", tables),
            (model, "_proj", gathered_projection),
            (model, "_replicated_fp32_gather", column_sharded_embedding),
            (model, "_selected_router_logits", unrolled_router_logits),
            (kda, "_parallel_chunk_recurrence", tpu_recurrence),
            (
                qb_balancing,
                "bincount_upper_quantile",
                partial(qb_balancing.bincount_upper_quantile, histogram_method="two-stage"),
            ),
        ):
            stack.enter_context(patch.object(module, name, implementation))
        yield


def build_config(
    *,
    data_path: Path,
    output: Path,
    checkpoint_path: str,
    run_name: str,
    wandb_entity: str,
    wandb_project: str,
    calibration: bool,
    tokenizer_path: str,
    width: int = 512,
    steps: int = 10980,
    batch_size: int = 16,
    tpu_chips: int = 16,
    grad_accum_microbatches: int = 1,
    expert_axis_size: int = 1,
    expert_chunks: int = 2,
) -> train.GrugRunConfig:
    """Freeze the optimizer batch and schedule; calibration stops after 400 updates."""
    if batch_size % tpu_chips or batch_size < tpu_chips:
        raise ValueError("Logical batch must contain an integer number of one-sequence-per-chip microbatches")
    if batch_size % (tpu_chips * grad_accum_microbatches) or tpu_chips % expert_axis_size:
        raise ValueError("Each microbatch and expert mesh must divide evenly across the TPU chips")
    source = json.loads(RECIPE.read_text())
    model_values = source["model"]
    model_values.update(
        kda_write_gate=False,
        kda_erase_gate=False,
        kda_onchip_erase_gate=False,
        attn_res_layer_backward="RECOMPUTE" if width >= 1024 else "SAVE",
        attn_res_remat_attention=width >= 1280,
        kda_save_chunk_states=False,
        kda_state_grads="XLA",
        kda_xla_scan_impl="parallel_stacked_split",
        hidden_dim=width,
        qb_defer_update=grad_accum_microbatches > 1,
        moe_ragged_chunks=expert_chunks,
        vocab_size=128256,
        max_seq_len=8192,
    )
    layers = {512: 6, 768: 8, 1024: 12, 1280: 14}[width]
    model_values.update(
        num_layers=layers, latent_out_full_layers=[layers - 2, layers - 1], initializer_std=0.5 / math.sqrt(width)
    )
    for name in (
        "intermediate_dim",
        "shared_expert_intermediate_dim",
        "latent_dim",
        "mla_num_heads",
        "num_heads",
        "global_local_heads",
    ):
        assert model_values[name] * width % 512 == 0
        model_values[name] = model_values[name] * width // 512
    if expert_axis_size > 1:
        # Each receiver chunk can accept every valid assignment in its EP group.
        # The backend divides capacity by both EP size and chunk count, so this
        # makes clipping an identity even if all tokens route to one chunk.
        model_values["capacity_factor"] = expert_axis_size * model_values["moe_ragged_chunks"]
    model_config = draccus.decode(model.GrugModelConfig, model_values)
    optimizer = source["optimizer"]
    optimizer["type"] = "grug_fast_track_muonh_v1"
    heuristic = MoeHeuristic().build_optimizer_config(
        num_train_steps=steps, batch_size=batch_size, hidden_dim=width, seq_len=8192
    )
    for name in ("learning_rate", "adam_lr", "epsilon", "beta2"):
        optimizer[name] = getattr(heuristic, name)
    outer = source["trainer"]
    outer.update(
        expert_axis_size=expert_axis_size,
        replica_axis_size=1,
        grad_accum_microbatches=grad_accum_microbatches,
        pipeline_depth=1,
        loader_buffer_batches=4,
        z_loss_weight=0.0,
        ema_beta=None,
        ema_last_steps=None,
        ema_blend_sweep=[],
        ema_group_sweep=False,
        save_checkpoints=not calibration,
        completion_marker_path=checkpoint_path + "/final_eval_done.json",
    )
    inner = outer["trainer"]
    inner.update(
        id=run_name,
        seed=0,
        num_train_steps=steps,
        train_batch_size=batch_size,
        per_device_parallelism=batch_size // tpu_chips,
        per_device_eval_parallelism=1,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        load_checkpoint=False if calibration else None,
        load_checkpoint_path=None,
        log_dir=str(output / "logs"),
        tracker=[{"type": "json_logger"}],
        log_jaxprs=False,
        log_xla_hlo=False,
        progress_watchdog={"step_timeout": None, "process_timeout": None, "startup_timeout": None},
    )
    inner["jax_config"]["jax_enable_pgle"] = False
    inner["checkpointer"].update(
        base_path=checkpoint_path,
        temporary_base_path=checkpoint_path + "/temporary",
        keep=[{"every": max(1, 1000 * 16 // batch_size)}],
        save_interval="0:10:00",
        append_run_id_to_base_path=False,
    )
    inner["checkpointer"]["write"].update(
        max_staged_host_bytes=256 * 1024**2,
        max_chunk_bytes=64 * 1024**2,
        cache_pool_bytes=64 * 1024**2,
        data_copy_concurrency=1,
    )
    trainer_config = draccus.decode(train.GrugTrainerConfig, outer)
    tracker = WandbConfig(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        id=run_name,
        mode="online",
        resume="never",
        save_code=False,
        background=False,
    )
    trainer_config = replace(
        trainer_config,
        trainer=replace(
            trainer_config.trainer,
            tracker=(*trainer_config.trainer.tracker, tracker),
            jax_config=dict(inner["jax_config"]),
        ),
    )
    data_values = json.loads(data_path.read_text())
    components = {
        name: DatasetComponent(
            cache_dir=value["cache_dir"],
            tags=value.get("tags"),
            source=UrlDatasetSourceConfig(),
            format=TextLmDatasetFormat(text_key=value.get("text_key", "text")),
        )
        for name, value in data_values["data_components"].items()
    }
    data = LmDataConfig(
        tokenizer=tokenizer_path,
        components=components,
        train_weights={name: data_values["train_weights"].get(name, 0.0) for name in components},
        cache_dir=None,
        auto_build_caches=False,
        shuffle=BlockShuffleConfig(**data_values["shuffle"]),
        permutation_type="feistel",
        block_cross_document_attention=True,
        mixture_block_size=data_values["mixture_block_size"],
    )
    return train.GrugRunConfig(
        model=model_config,
        optimizer=draccus.decode(OptimizerConfig, optimizer),
        data=data,
        trainer=trainer_config,
        resources=ResourceConfig.with_tpu(f"v4-{2 * tpu_chips}"),
        eval=(
            None
            if calibration
            else train.GrugEvalConfig(
                eval_batch_size=256,
                steps_per_eval=max(1, 1000 * 16 // batch_size),
                compute_bpb=True,
                dropless_eval=True,
                dropless_eval_moe_implementation="scatter",
            )
        ),
        tensorstore_cache_bytes=1024**3,
        stop_after_steps=400 if calibration else None,
    )


class MetricsFileHandler(logging.FileHandler):
    """Retain partial native loss logs in regional storage during training."""

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DATA)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--logging-check", action="store_true")
    parser.add_argument("--variant", choices=tuple(MODELS), default="cplm")
    parser.add_argument("--width", type=int, choices=(512, 768, 1024, 1280), default=512)
    parser.add_argument("--batch-size", type=int, choices=(16, 32, 64, 128, 256), default=16)
    parser.add_argument("--tpu-chips", type=int, choices=(16, 32), default=16)
    parser.add_argument("--grad-accum-microbatches", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--expert-axis-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--expert-chunks", type=int, choices=(1, 2), default=2)
    parser.add_argument("--memory-scheduler", choices=("default", "brkga"), default="default")
    parser.add_argument("--steps", type=int, default=10980)
    parser.add_argument("--compute-budget", type=Path)
    args = parser.parse_args(argv)
    plus_variant = args.variant in ("plus", "plus-frame-kda", "plus-stack4-kda")
    if args.variant == "plus-frame-kda" and args.width != 512:
        raise ValueError("The protected-frame experiment is frozen at d512")
    run_name = args.run_name
    args.output.mkdir(parents=True, exist_ok=True)
    if args.logging_check:
        if os.environ.get("IRIS_TASK_ID"):
            raise ValueError("Run the logging check once on the remote submit host before Iris submission")
        tracker = WandbConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name + "-logging-check",
            id="logcheck-" + uuid.uuid4().hex,
            mode="online",
            resume="never",
            background=False,
        ).init(None)
        tracker.log_hyperparameters({"training_updates": 0})
        tracker.log({"preflight/ok": 1}, step=0)
        tracker.finish()
        return
    memory_schedule = args.width == 768 and args.batch_size == 64
    compiler_dump = args.output / "compiler-dump"
    flags = ["--xla_allow_excess_precision=false"]
    tpu_flags = ["--xla_memory_scheduler=brkga"] if args.memory_scheduler == "brkga" else []
    if memory_schedule:
        flags += [
            f"--xla_dump_to={compiler_dump}",
            "--xla_dump_hlo_as_proto=true",
            "--xla_dump_hlo_as_text=true",
            "--xla_dump_hlo_module_re=jit_train_step",
        ]
    for variable, values in (("XLA_FLAGS", flags), ("LIBTPU_INIT_ARGS", tpu_flags)):
        for flag in values:
            if flag not in os.environ.get(variable, "").split():
                os.environ[variable] = os.environ.get(variable, "") + " " + flag
    # Iris task-attempt IDs differ by host. Broadcast one fresh process-group
    # identity before constructing tracker and checkpoint paths.
    jax_settings = json.loads(RECIPE.read_text())["trainer"]["trainer"]["jax_config"]
    jax_settings["jax_enable_pgle"] = False
    for name, value in jax_settings.items():
        jax.config.update(name, value)
    DistributedConfig().initialize()
    identity = broadcast_one_to_all(np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8).copy())
    attempt = bytes(identity).hex()
    run_name = f"{args.run_name}-{attempt}"
    args.output = args.output / attempt
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_root = args.artifact_root.rstrip("/") + "/" + attempt
    if args.variant == "plus-stack4-kda":
        # Each restart starts a new scientific trajectory and W&B curve.
        args.checkpoint_path = args.checkpoint_path.rstrip("/") + "/" + attempt

    def persist(name, value):
        if jax.process_index() == 0:
            content = json.dumps(value, default=str, indent=2).encode()
            (args.output / name).write_bytes(content)
            StoragePath(artifact_root + "/" + name).write_bytes(content)

    def retain_compiler_dump():
        if memory_schedule and compiler_dump.exists():
            try:
                archive = args.output / f"compiler-dump-host{jax.process_index()}.tar.gz"
                with tarfile.open(archive, "w:gz", compresslevel=1) as packed:
                    packed.add(compiler_dump, arcname="compiler-dump")
                StoragePath(artifact_root + "/" + archive.name).write_bytes(archive.read_bytes())
            except Exception:
                logging.exception("Could not retain compiler dump")

    identities = json.loads(Path(__file__).with_name("tokenizer_identity.json").read_text())
    for name, expected in identities.items():
        if hashlib.sha256((Path(args.tokenizer) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Tokenizer identity mismatch: {name}")
    config = build_config(
        data_path=args.data_config,
        output=args.output,
        checkpoint_path=args.checkpoint_path,
        run_name=run_name,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        calibration=args.calibration,
        tokenizer_path=args.tokenizer,
        width=args.width,
        steps=args.steps,
        batch_size=args.batch_size,
        tpu_chips=args.tpu_chips,
        grad_accum_microbatches=args.grad_accum_microbatches,
        expert_axis_size=args.expert_axis_size,
        expert_chunks=args.expert_chunks,
    )
    budget = None
    if args.variant == "plus-stack4-kda":
        if args.width != 512:
            raise ValueError("The layer-internal stack comparison is frozen at d512")
    execution_policy = "native-saved-global-qb" if args.grad_accum_microbatches > 1 else "native-saved"
    if args.width >= 1024:
        execution_policy = "native-ladder-recompute"
    if args.memory_scheduler == "brkga":
        execution_policy += "-memory-schedule"
    if args.compute_budget is not None:
        budget = next(
            row
            for row in json.loads(args.compute_budget.read_text())["rows"]
            if row["variant"] == args.variant
            and row["width"] == args.width
            and row.get("grad_accum_microbatches", 1) == args.grad_accum_microbatches
            and row.get("expert_axis_size", 1) == args.expert_axis_size
            and row["model_config"]["moe_ragged_chunks"] == args.expert_chunks
            and row.get("memory_scheduler", "default") == args.memory_scheduler
        )
        if budget["steps"] != args.steps or budget["model_config"] != json.loads(json.dumps(asdict(config.model))):
            raise ValueError("Execution configuration differs from frozen compute budget")
        if budget.get("batch_size", 16) != args.batch_size or budget.get("tpu_chips", 16) != args.tpu_chips:
            raise ValueError("Execution batch or hardware differs from frozen compute budget")
        if "optimizer_config" in budget and budget["optimizer_config"] != json.loads(
            json.dumps(asdict(config.optimizer))
        ):
            raise ValueError("Execution optimizer differs from frozen compute budget")
        if budget.get("execution_policy", "native-saved") != execution_policy:
            raise ValueError("Execution backward scheduling differs from frozen compute budget")
        if (budget.get("grad_accum_microbatches", 1), budget.get("expert_axis_size", 1)) != (
            args.grad_accum_microbatches,
            args.expert_axis_size,
        ):
            raise ValueError("Execution accumulation or expert mesh differs from frozen compute budget")
        persist("compute-budget.json", budget)
    persist(
        "execution-policy.json",
        {
            "backward": execution_policy,
            "additional_forward_recomputation": args.width >= 1024,
            "attn_res_layer_backward": config.model.attn_res_layer_backward,
            "attn_res_remat_attention": config.model.attn_res_remat_attention,
            "xla_flags": flags,
            "libtpu_init_args": tpu_flags,
            "grad_accum_microbatches": args.grad_accum_microbatches,
            "physical_batch_size": args.batch_size // args.grad_accum_microbatches,
            "expert_axis_size": args.expert_axis_size,
            "expert_chunks": args.expert_chunks,
            "dropless": True,
            "capacity_factor": config.model.capacity_factor,
        },
    )
    tables = None
    if plus_variant:
        tables = case_classes.build(Path(args.tokenizer) / "tokenizer.json", config.model.vocab_size)
        persist("case-classes.json", case_classes.summary(*tables))
    head_check = validate_head("plus" if plus_variant else args.variant, args.width)
    persist("head-preflight.json", head_check)
    if args.variant == "plus-frame-kda":
        frame_check = validate_frame_kda(config.model.max_seq_len)
        persist("frame-kda-preflight.json", frame_check)
    if args.variant == "plus-stack4-kda":
        persist("stack-kda-preflight.json", validate_stack_kda(config.model.max_seq_len))
    if args.batch_size in (32, 64, 128):
        histogram_check = validate_histogram(args.batch_size * 8192 // args.tpu_chips)
        persist("histogram-preflight.json", histogram_check)
        StoragePath(artifact_root + f"/histogram-preflight-host{jax.process_index()}.json").write_bytes(
            json.dumps(histogram_check, indent=2).encode()
        )
    config.trainer.trainer.distributed = replace(config.trainer.trainer.distributed, initialize_jax_distributed=False)
    handlers = []
    memory_stages = []

    def record_startup_memory(stage):
        devices = []
        for device in jax.local_devices():
            stats = device.memory_stats()
            devices.append(
                {
                    "device_id": device.id,
                    "in_use_gib": stats["bytes_in_use"] / 1024**3,
                    "peak_gib": stats["peak_bytes_in_use"] / 1024**3,
                    "limit_gib": stats["bytes_limit"] / 1024**3,
                }
            )
        memory_stages.append({"stage": stage, "time": time.time(), "devices": devices})
        StoragePath(artifact_root + f"/startup-memory-host{jax.process_index()}.json").write_bytes(
            json.dumps({"process_index": jax.process_index(), "stages": memory_stages}, indent=2).encode()
        )

    initialize = config.trainer.trainer.initialize

    def checked_initialize():
        initialize()
        if len(jax.devices()) != args.tpu_chips or any(d.device_kind != "TPU v4" for d in jax.devices()):
            raise ValueError(f"This run requires TPU v4 with {args.tpu_chips} physical chips")
        if jax.process_index() == 0:
            handler = MetricsFileHandler(args.output / "native-metrics.jsonl", artifact_root + "/native-metrics.jsonl")
            handlers.append(handler)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger("levanter.json_logger").addHandler(handler)
            persist("resolved-config.json", asdict(config))
        record_startup_memory("before_state_initialization")

    config.trainer.trainer.initialize = checked_initialize
    evaluator = train.TaggedEvaluator

    def bounded_evaluator(**kwargs):
        kwargs["max_examples_per_dataset"] = 2048
        native_loss = kwargs["loss_fn"]

        def evaluate(params, batch):
            if isinstance(batch, LmExample):
                batch = grug_lm_example_from_named(batch)

            def microbatch(index):
                chunk = jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, index * args.tpu_chips, args.tpu_chips, axis=0), batch
                )
                return native_loss(params, chunk)

            values = jax.lax.map(microbatch, jnp.arange(batch.tokens.shape[0] // args.tpu_chips))
            return jax.tree.map(
                lambda x: jax.lax.reshape(
                    x, (x.shape[0] * x.shape[1], *x.shape[2:]), out_sharding=P(model._BATCH_AXES, None)
                ),
                values,
            )

        kwargs["loss_fn"] = evaluate
        return evaluator(**kwargs)

    native_flat = train._flat_train_step

    def flat_step(step_fn, treedef):
        native = native_flat(step_fn, treedef)
        completed = 0
        started = None
        names = [] if args.variant == "belay" else ["cp_query", "cp_key", "cp_gate", "cp_sink"]
        if plus_variant:
            names += ["cp_prev_key", "cp_prev_gain", "cp_variant", "cp_variant_bias"]
        if args.variant == "plus-frame-kda":
            names += ["frame_gain"]

        def gradient_checks(loss, grads):
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]))
            norms = jnp.array([jnp.linalg.norm(getattr(grads, k)) for k in names])
            if args.variant == "plus-stack4-kda":
                action_norms = []
                for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
                    if any(isinstance(key, jax.tree_util.GetAttrKey) and key.name == "w_stack_actions" for key in path):
                        action_norms.append(jnp.linalg.norm(leaf.reshape(-1, *leaf.shape[-2:]), axis=(-2, -1)))
                if sum(x.size for x in action_norms) != 4:
                    raise ValueError("Expected action gradients in all four native KDA layers")
                norms = jnp.concatenate((norms, *action_norms))
            positive = jnp.concatenate((norms[:4], norms[5:])) if plus_variant else norms
            good = finite & jnp.isfinite(loss) & jnp.all(jnp.isfinite(positive) & (positive > 0))
            return good, finite, norms

        @jax.jit
        def state_is_finite(leaves):
            return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))

        @partial(jax.jit, static_argnames=("loop_active", "router_tie_active"))
        def preflight(leaves, batch, loop_active=None, router_tie_active=None):
            state = jax.tree.unflatten(treedef, leaves)
            params = train._apply_qb_betas(state.params, state.pending_qb_betas)
            if config.trainer.grad_accum_microbatches > 1:
                (loss, _), grads = train._accumulated_loss_and_grads(
                    config.trainer.grad_accum_microbatches,
                    params,
                    batch,
                    config.trainer.trainer.mp,
                    None,
                    state.step,
                    loop_active,
                    router_tie_active,
                )
            else:
                (loss, _), grads = train._loss_and_grads(
                    params,
                    batch,
                    config.trainer.trainer.mp,
                    None,
                    state.step,
                    loop_active,
                    router_tie_active=router_tie_active,
                )
            _, finite, norms = gradient_checks(loss, grads)
            return loss, finite, norms

        def run(leaves, batch, **kwargs):
            nonlocal completed, started
            if completed == 0:
                jax.block_until_ready(leaves)
                record_startup_memory("state_ready_before_preflight")
                state = jax.tree.unflatten(treedef, leaves)
                restored_step = int(jax.device_get(state.step))
                if jax.process_index() == 0:
                    if wandb.run is None or wandb.run.offline or wandb.run.disabled:
                        raise RuntimeError("Online W&B must initialize before optimizer updates")
                    wandb.config.update(
                        {
                            "seed": 0,
                            "restored_step": restored_step,
                            "checkpoint_root": args.checkpoint_path,
                            "variant": args.variant,
                            "width": args.width,
                            "logical_batch_size": args.batch_size,
                            "physical_batch_size": args.batch_size // args.grad_accum_microbatches,
                            "grad_accum_microbatches": config.trainer.grad_accum_microbatches,
                            "backward_scheduling": execution_policy,
                            "process_group_id": attempt,
                            "iris_attempt_uid": os.environ.get("IRIS_ATTEMPT_UID"),
                            "compute_budget": (
                                None
                                if budget is None
                                else {k: v for k, v in budget.items() if k not in ("model_config", "parameter_count")}
                            ),
                        }
                    )
                    if args.variant == "plus-frame-kda":
                        contract = {
                            "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/59",
                            "motivating_task_url": "https://github.com/WhenWen/astra-d8-research/issues/58",
                            "synthetic_task_evaluated": False,
                            "synthetic_gate_waiver": "User explicitly requested direct native LM testing",
                            "baseline_run_id": "4434036161f49ae783d3",
                            "comparison": "equal tokens; additional branch arithmetic is reported separately",
                            "additional_parameters": 313872,
                            "frame_kda": asdict(FRAME_CONFIG),
                            "data_manifest_sha256": hashlib.sha256(args.data_config.read_bytes()).hexdigest(),
                            "tokenizer_files_sha256": identities,
                        }
                        wandb.config.update(contract)
                        persist("frame-kda-contract.json", contract)
                    persist("tracking.json", {"url": wandb.run.url, "resumed": False, "restored_step": restored_step})
                    if args.variant == "plus-stack4-kda":
                        contract = {
                            "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/59",
                            "motivating_task_url": "https://github.com/WhenWen/astra-d8-research/issues/58",
                            "synthetic_task_evaluated": False,
                            "synthetic_gate_waiver": "User explicitly requested direct native LM testing",
                            "baseline_run_id": "4434036161f49ae783d3",
                            "comparison": "equal tokens; per-layer depth-four KDA replaces the native recurrence",
                            "additional_parameters": 32832,
                            "depth": 4,
                            "kernel_source_sha256": (
                                hashlib.sha256(
                                    Path(__file__).parent.parent.joinpath("moe/stack_kda.py").read_bytes()
                                ).hexdigest()
                            ),
                            "read": "top after update",
                            "action_initial_bias": [0, 0, 6, 0],
                            "data_manifest_sha256": hashlib.sha256(args.data_config.read_bytes()).hexdigest(),
                            "tokenizer_files_sha256": identities,
                        }
                        wandb.config.update(contract)
                        persist("stack-kda-contract.json", contract)
                if not memory_schedule:
                    loss, finite, norms = jax.device_get(preflight(leaves, batch, **kwargs))
                    record_startup_memory("after_full_model_preflight")
                    # A zero predecessor gain intentionally blocks W_prev's first gradient.
                    positive = np.delete(norms, 4) if plus_variant else norms
                    if not (bool(finite) and np.isfinite(loss) and np.all(positive > 0)):
                        raise ValueError("Full-model preflight failed before optimizer updates")
                    if jax.process_index() == 0:
                        persist(
                            "full-model-preflight.json",
                            {
                                "loss": float(loss),
                                "all_gradients_finite": True,
                                "copy_gradient_norms": norms[: len(names)].tolist(),
                                "stack_action_gradient_norms": norms[len(names) :].tolist(),
                                "training_updates": 0,
                                "validated_before_optimizer": True,
                                "method": "separate-gradient-preflight",
                                "restored_step": restored_step,
                            },
                        )
                preflight.clear_cache()
            result = native(leaves, batch, **kwargs)
            completed += 1
            if completed == 1:
                jax.block_until_ready(result)
                record_startup_memory("after_first_optimizer_step")
                if memory_schedule:
                    loss = float(jax.device_get(result[1]["train/loss"]))
                    finite = bool(jax.device_get(state_is_finite(result[0])))
                    dropped = int(np.asarray(jax.device_get(result[1]["moe/dropped_assignments"])).sum())
                    if dropped:
                        raise ValueError(f"Dropless execution lost {dropped} expert assignments")
                    if not (np.isfinite(loss) and finite):
                        raise ValueError("Nonfinite loss or state after the first native optimizer update")
                    persist(
                        "first-update-validation.json",
                        {
                            "loss": loss,
                            "all_state_finite": finite,
                            "dropped_assignments": dropped,
                            "training_updates": 1,
                            "restored_step": restored_step,
                        },
                    )
                    state_is_finite.clear_cache()
                    retain_compiler_dump()
            if args.calibration and completed in (100, 400):
                jax.block_until_ready(result)
                sync_global_devices(f"belay-window-{completed}")
                if completed == 100:
                    started = time.monotonic()
                else:
                    elapsed = float(np.asarray(process_allgather(np.array(time.monotonic() - started))).max())
                    if jax.process_index() == 0:
                        (args.output / "training-window.json").write_text(
                            json.dumps(
                                {
                                    "first_update": 100,
                                    "last_update": 399,
                                    "updates": 300,
                                    "tokens": 39321600,
                                    "max_host_seconds": elapsed,
                                    "tokens_per_second": 39321600 / elapsed,
                                }
                            )
                        )
            return result

        run.lower = native.lower
        return run

    try:
        with (
            execution(args.variant, tables),
            python_zarr_checkpoint_writes(),
            patch.object(train, "TaggedEvaluator", bounded_evaluator),
            patch.object(train, "_flat_train_step", flat_step),
        ):
            train._run_grug_local(config)
    finally:
        retain_compiler_dump()
        for handler in handlers:
            handler.upload()
        if jax.process_index() == 0:
            for path in args.output.glob("*.json"):
                StoragePath(artifact_root + "/" + path.name).write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
