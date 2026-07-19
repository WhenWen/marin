# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared launch wiring for the July-baseline Identity-HC variant.

Concrete profile and Gate 1 entry points live in ``profile_comparison.py`` and
``gate1.py`` so an accidental invocation of this module cannot launch a stale
demo cell.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import BlockShuffleConfig, LmDataConfig, TextLmDatasetFormat
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import lm_data_config
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_validation_sets
from experiments.grug.moe_identity_hyperconnection.model import GrugModelConfig
from experiments.grug.moe_identity_hyperconnection.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    run_grug,
)
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import nemotron_mix
from experiments.tokenization import default_tokenize


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    # Mesh size along the "expert" axis (expert-parallelism). 1 = no EP.
    expert_parallel: int = 1
    checkpointer: CheckpointerConfig | None = None
    """Override the checkpointer. None builds the default (periodic + final saves
    under output_path). Throughput experiments point this at node-local disk so a
    slow object-store commit can't wedge the end-of-run barrier."""


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def env_int(key: str, default: int) -> int:
    """Read an int from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def slimpajama_6b_data() -> LmDataConfig:
    """SlimPajama-6B, llama3-tokenized with block-shuffle, re-tokenized on first run.

    A small, R2-local corpus for GPU smoke/scale runs; returns a ready-to-train
    ``LmDataConfig``. A production pretraining mixture would instead need its
    tokenized cache already materialized to avoid a cross-region tokenize.
    """
    tokenize_step = default_tokenize(
        name="slimpajama-6b-cw",
        dataset="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        format=TextLmDatasetFormat(),
    )
    tokenize_step = dataclasses.replace(
        tokenize_step,
        config=dataclasses.replace(
            tokenize_step.config,
            # SlimPajama-6B tokenization OOMs at the default 10g worker_resources.
            worker_resources=ResourceConfig(ram="64g", disk="64g"),
        ),
    )
    return lm_data_config(
        training_set=tokenize_step,
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel"),
    )


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    # Map template launch knobs onto full Levanter TrainerConfig.
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=config.checkpointer
        or CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=None,
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer, expert_axis_size=config.expert_parallel)

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)
