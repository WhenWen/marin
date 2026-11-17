# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict full-state recovery and immutable attempt lineage for the transfer sweep."""

import dataclasses
import hashlib
import json
from datetime import UTC, datetime

import draccus
import fsspec
from levanter.checkpoint import discover_checkpoint_candidates
from levanter.main.train_lm import TrainLmConfig


def scientific_identity(resolved: dict) -> dict:
    # Exclude only operational Trainer settings; retain the entire model, data,
    # optimizer and loss configuration (including the schedule's total length).
    trainer_fields = (
        "seed",
        "mp",
        "train_batch_size",
        "per_device_parallelism",
        "num_train_steps",
        "mesh",
        "field_dtype_overrides",
        "quantization",
        "model_averaging",
        "allow_nondivisible_batch_size",
        "batch_axis",
        "parameter_axis_mapping",
    )
    identity = {k: v for k, v in resolved.items() if k not in ("trainer", "git_commit", "git_dirty", "git_tree_hash")}
    identity["trainer"] = {k: resolved["trainer"][k] for k in trainer_fields if k in resolved["trainer"]}
    return identity


def fingerprint(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: str, value: dict) -> None:
    fs, plain_path = fsspec.core.url_to_fs(path)
    fs.makedirs(plain_path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(plain_path, "w") as stream:
        json.dump(value, stream, indent=2)


def prepare_recovery(config: TrainLmConfig, legacy: dict) -> tuple[TrainLmConfig, dict]:
    """Select the latest complete compatible checkpoint; never fall back after failure.

    Identity files are published before training. An earlier attempt with no
    complete checkpoint therefore stops recovery instead of silently starting over.
    Old attempts are read-only and each new process writes to its own directory.
    """
    identity = scientific_identity(draccus.encode(config))
    digest = fingerprint(identity)
    root = config.trainer.checkpointer.base_path
    fs, plain_root = fsspec.core.url_to_fs(root)
    attempts = fs.glob(f"{plain_root}/*/identity.json")
    roots = []
    for path in attempts:
        with fs.open(path) as stream:
            prior = json.load(stream)
        if prior["fingerprint"] != digest or prior["scientific_config"] != identity:
            raise RuntimeError(f"Incompatible checkpoint identity: {path}")
        roots.append(fs.unstrip_protocol(path.rsplit("/", 1)[0]))

    minimum_step = -1
    if legacy:
        if legacy["fingerprint"] != digest:
            raise RuntimeError("Legacy checkpoint configuration differs from the requested scientific configuration")
        roots.extend(legacy["checkpoint_roots"])
        minimum_step = legacy["minimum_checkpoint_step"]
    candidates = discover_checkpoint_candidates(*roots) if roots else []
    if (attempts or minimum_step >= 0) and not candidates:
        raise RuntimeError(
            "A previous attempt exists but no complete checkpoint can be restored; refusing fresh training"
        )
    selected = candidates[-1] if candidates else None
    if selected and selected.step < minimum_step:
        raise RuntimeError("Checkpoint discovery lost previously verified progress; refusing rollback")
    if selected and selected.step >= config.trainer.num_train_steps:
        raise RuntimeError("Checkpoint step exceeds this experiment's schedule")
    trainer = dataclasses.replace(
        config.trainer,
        load_checkpoint=selected is not None,
        load_checkpoint_path=selected.path if selected else None,
        allow_partial_checkpoint=False,
    )
    # Levanter metadata stores the last completed, zero-indexed step. The full
    # TrainerState stores the NEXT step and train_lm seeks its loader to that step.
    lineage = {
        "run_id": trainer.id,
        "created_at": datetime.now(UTC).isoformat(),
        "fingerprint": digest,
        "scientific_config": identity,
        "requested_checkpoint": selected.path if selected else None,
        "expected_next_step": selected.step + 1 if selected else 0,
        "restoration_status": "requested" if selected else "fresh",
        "wandb_url": f"https://wandb.ai/marin-community/Hyperball/runs/{trainer.id}",
    }
    write_json(f"{trainer.checkpointer.expanded_path(trainer.id)}/identity.json", lineage)
    return dataclasses.replace(config, trainer=trainer), lineage


def require_final_checkpoint(config: TrainLmConfig) -> str:
    root = config.trainer.checkpointer.expanded_path(config.trainer.id)
    candidates = discover_checkpoint_candidates(root)
    if not candidates or candidates[-1].step != config.trainer.num_train_steps - 1:
        raise RuntimeError("Training returned without a complete final checkpoint; refusing completion marker")
    return candidates[-1].path
