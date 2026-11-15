# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run-level checkpoint compatibility and fresh-attempt lineage for KMA sweeps."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jax
import levanter.tracker
import numpy as np
from jax.experimental.multihost_utils import broadcast_one_to_all, sync_global_devices
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint


def recovery_contract(config) -> dict:
    value = asdict(config)
    value.pop("stop_after_steps", None)
    trainer = value["trainer"]["trainer"]
    for key in ("id", "tracker", "log_dir", "load_checkpoint", "load_checkpoint_path", "checkpointer"):
        trainer.pop(key, None)
    # The source is shipped without .git on workers. Hash the actual implementation.
    root = Path(__file__).parents[3]
    paths = sorted((root / "experiments/grug/fast_track").glob("*.py"))
    paths += sorted(Path(__file__).parent.glob("*.py"))
    paths += [root / "experiments/grug/checkpointing.py"]
    source = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    payload = {"config": value, "optimizer_type": type(config.optimizer).__qualname__, "source": source}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return {"sha256": hashlib.sha256(serialized.encode()).hexdigest(), "payload": json.loads(serialized)}


def prepare_recovery(config, artifact_root: str, attempt: str) -> bool:
    """An existing contract requires restoration, including a step-zero checkpoint.

    Publish the contract before initialization. If preemption interrupts the very
    first save, retry fails visibly instead of silently starting another run.
    """
    contract = recovery_contract(config)
    path = StoragePath(artifact_root) / "recovery-contract.json"
    existed = bool(broadcast_one_to_all(np.array(path.exists() if jax.process_index() == 0 else False)))
    if existed:
        previous = json.loads(path.read_bytes())
        if previous["sha256"] != contract["sha256"]:
            raise ValueError("Checkpoint configuration/source mismatch; refusing to restart or partially restore")
    elif jax.process_index() == 0:
        path.write_bytes(json.dumps({**contract, "initial_attempt": attempt}, indent=2).encode())
    sync_global_devices("kma-recovery-contract")
    return existed


def restore_or_initialize(state, *, config, persist, **kwargs):
    """Restore all native state fields, or commit the initial state before updates."""
    restored = restore_grug_state_from_checkpoint(state, **kwargs)
    resumed = kwargs["load_checkpoint_setting"] is True
    if not resumed:
        checkpointer = config.trainer.trainer.checkpointer.create(config.trainer.trainer.id)
        checkpointer.on_step(tree=restored, step=0, force=True)
        checkpointer.wait_until_finished()
    levanter.tracker.log_summary({"restored_optimizer_step": int(restored.step), "verified_checkpoint_restore": resumed})
    persist(
        "restoration.json",
        {
            "restored": resumed,
            "step": int(restored.step),
            "data_loader_start_step": int(restored.step),
            "checkpoint_search_paths": kwargs["checkpoint_search_paths"],
            "state_fields": list(restored.__dataclass_fields__),
            "randomness": {
                "policy": "no stochastic training layers; deterministic data indexed by restored step",
                "initialization_seed": config.trainer.trainer.seed,
                "data_seed": config.trainer.data_seed,
            },
            "wandb_semantics": "fresh run for this process; restored optimizer step logged explicitly",
        },
    )
    return restored
