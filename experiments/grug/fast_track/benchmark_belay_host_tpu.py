# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure host scheduling in the real-data Belay training loop."""

import argparse
import atexit
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter import checkpoint
from levanter.distributed import _finalize_iris_jax_after_clean_exit, _unregister_iris_exit_handler
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.fast_track import belay_tpu


def validate_broadcast() -> None:
    """Check every boolean pair from every source through both input paths."""
    for source in range(jax.process_count()):
        for bits in range(4):
            expected = np.array([bool(bits & 1), bool(bits & 2)], dtype=np.bool_)
            local = expected if jax.process_index() == source else ~expected
            for value in (local, jnp.asarray(local)):
                actual = checkpoint.broadcast_one_to_all(value, is_source=jax.process_index() == source)
                np.testing.assert_array_equal(actual, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("checkpoint", "pipeline"), default="checkpoint")
    parser.add_argument("--arm", choices=("control-before", "candidate", "control-after"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--job-completion", choices=("deferred", "normal"), required=True)
    args, remaining = parser.parse_known_args()
    updates, warmup_updates = (100, 10) if args.experiment == "checkpoint" else (600, 25)
    if not args.report_prefix.startswith(belay_tpu.REGION_PREFIX):
        parser.error("Reports must stay in us-central2")

    original_build = belay_tpu.build_config
    original_run = belay_tpu._run_grug_local
    original_on_step = checkpoint.Checkpointer.on_step
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "arm": args.arm,
        "experiment": args.experiment,
        "revision": args.revision,
        "status": "starting",
        "decisions": [],
    }
    previous_end = None
    window_start = None

    def build_config(**kwargs):
        config = original_build(**kwargs)
        if kwargs["execution_profile"] != "selected" or kwargs["seed"] != 0 or kwargs["resume"]:
            raise ValueError("Use selected execution, seed 0 and a fresh run")
        trainer = replace(config.trainer, completion_marker_path=f"{kwargs['checkpoint_path']}/training_done.json")
        if args.experiment == "pipeline":
            trainer = replace(trainer, pipeline_depth=int(args.arm == "candidate"))
        report["pipeline_depth"] = trainer.pipeline_depth
        return replace(config, stop_after_steps=updates, eval=None, trainer=trainer)

    def on_step(self, *, tree, step, force=False):
        nonlocal previous_end, window_start
        started = time.perf_counter()
        original_on_step(self, tree=tree, step=step, force=force)
        ended = time.perf_counter()
        row: dict[str, Any] = {
            "step": int(step),
            "force": force,
            "decision_seconds": ended - started,
            "completed_at_monotonic": ended,
        }
        if not force:
            row["cycle_seconds"] = None if previous_end is None else ended - previous_end
            previous_end = ended
            if step == warmup_updates:
                jax.block_until_ready(tree)
                window_start = time.perf_counter()
                report["window_start_monotonic"] = window_start
                previous_end = window_start
            elif step == updates:
                jax.block_until_ready(tree)
                assert window_start is not None
                report["drained_window_seconds"] = time.perf_counter() - window_start
                report["drained_window_updates"] = updates - warmup_updates
        report["decisions"].append(row)
        with (output / "checkpoint-timings.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if force:
            leaves = [leaf for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf)]
            finite = jax.jit(lambda xs: jnp.stack([jnp.all(jnp.isfinite(x)) for x in xs]).all())(leaves)
            report["state_leaves"] = len(leaves)
            report["finite_state"] = bool(finite)
            if not report["finite_state"]:
                raise RuntimeError("Nonfinite parameter, optimizer or EMA state")

    def run_training(config):
        rank = jax.process_index()
        report.update(rank=rank, status="preflight", started_at=time.time())
        artifact = StoragePath(f"{args.report_prefix}/attempts/{os.environ['IRIS_ATTEMPT_UID']}/rank-{rank}")
        try:
            validate_broadcast()
            report["broadcast_cases"] = jax.process_count() * 4 * 2
            checkpoint.Checkpointer.on_step = on_step
            report["status"] = "training"
            # Checkpointer uses jnp only to construct the two decision flags.
            # Keep CPU placement confined to this diagnostic; pipeline arms use the default.
            flag_scope = (
                patch.object(checkpoint, "jnp", np)
                if args.experiment == "checkpoint" and args.arm == "candidate"
                else nullcontext()
            )
            with flag_scope:
                original_run(config)
            report["status"] = "completed"
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            if args.job_completion == "deferred":
                _unregister_iris_exit_handler()
                atexit.register(_finalize_iris_jax_after_clean_exit)
            checkpoint.Checkpointer.on_step = original_on_step
            report["finished_at"] = time.time()
            payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
            (output / "host-timing.json").write_text(payload)
            (artifact / "host-timing.json").write_text(payload)
            timings = output / "checkpoint-timings.jsonl"
            if timings.exists():
                (artifact / timings.name).write_bytes(timings.read_bytes())

    belay_tpu.build_config = build_config
    belay_tpu._run_grug_local = run_training
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        belay_tpu.main()
    finally:
        sys.argv = old_argv
        belay_tpu.build_config = original_build
        belay_tpu._run_grug_local = original_run


if __name__ == "__main__":
    main()
