# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify full KMA continuation using three fresh, online-tracked worker groups."""

import argparse
import json
import subprocess
import sys

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


def main():
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--optimizer", choices=("muown", "muon_md"), required=True)
    args = parser.parse_args()
    root = args.artifact_root.rstrip("/")
    phases = [("continuous", "continuous", 4), ("prefix", "interrupted", 2), ("restored", "interrupted", 4)]
    for phase, experiment, stop in phases:
        receipt = StoragePath(root) / "phases" / f"{phase}.json"
        if receipt.exists():
            if not json.loads(receipt.read_bytes())["passed"]:
                raise RuntimeError(f"Failed qualification phase {phase}; manual investigation required")
            continue
        common = [
            sys.executable,
            "-m",
            "iris.hooks.multigpu_main",
            "--nproc",
            "8",
            "--",
            sys.executable,
            "-m",
            "experiments.grug.muon_mup_tpu.h100_worker",
            "--size",
            "d768",
            "--h100",
            "--ladder16k",
            "--tokenizer",
            "tokenizer-hero-16k",
            "--execution",
            "reference",
            "--steps",
            "6930",
            "--batch-size",
            "128",
            "--seed",
            "0",
            "--optimizer",
            args.optimizer,
            "--lr-mult",
            "1",
            "--stop-after-steps",
            str(stop),
            "--wandb-group",
            "kma-muown-md-b128-qualification-20261114",
            "--run-name",
            args.run_name + "-" + experiment,
            "--artifact-root",
            root + "/" + experiment,
        ]
        if phase == "continuous":
            subprocess.run([*common[6:], "--logging-check"], check=True)
        if phase == "restored":
            common += ["--compare-checkpoint", root + "/continuous/checkpoints"]
        result = subprocess.run(common, check=False)
        receipt.write_bytes(
            json.dumps({"phase": phase, "exit_code": result.returncode, "passed": result.returncode == 0}).encode()
        )
        if result.returncode:
            raise RuntimeError(f"Qualification {phase} failed with exit {result.returncode}")
    (StoragePath(root) / "qualification-complete.json").write_bytes(
        json.dumps(
            {
                "passed": True,
                "optimizer": args.optimizer,
                "steps": 4,
                "interruption_step": 2,
                "batch_size": 128,
                "context": 4096,
                "schedule_horizon": 6930,
                "parity": "all native model and optimizer state arrays compared exactly in restored worker",
            }
        ).encode()
    )


if __name__ == "__main__":
    main()
