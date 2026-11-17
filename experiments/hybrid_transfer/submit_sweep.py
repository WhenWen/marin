# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Queue independent root jobs; Iris chooses H100 peers as capacity becomes free."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.hybrid_transfer.train import GROUP, RunSpec


def submit_one(index, spec, devices, log_dir):
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    name = f"hybrid-transfer-{index:03d}-{identity}"
    record = log_dir / f"{index:03d}.json"
    # A submission journal is a duplicate guard, including an interrupted CLI.
    if record.exists():
        return json.loads(record.read_text())
    command = [
        str(Path(sys.executable).with_name("iris")),
        "--cluster=marin",
        "job",
        "run",
        "--no-wait",
        "--enable-extra-resources",
        "--gpu",
        f"H100x{devices}",
        "--cpu",
        str(devices),
        "--memory",
        f"{32 * devices}GB",
        "--disk",
        "64GB",
        "--priority",
        "batch",
        "--no-sync",
        "--job-name",
        name,
    ]
    for key in ("WANDB_API_KEY", "CW_KEY_ID", "CW_KEY_SECRET"):
        command.extend(["--secret", key, f"env:{key}"])
    command.extend(
        [
            "--",
            "python",
            "-u",
            "-m",
            "experiments.hybrid_transfer.worker",
            "--spec",
            json.dumps(spec),
            "--data-prefix",
            "s3://marin-us-east-02a/marin",
            "--output-prefix",
            "s3://marin-us-east-02a/marin/experiments",
            "--devices",
            str(devices),
            "--microbatch",
            "4" if spec["residual"] == "full_attnres" else "8",
        ]
    )
    item = {
        "index": index,
        "name": name,
        "spec": spec,
        "config_name": RunSpec(**spec).name,
        "command": command,
        "status": "submission_started",
        "group": GROUP,
    }
    with record.open("x") as stream:
        json.dump(item, stream, indent=2)
    result = subprocess.run(command, text=True, capture_output=True)
    (log_dir / f"{index:03d}.log").write_text(result.stdout + result.stderr)
    item["returncode"] = result.returncode
    item["status"] = "submitted" if result.returncode == 0 else "submission_needs_inspection"
    item["job_id"] = result.stdout.strip() if result.returncode == 0 else None
    record.write_text(json.dumps(item, indent=2))
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--exclude-residual", action="append", default=[])
    args = parser.parse_args()
    evidence = json.loads(Path(__file__).with_name("resume_evidence.json").read_text())
    assert len({(r["method"], r["optimizer"]) for r in evidence}) == 8
    assert all(r["status"] == "passed_exact_full_state_equality" for r in evidence)
    specs = json.loads(Path(__file__).with_name("sweep.json").read_text())
    args.journal.mkdir(parents=True, exist_ok=True)
    # This is submission RPC concurrency, not a GPU/run concurrency cap.
    with ThreadPoolExecutor(max_workers=6) as pool:
        ordered = sorted(enumerate(specs), key=lambda pair: abs(math.log2(pair[1]["lr"] / 0.02)))
        futures = [
            pool.submit(submit_one, i, spec, args.devices, args.journal)
            for i, spec in ordered
            if spec["residual"] not in args.exclude_residual
        ]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({key: result[key] for key in ("index", "name", "status", "job_id")}), flush=True)


if __name__ == "__main__":
    main()
