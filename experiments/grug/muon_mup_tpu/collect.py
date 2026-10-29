# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect completed final evaluations, keeping incomplete attempts out of rankings."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import fsspec
import wandb


def collect_attempt(api, fs, path):
    with fs.open(path + "/identity.json") as stream:
        identity = json.load(stream)
    with fs.open(path + "/resolved-config.json") as stream:
        config = json.load(stream)
    run_path = identity["wandb_url"].removeprefix("https://wandb.ai/").replace("/runs/", "/")
    run = api.run(run_path)
    record = {**identity, "state": run.state, "eligible": False, "final_bpb": {}}
    record["observed_step"] = run.summary.get("global_step")
    record["observed_train_loss"] = run.summary.get("train/loss")
    trainer = config["trainer"]["trainer"]
    comparison = {
        "model": config["model"],
        "execution": identity.get("execution", "reference"),
        "data_sha256": identity["data_sha256"],
        "data_seed": config["trainer"]["data_seed"],
        "seed": identity["seed"],
        "steps": identity["steps"],
        "batch": trainer["train_batch_size"],
        "precision": trainer["mp"],
        "jax_config": trainer["jax_config"],
        "eval": config["eval"],
    }
    record["comparison_key"] = hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()
    record["optimizer"] = config["optimizer"]
    if run.state != "finished" or not fs.exists(path + "/complete.json"):
        record["ineligible_reason"] = "Training or final evaluation has not completed."
        return record
    with fs.open(path + "/complete.json") as stream:
        completed = json.load(stream)
    if completed != {"steps": identity["steps"], "final_evaluation_complete": True}:
        raise ValueError(f"Completion marker does not match requested training: {path}")
    domains = [name for name in config["data"]["components"] if name.startswith(("paloma/", "uncheatable_eval/"))]
    # Callback steps are zero-based: the final parameters are logged at steps - 1.
    final_step = identity["steps"] - 1
    keys = ["_step", *[f"eval/{name}/bpb" for name in domains]]
    rows = list(run.scan_history(keys=keys, min_step=final_step, max_step=final_step + 1))
    if len(rows) != 1 or rows[0]["_step"] != final_step:
        record["ineligible_reason"] = "Final per-domain BPB is absent from W&B history."
        return record
    values = {name: rows[0][f"eval/{name}/bpb"] for name in domains}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Nonfinite final BPB: {path}")
    record["final_bpb"] = values
    record["macro_bpb"] = {
        group: (
            sum(value for name, value in values.items() if name.startswith(group + "/"))
            / sum(name.startswith(group + "/") for name in values)
        )
        for group in ("paloma", "uncheatable_eval")
    }
    record["eligible"] = True
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = wandb.Api()
    records = []
    for root in args.roots:
        fs, path = fsspec.core.url_to_fs(root)
        for identity in sorted(fs.glob(path.rstrip("/") + "/*/identity.json")):
            records.append(collect_attempt(api, fs, identity.rsplit("/", 1)[0]))
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    print(
        f"Collected {len(records)} attempts; {sum(record['eligible'] for record in records)} eligible final evaluations."
    )


if __name__ == "__main__":
    main()
