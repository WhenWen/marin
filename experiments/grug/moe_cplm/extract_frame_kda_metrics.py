# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract final native scalar metrics near their storage, without moving histograms."""

import argparse
import hashlib
import json
import math
import uuid
from pathlib import Path

import fsspec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    args = parser.parse_args()
    destination = args.output_root.rstrip("/") + "/" + uuid.uuid4().hex
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    train_steps, eval_steps = set(), set()
    records = 0
    with fsspec.open(args.input, "rb") as source:
        before = source.fs.info(args.input)
        with fsspec.open(destination + "/metrics.jsonl", "wb", auto_mkdir=True) as output:
            for line in source:
                source_hash.update(line)
                row = json.loads(line)
                step, metrics = row.get("step"), row.get("metrics", {})
                selected = {
                    k: v
                    for k, v in metrics.items()
                    if (
                        k == "train/loss"
                        or (k.startswith("eval/") and "loss" in k)
                        or "throughput" in k
                        or k.startswith("moe/")
                    )
                }
                if step is None or not selected:
                    continue
                for key in ("train/loss", "eval/loss"):
                    if key in selected and not math.isfinite(selected[key]):
                        raise ValueError(f"Nonfinite {key} at step {step}")
                if "train/loss" in selected:
                    train_steps.add(step)
                if "eval/loss" in selected:
                    eval_steps.add(step)
                encoded = (json.dumps({"step": step, "metrics": selected}) + "\n").encode()
                output.write(encoded)
                output_hash.update(encoded)
                records += 1
        source.fs.invalidate_cache(args.input)
        after = source.fs.info(args.input)
    for key in ("size", "generation", "etag", "mtime"):
        if before.get(key) != after.get(key):
            raise ValueError(f"Source changed during extraction: {key}")
    if train_steps != set(range(args.expected_updates)):
        raise ValueError("Training observations do not cover the complete requested schedule")
    expected_eval = set(range(999, args.expected_updates, 1000)) | {args.expected_updates - 1}
    if eval_steps != expected_eval:
        raise ValueError("Validation observations do not cover the scheduled and final evaluations")
    manifest = {
        "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/59",
        "motivating_task": "scoped_binding@1 (#58), not evaluated",
        "training_updates_performed": 0,
        "source": args.input,
        "source_metadata": {k: before.get(k) for k in ("size", "generation", "etag")},
        "source_sha256": source_hash.hexdigest(),
        "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output": destination + "/metrics.jsonl",
        "output_sha256": output_hash.hexdigest(),
        "records": records,
        "training_observations": len(train_steps),
        "validation_steps": sorted(eval_steps),
        "complete": True,
    }
    with fsspec.open(destination + "/manifest.json", "w", auto_mkdir=True) as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
