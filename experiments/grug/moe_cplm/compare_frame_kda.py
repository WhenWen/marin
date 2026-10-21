# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare native metric logs at matched steps without smoothing or significance claims."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read_metrics(path, until):
    rows = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        step = record.get("step")
        if step is None or (until is not None and step > until):
            continue
        metrics = rows.setdefault(step, {})
        for name, value in record.get("metrics", {}).items():
            if name in metrics and metrics[name] != value:
                raise ValueError(f"Conflicting observations in {path} at step {step}: {name}")
            metrics[name] = value
    return rows


def compare(baseline, candidate, timing_start):
    validation = []
    for step in sorted(baseline.keys() | candidate.keys()):
        left = {k: v for k, v in baseline.get(step, {}).items() if k.startswith("eval/") and "loss" in k}
        right = {k: v for k, v in candidate.get(step, {}).items() if k.startswith("eval/") and "loss" in k}
        if not left and not right:
            continue
        validation.append(
            {
                "step": step,
                "missing_from_baseline": sorted(right.keys() - left.keys()),
                "missing_from_candidate": sorted(left.keys() - right.keys()),
                "metrics": {
                    k: {"baseline": left[k], "candidate": right[k], "difference": right[k] - left[k]}
                    for k in sorted(left.keys() & right.keys())
                },
            }
        )
    timing = "throughput/tokens_per_second"
    common = sorted(s for s in baseline.keys() & candidate.keys() if s >= timing_start)
    timed = [s for s in common if timing in baseline[s] and timing in candidate[s]]
    medians = {}
    if timed:
        medians = {
            name: statistics.median(rows[s][timing] for s in timed)
            for name, rows in (("baseline", baseline), ("candidate", candidate))
        }
    training = {}
    for name, rows in (("baseline", baseline), ("candidate", candidate)):
        steps = sorted(s for s, metrics in rows.items() if "train/loss" in metrics)
        training[name] = {
            "observations": len(steps),
            "last_step": steps[-1] if steps else None,
            "last_loss": rows[steps[-1]]["train/loss"] if steps else None,
            "missing_steps": sorted(set(range(steps[-1] + 1)) - set(steps)) if steps else [],
        }
    return {
        "training": training,
        "validation": validation,
        "throughput": {
            "metric": timing,
            "requested_start": timing_start,
            "matched_steps": timed,
            "median_tokens_per_second": medians,
            "candidate_to_baseline_ratio": medians["candidate"] / medians["baseline"] if timed else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--until", type=int)
    parser.add_argument("--timing-start", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    rows = {name: read_metrics(path, args.until) for name, path in paths.items()}
    result = compare(rows["baseline"], rows["candidate"], args.timing_start)
    result["until"] = args.until
    result["inputs"] = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
