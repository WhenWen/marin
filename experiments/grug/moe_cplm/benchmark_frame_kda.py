# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare parallel schedules for the same frame-KDA recurrence without training."""

import argparse
import hashlib
import json
import os
import statistics
import uuid
from pathlib import Path

import jax
import jaxlib
import numpy as np
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_cplm.frame_kda_validation import validate_frame_kda


def write_record(output: Path, artifact_root: str, name: str, value):
    content = (json.dumps(value, indent=2) + "\n").encode()
    (output / name).write_bytes(content)
    StoragePath(artifact_root + "/" + name).write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 16, 32, 128])
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    source = Path(__file__).parent
    jax_settings = json.loads((source / "belay_recipe.json").read_text())["trainer"]["trainer"]["jax_config"]
    jax_settings["jax_enable_pgle"] = False
    for name, value in jax_settings.items():
        jax.config.update(name, value)
    precision_flag = "--xla_allow_excess_precision=false"
    if precision_flag not in os.environ.get("XLA_FLAGS", "").split():
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " " + precision_flag
    if any(device.device_kind != "TPU v4" for device in jax.local_devices()):
        raise ValueError("Performance qualification requires TPU v4")
    attempt = uuid.uuid4().hex
    output = args.output / attempt
    artifact_root = args.artifact_root.rstrip("/") + "/" + attempt
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "training_updates": 0,
        "execution_id": attempt,
        "artifact_root": artifact_root,
        "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/59",
        "motivating_task": "scoped_binding@1 (#58), not evaluated",
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_settings": jax_settings,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "length": args.length,
        "chunks": args.chunks,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("frame_kda.py", "frame_kda_validation.py", "benchmark_frame_kda.py", "belay_recipe.json")
        },
    }
    write_record(output, artifact_root, "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    summaries = []
    input_hashes = set()
    objectives = []
    for chunk in args.chunks:
        result = validate_frame_kda(args.length, chunk=chunk)
        write_record(output, artifact_root, f"chunk-{chunk}.json", result)
        input_hashes.add(result["native_inputs_sha256"])
        if len(input_hashes) != 1:
            raise ValueError("Parallel schedules received different benchmark inputs")
        objectives.append(result["native_objective"])
        np.testing.assert_allclose(objectives[-1], objectives[0], atol=1e-5, rtol=1e-4)
        summaries.append(
            {
                "chunk": chunk,
                "median_seconds": statistics.median(result["forward_backward_seconds"]),
                "compiled_temp_bytes": result["compiled_temp_bytes"],
                "compile_seconds": result["compile_seconds"],
            }
        )
        print(json.dumps(summaries[-1]), flush=True)
        write_record(output, artifact_root, "summary.json", summaries)


if __name__ == "__main__":
    main()
