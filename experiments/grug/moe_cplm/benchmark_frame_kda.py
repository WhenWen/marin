# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare parallel schedules for the same frame-KDA recurrence without training."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import jax
import jaxlib

from experiments.grug.moe_cplm.frame_kda_validation import validate_frame_kda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 16, 32, 128])
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(device.device_kind != "TPU v4" for device in jax.local_devices()):
        raise ValueError("Performance qualification requires TPU v4")
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    manifest = {
        "training_updates": 0,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "length": args.length,
        "chunks": args.chunks,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("frame_kda.py", "frame_kda_validation.py", "benchmark_frame_kda.py")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = []
    input_hashes = set()
    for chunk in args.chunks:
        result = validate_frame_kda(args.length, chunk=chunk)
        (args.output / f"chunk-{chunk}.json").write_text(json.dumps(result, indent=2) + "\n")
        input_hashes.add(result["native_inputs_sha256"])
        if len(input_hashes) != 1:
            raise ValueError("Parallel schedules received different benchmark inputs")
        summaries.append(
            {
                "chunk": chunk,
                "median_seconds": statistics.median(result["forward_backward_seconds"]),
                "compiled_temp_bytes": result["compiled_temp_bytes"],
                "compile_seconds": result["compile_seconds"],
            }
        )
        print(json.dumps(summaries[-1]), flush=True)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
