# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain an Iris allocation after worker exit, or release on success when requested."""

import argparse
import datetime
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


def run():
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--release-on-success", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a worker command is required")
    receipt = {
        "holder_id": uuid.uuid4().hex,
        "task_attempt": os.environ.get("IRIS_TASK_ID"),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    try:
        receipt["worker_exit_code"] = subprocess.run(command, check=False).returncode
    except Exception as exc:
        receipt["worker_error"] = type(exc).__name__
    receipt["holding_since_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    release = args.release_on_success and receipt.get("worker_exit_code") == 0
    receipt["state"] = "completed_allocation_releasing" if release else "idle_allocation_retained"
    content = json.dumps(receipt, indent=2)
    local = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp")) / "allocation-holder.json"
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(content)
    except OSError as exc:
        print(f"Local holder receipt failed: {type(exc).__name__}; allocation retained", flush=True)
    print(content, flush=True)
    # A failed artifact upload must never release the allocation. The local
    # receipt and task logs remain evidence even when object storage is down.
    while True:
        try:
            (StoragePath(args.artifact_root) / "holders" / (receipt["holder_id"] + ".json")).write_text(content)
            break
        except Exception as exc:
            print(f"Holder receipt upload failed: {type(exc).__name__}; allocation retained", flush=True)
            time.sleep(30)
    if release:
        return
    while True:
        time.sleep(60)


if __name__ == "__main__":
    run()
