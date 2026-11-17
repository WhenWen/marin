# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install the locked GPU runtime, then run a single configuration and exit."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    subprocess.run(["uv", "sync", "--frozen", "--package", "marin-levanter", "--extra", "gpu", "--no-dev"], check=True)
    python = str(Path(os.environ.get("IRIS_VENV", ".venv")) / "bin" / "python")
    os.execv(python, [python, "-u", "-m", "experiments.hybrid_transfer.train", *sys.argv[1:]])


if __name__ == "__main__":
    main()
