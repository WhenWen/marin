# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the H100 comparison without completing its enclosing allocation."""

import atexit

import levanter.distributed
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.train import _apply_runtime_defaults
from experiments.grug.muon_mup_tpu.train import main


def run():
    configure_coreweave_s3()
    _apply_runtime_defaults(inline_watch_enabled=False)
    try:
        main()
    finally:
        # The outer holder owns the allocation. Keep normal JAX shutdown, but
        # remove rank zero's callback that would complete the whole Iris job.
        finalize = levanter.distributed._finalize_iris_jax_after_clean_exit
        atexit.unregister(finalize)
        atexit.register(finalize)


if __name__ == "__main__":
    run()
