# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the H100 comparison without completing its enclosing allocation."""

import atexit

import jax
import levanter.distributed
from jax.experimental import multihost_utils
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.train import _apply_runtime_defaults
from experiments.grug.muon_mup_tpu.train import main


def run():
    configure_coreweave_s3()
    _apply_runtime_defaults(inline_watch_enabled=False)
    try:
        main()
        # Rank zero drains online tracking and uploads final diagnostics after
        # training. Other ranks must not start JAX shutdown while that is pending.
        if jax.distributed.is_initialized():
            multihost_utils.sync_global_devices("kma-h100-tracking-complete")
    finally:
        # The outer holder owns the allocation. Keep normal JAX shutdown, but
        # remove rank zero's callback that would complete the whole Iris job.
        finalize = levanter.distributed._finalize_iris_jax_after_clean_exit
        atexit.unregister(finalize)
        atexit.register(finalize)


if __name__ == "__main__":
    run()
