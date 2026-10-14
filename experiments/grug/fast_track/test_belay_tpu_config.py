# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration regressions without model initialization or training."""

import json

import jax

from experiments.grug.fast_track.belay_tpu import SOURCE_CONFIG, build_config


def test_tpu_config_disables_pgle_in_jax(tmp_path):
    source = json.loads(SOURCE_CONFIG.read_text())
    cache_map = {name: f"gs://marin-us-central2/test/{name}" for name in source["data"]["components"]}
    config = build_config(
        run_id="config-regression",
        cache_map=cache_map,
        output=tmp_path,
        checkpoint_path="gs://marin-us-central2/test/checkpoints",
        chips=16,
        microbatches=1,
        seed=0,
        execution_profile="selected",
    )
    settings = config.trainer.trainer.jax_config
    previous = {key: getattr(jax.config, key) for key in settings}
    try:
        jax.config.update("jax_enable_pgle", True)
        for key, value in settings.items():
            jax.config.update(key, value)
        assert jax.config.jax_enable_pgle is False
        assert jax.config.jax_softmax_custom_jvp is True
        assert jax.config.jax_threefry_partitionable is True
    finally:
        for key, value in previous.items():
            jax.config.update(key, value)
