# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig

from experiments.grug.moe_july_over_encoding.normsum_sparsecore_gate2_d1024_ram400 import build_step
from experiments.grug.moe_july_over_encoding.normsum_sparsecore_gate2_large import build_steps


def test_d1024_ram400_retry_only_changes_identity_tracker_and_host_ram():
    original = build_steps()[0]
    retry = build_step()

    assert retry.name == "grug/MOE-OE-JULY-NORMSUM-SC-GATE2-RAM400-d1024"
    assert retry.config.run_id == "MOE-OE-JULY-NORMSUM-SC-GATE2-RAM400-d1024"
    assert retry.config.resources.value == ResourceConfig.with_tpu("v5p-8", ram="400g")
    assert isinstance(retry.config.tracker, WandbConfig)
    assert retry.config.tracker.group == "MOE-OE-JULY-normsum-sc-gate2-d1024-ram400-issue-7368"

    ignored_fields = {"run_id", "resources", "tracker"}
    for field in dataclasses.fields(original.config):
        if field.name not in ignored_fields:
            assert getattr(retry.config, field.name) == getattr(original.config, field.name)
