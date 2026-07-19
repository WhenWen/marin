# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from experiments.grug.moe_july_over_encoding.throughput_benchmark import build_step


def test_throughput_cells_differ_only_by_over_encoding_model_fields():
    baseline = build_step(enable_over_encoding=False).config
    over_encoding = build_step(enable_over_encoding=True).config

    baseline_model = baseline.model.value
    over_encoding_model = over_encoding.model.value
    differing_model_fields = {
        field.name
        for field in dataclasses.fields(baseline_model)
        if getattr(baseline_model, field.name) != getattr(over_encoding_model, field.name)
    }

    assert differing_model_fields == {"over_encoding_table_dim", "over_encoding_vocab_size"}
    assert baseline_model.over_encoding_vocab_size == 0
    assert over_encoding_model.over_encoding_vocab_size == 1_619_087
    assert baseline_model.over_encoding_table_dim == 0
    assert over_encoding_model.over_encoding_table_dim == 16
    assert baseline.optimizer.value == over_encoding.optimizer.value
    assert baseline.data == over_encoding.data
    assert baseline.resources.value == over_encoding.resources.value
    assert baseline.steps.value == over_encoding.steps.value == 700
    assert baseline.batch_size.value == over_encoding.batch_size.value == 16
    assert baseline.seed.value == over_encoding.seed.value == 0
    assert baseline.mp.value == over_encoding.mp.value
    assert baseline.eval is None
    assert over_encoding.eval is None
    assert baseline.profiler == over_encoding.profiler
    assert baseline.profiler.enabled
    assert baseline.profiler.start_step == 100
    assert baseline.profiler.num_steps == 50
    assert baseline.run_id == "MOE-JULY-RANK16-PERF-BASELINE-d512"
    assert over_encoding.run_id == "MOE-JULY-RANK16-PERF-OE-d512"
    assert baseline.tracker.group == over_encoding.tracker.group == ("MOE-OE-JULY-rank16-throughput-issue-7368")
