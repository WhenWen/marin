# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math

from experiments.grug.moe.moe_may_july_baseline import _build_step as _build_july_step
from experiments.grug.moe_july_over_encoding.lr_sweep import (
    LR_SWEEP_POINTS,
    OVER_ENCODING_SCALE_CONSTANT,
    _build_step,
)


def _common_field_values(reference, candidate, excluded: set[str]) -> dict[str, object]:
    return {
        field.name: getattr(candidate, field.name)
        for field in dataclasses.fields(reference)
        if field.name not in excluded
    }


def test_sweep_is_fixed_c_and_changes_only_over_encoding_table_lr():
    steps = [_build_step(point) for point in LR_SWEEP_POINTS]
    expected_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]

    assert [point.over_encoding_lr_multiplier for point in LR_SWEEP_POINTS] == expected_multipliers
    assert len({step.config.run_id for step in steps}) == len(steps)
    assert {step.config.model.value.over_encoding_vocab_size for step in steps} == {1_619_087}
    assert {step.config.model.value.over_encoding_table_dim for step in steps} == {64}
    assert {step.config.optimizer.value.over_encoding_lr_multiplier for step in steps} == set(expected_multipliers)
    assert all(step.config.batch_size.value == 16 for step in steps)
    assert all(step.config.steps.value == 10_980 for step in steps)
    assert all(step.config.seed.value == 0 for step in steps)

    scaled_m = OVER_ENCODING_SCALE_CONSTANT * 512**1.5
    assert abs(1_619_087 - scaled_m) <= 2
    assert math.gcd(1_619_087, steps[0].config.model.value.vocab_size) == 1


def test_sweep_cell_matches_canonical_july_baseline_outside_oe_fields():
    july = _build_july_step(hidden_dim=512, batch_size=16, num_steps=10_980).config
    candidate = _build_step(LR_SWEEP_POINTS[2]).config

    july_model = july.model.value
    candidate_model = candidate.model.value
    assert _common_field_values(july_model, candidate_model, excluded=set()) == {
        field.name: getattr(july_model, field.name) for field in dataclasses.fields(july_model)
    }

    july_optimizer = july.optimizer.value
    candidate_optimizer = candidate.optimizer.value
    assert _common_field_values(july_optimizer, candidate_optimizer, excluded=set()) == {
        field.name: getattr(july_optimizer, field.name) for field in dataclasses.fields(july_optimizer)
    }

    assert candidate_model.over_encoding_vocab_size == 1_619_087
    assert candidate_model.over_encoding_table_dim == 64
    assert candidate_model.over_encoding_splits == 4
    assert candidate_model.over_encoding_num_grams == 3
    assert candidate_optimizer.over_encoding_lr_multiplier == 1.0
    assert candidate.data == july.data
    assert candidate.resources.value == july.resources.value
    assert candidate.steps.value == july.steps.value
    assert candidate.batch_size.value == july.batch_size.value
    assert candidate.seed.value == july.seed.value
    assert candidate.mp.value == july.mp.value
    assert dataclasses.asdict(candidate.grug_trainer.value) == dataclasses.asdict(july.grug_trainer.value)
    assert dataclasses.asdict(candidate.eval.value) == dataclasses.asdict(july.eval.value)
