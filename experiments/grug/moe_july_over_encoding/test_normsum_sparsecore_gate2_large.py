# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math

import pytest
from levanter.tracker.wandb import WandbConfig

from experiments.grug.moe.moe_may_july_baseline import _build_step as build_july_step
from experiments.grug.moe_july_over_encoding.lr_sweep import OVER_ENCODING_SCALE_CONSTANT
from experiments.grug.moe_july_over_encoding.normsum_sparsecore_gate2_large import build_steps


def _assert_shared_fields_match(reference, candidate) -> None:
    for field in dataclasses.fields(reference):
        assert getattr(candidate, field.name) == getattr(reference, field.name)


@pytest.mark.parametrize(
    ("index", "hidden_dim", "batch_size", "num_steps", "over_encoding_vocab_size", "peak_table_lr"),
    [
        (0, 1024, 64, 16_080, 4_579_465, 0.0010143139177753492),
        (1, 1280, 128, 14_325, 6_400_001, 0.0011042556770567953),
    ],
)
def test_large_gate2_points_match_canonical_july_outside_oe_fields(
    index: int,
    hidden_dim: int,
    batch_size: int,
    num_steps: int,
    over_encoding_vocab_size: int,
    peak_table_lr: float,
) -> None:
    july = build_july_step(hidden_dim=hidden_dim, batch_size=batch_size, num_steps=num_steps).config
    candidate = build_steps()[index].config

    _assert_shared_fields_match(july.model.value, candidate.model.value)
    _assert_shared_fields_match(july.optimizer.value, candidate.optimizer.value)

    model = candidate.model.value
    optimizer = candidate.optimizer.value
    scaled_m = OVER_ENCODING_SCALE_CONSTANT * hidden_dim**1.5

    assert candidate.run_id == f"MOE-OE-JULY-NORMSUM-SC-GATE2-d{hidden_dim}"
    assert model.hidden_dim == hidden_dim
    assert model.over_encoding_vocab_size == over_encoding_vocab_size
    assert abs(model.over_encoding_vocab_size - scaled_m) <= 3
    assert math.gcd(model.over_encoding_vocab_size, model.vocab_size) == 1
    assert model.over_encoding_splits == 4
    assert model.over_encoding_num_grams == 3
    assert model.hidden_dim // (model.over_encoding_splits * (model.over_encoding_num_grams - 1)) == hidden_dim // 8
    assert optimizer.over_encoding_lr_multiplier == 0.5
    assert math.isclose(optimizer.adam_lr * optimizer.over_encoding_lr_multiplier, peak_table_lr)
    assert candidate.batch_size.value == batch_size
    assert candidate.steps.value == num_steps
    assert candidate.seed.value == 0
    assert candidate.data == july.data
    assert candidate.resources.value == july.resources.value
    assert candidate.mp.value == july.mp.value
    assert dataclasses.asdict(candidate.grug_trainer.value) == dataclasses.asdict(july.grug_trainer.value)
    assert dataclasses.asdict(candidate.eval.value) == dataclasses.asdict(july.eval.value)
    assert isinstance(candidate.tracker, WandbConfig)
    assert candidate.tracker.project == "dial_moe"
    assert candidate.tracker.group == "MOE-OE-JULY-normsum-sc-gate2-large-issue-7368"
