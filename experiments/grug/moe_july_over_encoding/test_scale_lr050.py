# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math

from levanter.tracker.wandb import WandbConfig

from experiments.grug.moe.moe_may_july_baseline import _build_step as build_july_step
from experiments.grug.moe_july_over_encoding.lr_sweep import OVER_ENCODING_SCALE_CONSTANT
from experiments.grug.moe_july_over_encoding.scale_lr050 import build_step


def _assert_shared_fields_match(reference, candidate) -> None:
    for field in dataclasses.fields(reference):
        assert getattr(candidate, field.name) == getattr(reference, field.name)


def test_lr050_d768_matches_canonical_july_outside_oe_fields():
    july = build_july_step(hidden_dim=768, batch_size=32, num_steps=16_875).config
    candidate = build_step().config

    _assert_shared_fields_match(july.model.value, candidate.model.value)
    _assert_shared_fields_match(july.optimizer.value, candidate.optimizer.value)

    model = candidate.model.value
    optimizer = candidate.optimizer.value
    scaled_m = OVER_ENCODING_SCALE_CONSTANT * 768**1.5

    assert candidate.run_id == "MOE-OE-JULY-SCALE-LR050-d768"
    assert model.over_encoding_vocab_size == 2_974_451
    assert abs(model.over_encoding_vocab_size - scaled_m) <= 2
    assert math.gcd(model.over_encoding_vocab_size, model.vocab_size) == 1
    assert model.over_encoding_splits == 4
    assert model.over_encoding_num_grams == 3
    assert optimizer.over_encoding_lr_multiplier == 0.5
    assert candidate.batch_size.value == 32
    assert candidate.steps.value == 16_875
    assert candidate.seed.value == 0
    assert candidate.data == july.data
    assert candidate.resources.value == july.resources.value
    assert candidate.mp.value == july.mp.value
    assert dataclasses.asdict(candidate.grug_trainer.value) == dataclasses.asdict(july.grug_trainer.value)
    assert dataclasses.asdict(candidate.eval.value) == dataclasses.asdict(july.eval.value)
    assert isinstance(candidate.tracker, WandbConfig)
    assert candidate.tracker.project == "dial_moe"
    assert candidate.tracker.group == "MOE-OE-JULY-scale-lr050-issue-7368"
