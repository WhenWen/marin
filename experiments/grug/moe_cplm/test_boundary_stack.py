# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training numerical checks for fused boundary head mixing."""

import jax.numpy as jnp
import pytest

from experiments.grug.moe_cplm.boundary_stack_validation import candidate, compare, evaluate, fixture, literal


def test_fp32_forward_and_all_adjoints_with_reset():
    xs, state = fixture(8, rank=8, chunk=4)
    compare(evaluate(lambda x, s: candidate(x, s, 4, interpret=True), xs, state),
            evaluate(lambda x, s: literal(x, s, 4), xs, state))


@pytest.mark.parametrize("mixed", [False, True])
def test_fused_reverse_matches_executed_native_scan(mixed):
    xs, state = fixture(8, rank=8, chunk=4)
    compare(evaluate(lambda x, s: candidate(x, s, 4, mixed, True), xs, state),
            evaluate(lambda x, s: candidate(x, s, 4, mixed, reference=True), xs, state))


def test_push_pop_preserves_three_states_and_boundary_is_causal():
    xs, state = fixture(8, rank=8, chunk=4)
    actions = jnp.array([[[1, 0, 0], [0, 1, 0]]], jnp.float32)
    # Zero decay log and beta disable KDA writes; all slots retain their values.
    fixed = (actions, *xs[1:4], jnp.zeros_like(xs[4]), jnp.zeros_like(xs[5]), jnp.ones_like(xs[6]))
    out, final = candidate(fixed, state, 4, interpret=True)
    compare(final, jnp.concatenate((state[:, :3], jnp.zeros_like(state[:, :1])), axis=1))
    stay = (jnp.broadcast_to(jnp.array([0, 0, 1], jnp.float32), actions.shape), *fixed[1:])
    other, _ = candidate(stay, state, 4, interpret=True)
    compare(out[:, :4], other[:, :4])


def test_chunk_continuation():
    xs, state = fixture(8, rank=8, chunk=4)
    xs = (*xs[:-1], jnp.ones_like(xs[-1]))
    first = (xs[0][:, :1], *(x[:, :4] for x in xs[1:]))
    second = (xs[0][:, 1:], *(x[:, 4:] for x in xs[1:]))
    left, middle = candidate(first, state, 4, interpret=True)
    right, final = candidate(second, middle, 4, interpret=True)
    compare((jnp.concatenate((left, right), axis=1), final), candidate(xs, state, 4, interpret=True))
