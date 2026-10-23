# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training algebra and fused-program checks for the interval tree."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe import head_stack_kda as stack
from experiments.grug.moe.head_stack_state_tpu import chunked
from experiments.grug.moe.head_stack_tree import fused_prepare, prepare_tree
from experiments.grug.moe_cplm.benchmark_head_stack_kda import fixture, objective


def assert_tree_close(actual, expected, atol=1e-5, rtol=1e-4):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def test_tree_matches_sequential_preparation_and_adjoint():
    inputs, _ = fixture(16, rank=8, width=8, mixed=True)
    split = stack._split(inputs, 16)
    assert_tree_close(prepare_tree(split), stack._prepare(split, "fp32"))
    def loss(fn, xs):
        return sum(jnp.sin(x).sum() for x in fn(xs))
    actual = jax.jit(jax.grad(lambda xs: loss(prepare_tree, xs)))(split)
    expected = jax.jit(jax.grad(lambda xs: loss(lambda x: stack._prepare(x, "fp32"), xs)))(split)
    assert_tree_close(actual, expected)


@pytest.mark.parametrize("precision", ["fp32", "mixed"])
def test_fused_tree_and_its_executed_vjp(precision):
    inputs, _ = fixture(4, rank=8, width=8, mixed=True)
    split = stack._split(inputs, 4)
    expected, pullback = jax.vjp(lambda xs: prepare_tree(xs, precision), split)
    actual, fused_pullback = jax.vjp(lambda xs: fused_prepare(xs, precision, True), split)
    assert_tree_close(actual, expected)
    cotangent = jax.tree.map(jnp.cos, expected)
    assert_tree_close(fused_pullback(cotangent), pullback(cotangent))


def test_fused_state_forward_and_reverse_with_document_reset():
    inputs, initial = fixture(8, rank=8, width=8, mixed=True)
    fn = lambda xs, s: chunked(xs, s, 4, interpret=True)
    assert_tree_close(jax.jit(fn)(inputs, initial), stack.reference(inputs, initial))
    actual = jax.jit(jax.value_and_grad(lambda x, s: objective(fn, x, s), argnums=(0, 1)))(inputs, initial)
    expected = jax.jit(jax.value_and_grad(lambda x, s: objective(stack.reference, x, s), argnums=(0, 1)))(inputs, initial)
    assert_tree_close(actual, expected)
