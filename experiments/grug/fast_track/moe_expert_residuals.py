# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental retained pullbacks for the portable expert MLP."""

import jax
import jax.numpy as jnp


def saved_expert_forward(self, x_dispatch, up, down, physical_sizes, active_sizes):
    """Construct the expert pullback once, retaining its forward residuals."""
    output, pullback = jax.vjp(lambda x, u, d: self._apply(x, u, d, physical_sizes, active_sizes), x_dispatch, up, down)
    return output, (output, pullback)


def saved_expert_backward(self, residuals, cotangent):
    """Reuse forward intermediates for activation and parameter gradients."""
    output, pullback = residuals
    dx, dup, ddown = pullback(cotangent)
    row_dot = jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32), axis=-1)
    return dx, dup, ddown, row_dot
