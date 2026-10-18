# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero-update TPU check of the fused head against its dense vocabulary path."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe_cplm import belay, plus
from experiments.grug.fast_track import qb_balancing


def validate_histogram(local_tokens):
    """Check the B128 integer counts and two-stage quantile before model allocation."""
    bins, experts = 10000, 512
    mesh = jax.sharding.Mesh(np.asarray([jax.local_devices()[0]]), ("probe",))
    rank = np.full(experts, local_tokens * 8 / experts, np.float32)

    def quantile(indices):
        crossing, count, height = qb_balancing.two_stage_histogram_crossing(
            indices, jnp.asarray(rank), batch_axes=("probe",), n_bins=bins, group_size=64
        )
        return crossing.astype(jnp.float32) + (count - jnp.asarray(rank)) / jnp.maximum(height, 1)

    with jax.set_mesh(mesh), jax.default_device(jax.local_devices()[0]):
        counts = jax.jit(lambda x: qb_balancing.factorized_expert_histogram(x, bins, 64, 64))
        threshold = jax.jit(jax.shard_map(quantile, mesh=mesh, in_specs=jax.P(), out_specs=jax.P()))
        rng = np.random.default_rng(53)
        cases = []
        for distribution in ("random", "constant", "edges"):
            indices = rng.integers(0, bins, (local_tokens, experts), dtype=np.int32)
            if distribution == "constant":
                indices.fill(42)
            elif distribution == "edges":
                indices = np.where(indices % 2, 0, bins - 1).astype(np.int32)
            expected = np.stack([np.bincount(column, minlength=bins) for column in indices.T])
            actual = np.asarray(counts(jnp.asarray(indices)))
            np.testing.assert_array_equal(actual, expected)
            want = qb_balancing.histogram_quantile(jnp.asarray(expected, jnp.float32), jnp.asarray(rank), 0.0, 1.0)
            np.testing.assert_array_equal(np.asarray(threshold(jnp.asarray(indices))), np.asarray(want))
            cases.append({"distribution": distribution, "exact_counts": True, "exact_quantile": True})
        counts.clear_cache()
        threshold.clear_cache()
    return {"training_updates": 0, "shape": [local_tokens, experts], "bins": bins, "cases": cases}


def validate_head(variant, width):
    """Compare loss and VJP at the real hidden width before any optimizer step."""
    # 1024 rows satisfy the fused CE block. A small vocabulary exercises the
    # same width-dependent kernel tiles without allocating the full LM state.
    shapes = [(1, 1024, width), (width, 512), (width, width), (width, width), (width,), (width,)]
    if variant == "plus":
        shapes += [(width, width), (width,), (width,), ()]
    with jax.default_device(jax.local_devices()[0]):
        params = tuple(
            (0.02 * jax.random.normal(k, s)).astype(jnp.bfloat16)
            for k, s in zip(jax.random.split(jax.random.key(16), len(shapes)), shapes, strict=True)
        )
        tokens = (jnp.arange(1024) % 31)[None]
        labels = jnp.roll(tokens, -1, axis=1)
        segments = (jnp.arange(1024) // 256)[None]
        classes = jnp.arange(512) // 2
        members = (jnp.arange(512) // 2)[:, None] * 2 + jnp.arange(2)

        def loss(p, reference):
            h, head, qw, kw, gw, sink, *extra = p
            if variant == "belay":
                return belay.vocabulary_statistics(h, head, labels, reference=reference)[0].mean()
            fn = plus.local_head if variant == "plus" else belay.local_head
            args = (h, head, qw, kw, gw, sink, tokens, labels, segments)
            if variant == "plus":
                args += (*extra, classes, members)
            return fn(*args, reference=reference)[0].mean()

        fused = jax.jit(jax.value_and_grad(lambda p: loss(p, False)))
        dense = jax.jit(jax.value_and_grad(lambda p: loss(p, True)))
        got, got_grad = jax.device_get(fused(params))
        want, want_grad = jax.device_get(dense(params))
        # BF16 cotangents are rounded at the head's input boundary. Compare
        # norm error as well as the loss; tiny zero entries cannot set a scale.
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
        errors = []
        for a, b in zip(got_grad, want_grad, strict=True):
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            assert np.isfinite(a).all()
            error = float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-9))
            assert error < 0.02, error
            errors.append(error)
        fused.clear_cache()
        dense.clear_cache()
        return {
            "loss": float(got),
            "reference_loss": float(want),
            "gradient_relative_errors": errors,
            "training_updates": 0,
            "width": width,
            "variant": variant,
        }
