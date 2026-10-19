# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay accumulation with one QB grid and update per logical batch."""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.pipeline import reshape_batch_into_microbatches

from experiments.grug.fast_track import model, qb_balancing, train
from experiments.grug.fast_track.router_metrics import router_metrics_from_sums, summarize_router_metrics


def pooled_qb(margins, *, num_experts_per_token, n_bins=model._QB_HIST_BINS):
    """Read whole-batch thresholds from detached [micro, layer, token, expert] margins."""
    _, layers, tokens, experts = margins.shape
    # Put the micro axis inside the sharded token axis: this needs no token redistribution.
    joined = margins.transpose(1, 2, 0, 3).reshape(layers, tokens * margins.shape[0], experts)
    joined = jax.sharding.reshard(joined, P(None, model._BATCH_AXES, None))

    def one_layer(values):
        return qb_balancing.qb_beta_hist(
            values,
            jax.sharding.get_abstract_mesh(),
            target_share=num_experts_per_token / experts,
            num_experts=experts,
            n_bins=n_bins,
            batch_axes=model._BATCH_AXES,
        )

    return jax.lax.map(one_layer, joined)


def accumulated_loss_and_grads(num_microbatches, params, batch, mp, z_loss, step, loop_active, router_tie_active):
    """Accumulate weighted NLL gradients, then pool QB and router telemetry once.

    Parameters and routing biases are held fixed throughout the scan. Only detached
    margins survive between microbatches; the model forward is never replayed for QB.
    The frozen Belay recipe has no stochastic routing or batch-dependent loss terms.
    """
    if not params.config.qb_defer_update:
        raise ValueError("Belay accumulation requires deferred whole-batch QB statistics")
    micro = reshape_batch_into_microbatches(batch, num_microbatches)
    denominator = jnp.maximum(jnp.sum(batch.loss_weight.astype(jnp.float32)), 1.0)

    def one(microbatch):
        scale = jnp.maximum(jnp.sum(microbatch.loss_weight.astype(jnp.float32)), 1.0) / denominator
        (loss, metrics), grads = train._loss_and_grads(
            params,
            microbatch,
            mp,
            z_loss,
            step,
            loop_active,
            router_tie_active=router_tie_active,
            loss_scale=scale,
        )
        raw = metrics["router"]
        additive = {
            "counts": raw["routing_counts_per_layer"],
            "prob_sum": raw["qb_accum_prob_sum"],
            "z_sq_sum": raw["qb_accum_z_sq_sum"],
            "capacity_overflow_per_layer": raw["capacity_overflow_per_layer"],
            "sender_capacity_overflow_per_layer": raw["sender_capacity_overflow_per_layer"],
            "receiver_capacity_overflow_per_layer": raw["receiver_capacity_overflow_per_layer"],
            "token_means": jax.tree.map(lambda x: x / num_microbatches, metrics["token_means"]),
            "weighted_means": jax.tree.map(lambda x: x * scale, metrics["weighted_means"]),
        }
        return (loss, additive, grads), raw["qb_accum_margins"]

    shapes, _ = jax.eval_shape(one, jax.tree.map(lambda x: x[0], micro))
    zeros = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype, out_sharding=s.sharding.spec), shapes)

    def accumulate(carry, microbatch):
        values, margins = one(microbatch)
        return jax.tree.map(jnp.add, carry, values), margins

    (loss, totals, grads), margins = jax.lax.scan(accumulate, zeros, micro)
    beta, lo, hi = pooled_qb(margins, num_experts_per_token=params.config.num_experts_per_token)
    reduced = router_metrics_from_sums(
        totals.pop("counts"),
        totals.pop("prob_sum"),
        totals.pop("z_sq_sum"),
        batch.tokens.size,
        params.config.num_experts_per_token,
        params.config.num_experts,
    )
    token_means, weighted_means = totals.pop("token_means"), totals.pop("weighted_means")
    router = {f"{key}_per_layer": value for key, value in reduced.items()}
    router.update(totals)
    router.update(qb_beta_per_layer=beta, margin_min_per_layer=lo, margin_max_per_layer=hi)
    metrics = summarize_router_metrics(router)
    metrics.update(token_means)
    metrics.update(weighted_means)
    metrics["train/router/z_loss_logging_only"] = jnp.mean(reduced["router_z_loss"])
    for name, key in (
        ("dropped_assignments", "capacity_overflow_per_layer"),
        ("sender_dropped_assignments", "sender_capacity_overflow_per_layer"),
        ("receiver_dropped_assignments", "receiver_capacity_overflow_per_layer"),
    ):
        metrics[f"moe/{name}"] = totals[key]
    return (loss, metrics), grads
