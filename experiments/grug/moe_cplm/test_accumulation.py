# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whole-batch QB and NLL/VJP comparisons, with no optimizer updates."""

from dataclasses import replace
from functools import partial
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug._moe.common import _scaled_capacity
from levanter.grug._moe.ep_common import _clip_receiver_group_sizes
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track import model, train
from experiments.grug.moe_cplm import accumulation, belay
from experiments.grug.moe_cplm.train_belay import DATA, build_config


def _mesh():
    return Mesh(
        np.asarray(jax.devices()).reshape(1, len(jax.devices()), 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


@pytest.mark.parametrize("micros", [1, 2, 4])
@pytest.mark.parametrize("constant", [False, True])
def test_pooled_qb_uses_the_whole_batch_range(micros, constant):
    mesh = _mesh()
    rng = np.random.default_rng(14)
    tokens, experts, layers, bins = 32 * len(jax.devices()), 4, 2, 10000
    values = rng.normal(size=(layers, tokens, experts)).astype(np.float32)
    values[:, : tokens // 4] *= 13
    if constant:
        values.fill(2)
    expected = []
    for layer in values:
        lo, hi = layer.min(), layer.max()
        # XLA lowers division by the constant bin count to multiplication by its float32 reciprocal.
        width = (np.maximum(hi, lo + np.float32(1e-6)) - lo) * np.float32(1 / bins)
        indices = np.clip(((layer - lo) / width).astype(np.int32), 0, bins - 1)
        thresholds = []
        for column in indices.T:
            counts = np.bincount(column, minlength=bins)
            cumulative = counts[::-1].cumsum()[::-1]
            rank = tokens * 2 / experts
            crossing = np.clip(np.count_nonzero(cumulative >= rank) - 1, 0, bins - 1)
            # The production quantile uses float32, including the crossing-bin interpolation.
            lower = np.float32(np.float64(lo) + np.float64(crossing) * np.float64(width))
            offset = np.float32(width * np.float32(cumulative[crossing] - rank) / np.float32(max(counts[crossing], 1)))
            thresholds.append(np.float32(lower + offset))
        expected.append(thresholds)
    split = values.reshape(layers, micros, tokens // micros, experts).transpose(1, 0, 2, 3)
    with jax.set_mesh(mesh):
        x = jax.device_put(split, NamedSharding(mesh, P(None, None, model._BATCH_AXES, None)))
        beta, lo, hi = jax.jit(partial(accumulation.pooled_qb, num_experts_per_token=2))(x)
    np.testing.assert_allclose(beta, expected, rtol=2e-7, atol=2e-6)
    np.testing.assert_array_equal(lo, values.min(axis=(1, 2)))
    np.testing.assert_array_equal(hi, values.max(axis=(1, 2)))


@pytest.mark.parametrize("ep", [2, 4])
@pytest.mark.parametrize("padded", [False, True])
def test_dropless_capacity_retains_all_assignments_when_one_receiver_chunk_gets_every_token(tmp_path, ep, padded):
    cfg = build_config(
        data_path=DATA,
        output=tmp_path,
        checkpoint_path=str(tmp_path / "checkpoints"),
        run_name="capacity-reference",
        wandb_entity="marin-community",
        wandb_project="auto-understanding-arch",
        calibration=False,
        tokenizer_path=str(tmp_path / "tokenizer"),
        width=768,
        batch_size=64,
        expert_axis_size=ep,
    ).model
    experts, chunks, tokens, k = cfg.num_experts, cfg.moe_ragged_chunks, 64 * 8192 // 16, cfg.num_experts_per_token
    local_experts = experts // ep
    # Every token chooses k distinct experts in receiver 0's first chunk.
    counts = np.zeros((ep, experts), np.int32)
    for sender in range(ep):
        counts[sender, :k] = tokens - (sender * tokens // ep if padded else 0)
    counts = jnp.asarray(counts)
    physical_capacity = int(np.ceil(cfg.capacity_factor * tokens * k))

    @jax.jit
    def accepted(group_sizes):
        logical = _scaled_capacity(
            jnp.sum(group_sizes),
            capacity_factor=cfg.capacity_factor,
            divisor=ep,
            minimum=local_experts,
            maximum=physical_capacity,
        )
        per_chunk = jnp.maximum((logical + chunks - 1) // chunks, local_experts // chunks)
        chunk_ids = (jnp.arange(experts) % local_experts) // (local_experts // chunks)
        return sum(
            _clip_receiver_group_sizes(
                jnp.where(chunk_ids[None] == chunk, group_sizes, 0),
                local_expert_size=local_experts,
                receiver_capacity=per_chunk,
            )
            for chunk in range(chunks)
        )

    np.testing.assert_array_equal(accepted(counts), counts)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("kind", [belay.Belay, belay.CopyPasteBelay])
@pytest.mark.parametrize("micros", [2, 4])
def test_accumulation_matches_whole_batch_nll_gradients_and_router_statistics(kind, micros):
    mesh = _mesh()
    cfg = model.GrugModelConfig(
        vocab_size=32,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=16,
        max_seq_len=16,
        global_every=1,
        attn_res=True,
        attn_res_num_blocks=2,
        attn_res_layer_backward=model.AttnResLayerBackward.SAVE,
        moe_implementation="ragged_all_to_all",
    )
    batch_size = 4 * len(jax.devices())
    tokens = (jnp.arange(batch_size * 16).reshape(batch_size, 16) % 29).astype(jnp.int32)
    weights = jnp.linspace(0, 1, tokens.size).reshape(tokens.shape).at[: batch_size // 4].set(0)
    segments = jnp.broadcast_to(jnp.arange(16)[None] // 8, tokens.shape)
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    # CPU executes the dense vocabulary reference of the same softcap-10 head.
    reference_head = partial(belay.local_head, tile=4, reference=True)
    vocabulary = partial(belay.vocabulary_statistics, reference=True)
    with (
        jax.set_mesh(mesh),
        patch.object(belay, "local_head", reference_head),
        patch.object(belay, "vocabulary_statistics", vocabulary),
    ):
        params = kind.init(cfg, key=jax.random.key(8))
        deferred = kind.init(replace(cfg, qb_defer_update=True), key=jax.random.key(8))
        spec = NamedSharding(mesh, P(model._BATCH_AXES, None))
        batch = GrugLmExample(
            tokens=jax.device_put(tokens, spec),
            loss_weight=jax.device_put(weights, spec),
            attn_mask=AttentionMask(is_causal=True, segment_ids=(jax.device_put(segments, spec),) * 2),
        )
        (want_loss, want_metrics), want_grad = jax.jit(partial(train._loss_and_grads, mp=mp, z_loss=None))(params, batch)
        (got_loss, got_metrics), got_grad = jax.jit(
            partial(
                accumulation.accumulated_loss_and_grads,
                micros,
                mp=mp,
                z_loss=None,
                step=None,
                loop_active=None,
                router_tie_active=None,
            )
        )(deferred, batch)
    np.testing.assert_allclose(got_loss, want_loss, rtol=3e-6, atol=3e-6)
    for actual, expected in zip(jax.tree.leaves(got_grad), jax.tree.leaves(want_grad), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    for key in (
        "qb_beta_per_layer",
        "train/router/routing_counts_per_layer",
        "train/router/routing_entropy_mean",
        "train/router/load_balancing_loss",
        "train/router/router_z_loss",
        "train/router/margin_min",
        "train/router/margin_max",
        "train/aux/cplm/native_head_nll",
        "train/aux/cplm/copy_gate",
        "train/aux/cplm/effective_copy",
    ):
        np.testing.assert_allclose(got_metrics[key], want_metrics[key], rtol=3e-5, atol=3e-6)
    for layer in range(cfg.num_layers):
        key = f"train/router/layer_{layer}/routing_hist"
        np.testing.assert_array_equal(
            got_metrics[key].histogram.bucket_counts, want_metrics[key].histogram.bucket_counts
        )
