# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.kernels.pallas.relative_position_attention import SegmentIds
from levanter.kernels.pallas.relative_position_attention import api


def _inputs(*, seq_len: int = 256, relative_extent: int = 128):
    key = jax.random.key(0)
    keys = jax.random.split(key, 4)
    shape = (1, 1, seq_len, 128)
    q, k, v = (
        jax.random.normal(array_key, shape, dtype=jnp.float32).astype(jnp.bfloat16) * 0.03 for array_key in keys[:3]
    )
    relative_logits = (
        jax.random.normal(keys[3], (1, 1, seq_len, relative_extent), dtype=jnp.float32).astype(jnp.bfloat16) * 0.03
    )
    return q, k, v, relative_logits


def test_pallas_relative_position_attention_matches_reference_values_and_gradients():
    inputs = _inputs(seq_len=512, relative_extent=1024)
    segment_ids = jnp.repeat(jnp.arange(4, dtype=jnp.int32)[None, :], 128, axis=1)
    segments = SegmentIds(segment_ids, segment_ids)
    kwargs = {
        "segment_ids": segments,
        "causal": True,
        "sliding_window": 128,
        "sm_scale": 128**-0.5,
    }

    def loss(*arrays, implementation):
        output = api.relative_position_attention(
            *arrays,
            implementation=implementation,
            interpret=implementation == "pallas_tpu",
            **kwargs,
        )
        return jnp.sum(output.astype(jnp.float32) ** 2)

    def reference_loss(*arrays):
        return loss(*arrays, implementation="reference")

    def pallas_loss(*arrays):
        return loss(*arrays, implementation="pallas_tpu")

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3))(*inputs)
    pallas_value, pallas_grads = jax.value_and_grad(pallas_loss, argnums=(0, 1, 2, 3))(*inputs)

    assert jnp.allclose(pallas_value, reference_value, atol=2e-4, rtol=2e-4)
    for pallas_grad, reference_grad in zip(pallas_grads, reference_grads, strict=True):
        assert jnp.allclose(pallas_grad, reference_grad, atol=1e-3, rtol=2e-2)


def test_reference_uses_inkling_direct_distance_indexing():
    q = jnp.zeros((1, 1, 4, 1), dtype=jnp.float32)
    k = jnp.zeros_like(q)
    v = jnp.arange(4, dtype=jnp.float32).reshape(1, 1, 4, 1)
    relative_logits = jnp.broadcast_to(jnp.arange(4, dtype=jnp.float32), (1, 1, 4, 4))

    output = api.relative_position_attention(
        q,
        k,
        v,
        relative_logits,
        causal=True,
        sliding_window=None,
        sm_scale=1.0,
        implementation="reference",
    )

    # Inkling gathers column query_position - key_position directly. At query
    # 2 the scores for keys [0, 1, 2] are therefore [2, 1, 0].
    expected_query_2 = jnp.sum(jax.nn.softmax(jnp.array([2.0, 1.0, 0.0])) * jnp.array([0.0, 1.0, 2.0]))
    assert jnp.allclose(output[0, 0, 2, 0], expected_query_2, atol=1e-6, rtol=1e-6)


def test_explicit_pallas_requires_tpu_or_interpret_mode():
    with pytest.raises(RuntimeError, match="requires TPU or interpret=True"):
        api.relative_position_attention(
            *_inputs(seq_len=128),
            causal=True,
            sliding_window=64,
            sm_scale=128**-0.5,
            implementation="pallas_tpu",
        )


def test_tpu_api_has_explicit_shard_map_boundary(monkeypatch: pytest.MonkeyPatch):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    sharding = NamedSharding(mesh, P("data", None, None, None))
    inputs = tuple(jax.device_put(x, sharding) for x in _inputs(seq_len=128))
    segment_sharding = NamedSharding(mesh, P("data", None))
    segment_ids = jax.device_put(jnp.zeros((1, 128), dtype=jnp.int32), segment_sharding)
    monkeypatch.setattr(api.jax, "default_backend", lambda: "tpu")

    with jax.set_mesh(mesh):
        jaxpr = jax.make_jaxpr(
            jax.value_and_grad(
                lambda *arrays: jnp.sum(
                    api.relative_position_attention(
                        *arrays,
                        SegmentIds(segment_ids, segment_ids),
                        causal=True,
                        sliding_window=64,
                        sm_scale=128**-0.5,
                    ).astype(jnp.float32)
                ),
                argnums=(0, 1, 2, 3),
            )
        )(*inputs)

    jaxpr_text = str(jaxpr)
    assert jaxpr_text.count("shard_map") >= 3
    assert "pallas_call" in jaxpr_text
