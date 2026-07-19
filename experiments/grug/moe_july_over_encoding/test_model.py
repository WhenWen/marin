# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.moe.model import GrugModelConfig as JulyGrugModelConfig
from experiments.grug.moe.model import Transformer as JulyTransformer
from experiments.grug.moe_july_over_encoding.model import (
    GrugModelConfig,
    OverEncoding,
    Transformer,
    _batch_spec,
    _causal_ngram_ids,
)


def _single_device_grug_mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _tiny_model_fields() -> dict[str, object]:
    return {
        "vocab_size": 32,
        "hidden_dim": 8,
        "intermediate_dim": 4,
        "shared_expert_intermediate_dim": 4,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_layers": 1,
        "num_heads": 2,
        "num_kv_heads": 1,
        "max_seq_len": 4,
        "sliding_window": 4,
    }


def test_causal_ngram_ids_respect_packed_document_boundaries():
    token_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    segment_ids = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)

    bigrams = _causal_ngram_ids(token_ids, order=2, modulus=97, base_vocab_size=10, segment_ids=segment_ids)
    trigrams = _causal_ngram_ids(token_ids, order=3, modulus=97, base_vocab_size=10, segment_ids=segment_ids)

    np.testing.assert_array_equal(bigrams, [[1, 12, 3, 34]])
    np.testing.assert_array_equal(trigrams, [[1, 12, 3, 34]])


def test_over_encoding_single_rank_uses_all_hierarchical_slices():
    config = GrugModelConfig(
        **_tiny_model_fields(),
        over_encoding_vocab_size=17,
        over_encoding_splits=2,
        over_encoding_num_grams=3,
    )
    token_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    segment_ids = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)

    with jax.set_mesh(_single_device_grug_mesh()):
        over_encoding = OverEncoding.init(config, key=jax.random.PRNGKey(0))
        output = over_encoding(token_ids, segment_ids)

    assert over_encoding.logical_vocab_sizes == (17, 19, 21, 23)
    assert all(table.shape == (256, 2) for table in over_encoding.tables)
    assert output.shape == (1, 4, 8)
    assert bool(jnp.all(jnp.isfinite(output)))


def test_input_embedding_normalizes_token_and_over_encoding_before_combining():
    config = GrugModelConfig(
        **_tiny_model_fields(),
        over_encoding_vocab_size=17,
        over_encoding_splits=2,
        over_encoding_num_grams=3,
    )
    token_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    segment_ids = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)

    with jax.set_mesh(_single_device_grug_mesh()):
        model = Transformer.init(config, key=jax.random.PRNGKey(42))
        actual = model.input_embedding(token_ids, segment_ids)
        token_embedding = model.token_embed.at[token_ids].get(out_sharding=_batch_spec())
        assert model.over_encoding is not None
        over_encoding_embedding = model.over_encoding(token_ids, segment_ids)
        expected = model.embed_gated_norm(
            (model.embed_norm(token_embedding) + model.embed_norm(over_encoding_embedding)) / jnp.sqrt(2.0)
        )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_disabling_over_encoding_preserves_canonical_july_initialization():
    july_config = JulyGrugModelConfig(**_tiny_model_fields())
    oe_config = GrugModelConfig(**_tiny_model_fields(), over_encoding_vocab_size=0)
    key = jax.random.PRNGKey(42)

    with jax.set_mesh(_single_device_grug_mesh()):
        july_model = JulyTransformer.init(july_config, key=key)
        oe_model = Transformer.init(oe_config, key=key)

    july_arrays = [leaf for leaf in jax.tree.leaves(eqx.filter(july_model, eqx.is_array)) if leaf is not None]
    oe_arrays = [leaf for leaf in jax.tree.leaves(eqx.filter(oe_model, eqx.is_array)) if leaf is not None]
    assert len(oe_arrays) == len(july_arrays)
    for oe_array, july_array in zip(oe_arrays, july_arrays, strict=True):
        np.testing.assert_array_equal(oe_array, july_array)

    july_fields = {field.name for field in dataclasses.fields(july_config)}
    assert all(getattr(oe_config, field) == getattr(july_config, field) for field in july_fields)
