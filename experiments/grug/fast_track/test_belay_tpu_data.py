# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data-only integration checks; these tests never initialize a neural model."""

import json

import numpy as np
from levanter.store.cache import TreeCache
from levanter.tokenizers import load_tokenizer
from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

from experiments.grug.fast_track.build_belay_tpu_data import EXEMPLAR, SPECIALS, write_cache


def test_cache_preserves_document_boundaries_and_unicode(tmp_path):
    texts = ["Hello 世界!\n\n", "  café and λ\t1234567", "🙂", ""]
    backend = Tokenizer(BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=512, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=list(SPECIALS)
    ))
    tokenizer_dir = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token=SPECIALS[0], eos_token=SPECIALS[1], pad_token=SPECIALS[7]
    ).save_pretrained(tokenizer_dir)
    tokenizer = load_tokenizer(str(tokenizer_dir))
    url = str(tmp_path / "cache")
    stats = write_cache(url, iter(texts), tokenizer)
    cache = TreeCache.load(url, EXEMPLAR)
    rows = cache.get_batch_sync(list(range(len(texts))))
    for text, row in zip(texts, rows, strict=True):
        ids = row["input_ids"]
        np.testing.assert_array_equal(ids[[0, -1]], [0, 1])
        assert tokenizer.decode(ids[1:-1].tolist()) == text + " "
    assert stats["tokens"] == sum(len(row["input_ids"]) for row in rows)
    assert json.loads((tmp_path / "cache" / "validated.json").read_text()) == stats
