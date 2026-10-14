# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a separately identified 16k-tokenizer corpus for the TPU Belay port."""

import argparse
import gc
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import TreeCache
from levanter.store.jagged_array import set_jagged_array_read_cache_bytes
from levanter.tokenizers import load_tokenizer
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from zephyr.readers import load_file

from experiments.datakit.store.bucket_writer import write_bucket_cache
from experiments.grug.fast_track.belay_tpu import REGION_PREFIX, SOURCE_CONFIG

LOGGER = logging.getLogger(__name__)
SPECIALS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_think|>",
    "<|end_think|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|pad|>",
)
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}
SOURCE_TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"


@dataclass(frozen=True)
class BuildConfig:
    output: str
    revision: str
    source_store: str = "gs://marin-us-central2/datakit/store_8ac06c74"
    eval_inventory: str = "gs://marin-us-central2/kaiyue/belay-base16-tpu/retokenization-v1/eval-source-inventory.json"
    corpus_bytes: int = 12 * 1024**3
    tokenizer_bytes: int = 256 * 1024**2
    document_bytes: int = 32768
    seed: int = 0


def read_json(url):
    return json.loads(StoragePath(url).read_text())


def write_json(url, value):
    StoragePath(url).write_text(json.dumps(value, indent=2) + "\n")


def object_identity(url):
    fs, path = filesystem_for(url)
    info = fs.info(path)
    return {"url": url, **{key: info.get(key) for key in ("size", "generation", "md5Hash")}}


def capped(text: str, limit: int) -> str:
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def eval_records(inventory):
    for name, source in inventory["sources"].items():
        for record in source["files"]:
            if not record["url"].startswith(REGION_PREFIX):
                raise ValueError("Evaluation data must be in us-central2")
            current = object_identity(record["url"])
            for key in ("generation", "md5Hash", "size"):
                if current[key] != record[key]:
                    raise ValueError(f"Evaluation source changed: {record['url']} ({key})")
            for item in load_file(record["url"]):
                yield name, item["text"]


def freeze(config: BuildConfig):
    manifest_path = f"{config.output}/frozen.json"
    store_url = f"{config.source_store}/.artifact.json"
    store = read_json(store_url)
    if store["tokenizer"] != "marin-community/marin-tokenizer":
        raise ValueError("Unexpected source-store tokenizer")
    identity = object_identity(store_url)
    inventory = read_json(config.eval_inventory)
    manifest = {
        "config": asdict(config),
        "source_store_identity": identity,
        "source_store": store,
        "source_tokenizer_revision": SOURCE_TOKENIZER_REVISION,
        "evaluation": inventory,
        "mixture": "UTF-8 byte quotas proportional to the source store's per-cell token counts",
        "selection": "seeded affine permutation of 64-document blocks within each cluster/quality cell",
        "holdout_filter": "exclude exact capped-text SHA256 matches with any evaluation document",
        "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/96",
        "baseline_kind": "new tokenizer and corpus; not an original Belay data reproduction",
    }
    if StoragePath(manifest_path).exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("Frozen inputs/config changed; use a new output prefix")
    else:
        write_json(manifest_path, manifest)
    return manifest


def sample(config: BuildConfig, manifest):
    holdouts = {text_digest(capped(text, config.document_bytes)) for _, text in eval_records(manifest["evaluation"])}
    source_file = hf_hub_download(
        "marin-community/marin-tokenizer", "tokenizer.json", revision=SOURCE_TOKENIZER_REVISION
    )
    source_tokenizer = Tokenizer.from_file(source_file)
    write_json(
        f"{config.output}/source-decoder.json",
        {
            "repo": "marin-community/marin-tokenizer",
            "revision": SOURCE_TOKENIZER_REVISION,
            "tokenizer_json_sha256": hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
        },
    )
    cells = manifest["source_store"]["buckets"]
    total = sum(cell["total_tokens"] for cell in cells)
    summaries = []
    for cell in cells:
        if not cell["total_tokens"]:
            continue
        name = f"c{cell['cluster_id']:02d}q{cell['quality_bucket']}"
        done = f"{config.output}/sample/{name}.json"
        if StoragePath(done).exists():
            summaries.append(read_json(done))
            continue
        quota = max(1, math.ceil(config.corpus_bytes * cell["total_tokens"] / total))
        cache = TreeCache.load(cell["path"], EXEMPLAR)
        if len(cache) != cell["total_elements"]:
            raise ValueError(f"Source cell count changed: {name}")
        ledger_url = f"{cell['path']}/shard_ledger.json"
        ledger_identity = object_identity(ledger_url)
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, cell["cluster_id"], cell["quality_bucket"]]))
        blocks = (len(cache) + 63) // 64
        start = int(rng.integers(blocks))
        stride = int(rng.integers(1, max(2, blocks)))
        while math.gcd(stride, blocks) != 1:
            stride += 1
        rows, selected_bytes, excluded = [], 0, 0
        for step in range(blocks):
            base = ((start + step * stride) % blocks) * 64
            indices = range(base, min(base + 64, len(cache)))
            for index, item in zip(indices, cache.get_batch_sync(indices), strict=True):
                text = capped(
                    source_tokenizer.decode(item["input_ids"].tolist(), skip_special_tokens=True), config.document_bytes
                )
                if not text or text_digest(text) in holdouts:
                    excluded += 1
                    continue
                rows.append({"source_row": index, "text": text})
                selected_bytes += len(text.encode("utf-8"))
                if selected_bytes >= quota:
                    break
            if selected_bytes >= quota:
                break
            if step % 100 == 0:
                LOGGER.info("sampling %s: %d / %d bytes", name, selected_bytes, quota)
        if selected_bytes < quota:
            raise ValueError(f"Insufficient unique text in {name}: {selected_bytes} < {quota}")
        url = f"{config.output}/sample/{name}.parquet"
        with StoragePath(url).open("wb") as stream:
            pq.write_table(pa.Table.from_pylist(rows), stream, compression="zstd")
        summary = {
            "cell": name,
            "source": cell["path"],
            "ledger_identity": ledger_identity,
            "raw": object_identity(url),
            "rows": len(rows),
            "bytes": selected_bytes,
            "quota": quota,
            "excluded_empty_or_exact_holdout": excluded,
        }
        write_json(done, summary)
        summaries.append(summary)
        LOGGER.info("sample %s: %d documents, %d bytes", name, len(rows), selected_bytes)
        del cache, rows
        gc.collect()
    write_json(f"{config.output}/sample.json", summaries)
    return summaries


def raw_texts(summary):
    url = summary["raw"]["url"]
    if object_identity(url) != summary["raw"]:
        raise ValueError(f"Sample changed: {url}")
    with StoragePath(url).open("rb") as stream:
        reader = pq.ParquetFile(stream)
        for batch in reader.iter_batches(batch_size=256, columns=["text"]):
            yield from batch.column(0).to_pylist()


def fit_tokenizer(config: BuildConfig, summaries, local: Path):
    target = f"{config.output}/tokenizer.json"
    if StoragePath(target).exists():
        info = read_json(target)
        for name, identity in info["files"].items():
            data = StoragePath(identity["url"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != identity["sha256"]:
                raise ValueError("Tokenizer file changed")
            (local / name).write_bytes(data)
        return info
    texts = []
    for summary in summaries:
        quota = math.ceil(config.tokenizer_bytes * summary["bytes"] / sum(s["bytes"] for s in summaries))
        used = 0
        for text in raw_texts(summary):
            texts.append(text)
            used += len(text.encode("utf-8"))
            if used >= quota:
                break
    np.random.default_rng(config.seed).shuffle(texts)
    tokenizer = Tokenizer(BPE(byte_fallback=False, unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    LOGGER.info("Fitting 16k BPE on %d documents", len(texts))
    tokenizer.train_from_iterator(
        iter(texts),
        trainers.BpeTrainer(
            vocab_size=16384,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=list(SPECIALS),
            show_progress=False,
        ),
    )
    if tokenizer.get_vocab_size() != 16384:
        raise ValueError("BPE fitting did not reach 16384 tokens")
    for i, token in enumerate(SPECIALS):
        assert tokenizer.token_to_id(token) == i
    for text in texts[:: max(1, len(texts) // 1000)]:
        assert tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False) == text
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token=SPECIALS[0], eos_token=SPECIALS[1], pad_token=SPECIALS[7]
    )
    wrapped.save_pretrained(local)
    info = {
        "vocab_size": 16384,
        "fit_documents": len(texts),
        "fit_bytes": sum(len(text.encode("utf-8")) for text in texts),
        "files": {},
    }
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = local / name
        data = path.read_bytes()
        url = f"{config.output}/tokenizer/{path.name}"
        StoragePath(url).write_bytes(data)
        info["files"][path.name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
    write_json(target, info)
    return info


def write_cache(url, texts, tokenizer, seed=None):
    processor = TextLmDatasetFormat().build_preprocessor(tokenizer, enforce_bos=True, enforce_eos=True)
    documents, batch = [], []

    def flush():
        for row in processor(batch):
            ids = np.asarray(row["input_ids"], dtype=np.int32)
            if ids[0] != tokenizer.bos_token_id or ids[-1] != tokenizer.eos_token_id:
                raise ValueError("Missing document boundary token")
            if ids.min() < 0 or ids.max() >= 16384:
                raise ValueError("Token outside 16k vocabulary")
            documents.append(ids)
        batch.clear()

    for text in texts:
        batch.append({"text": text})
        if len(batch) == 64:
            flush()
            if len(documents) % 64000 == 0:
                LOGGER.info("tokenizing %s: %d documents", url, len(documents))
    if batch:
        flush()
    if seed is not None:
        np.random.default_rng(seed).shuffle(documents)
    lengths = [len(doc) for doc in documents]
    ledger = write_bucket_cache(url, iter(documents), lengths)
    loaded = TreeCache.load(url, EXEMPLAR)
    assert len(loaded) == len(documents)
    indices = np.unique(np.linspace(0, len(documents) - 1, min(1024, len(documents)), dtype=int))
    for index, row in zip(indices, loaded.get_batch_sync(indices.tolist()), strict=True):
        np.testing.assert_array_equal(row["input_ids"], documents[index])
    stats = {"rows": len(documents), "tokens": ledger.field_counts["input_ids"], "readback_rows": len(indices)}
    write_json(f"{url}/.stats.json", {"total_elements": stats["rows"], "total_tokens": stats["tokens"]})
    write_json(f"{url}/validated.json", stats)
    return stats


def build(config: BuildConfig, local: Path):
    if not config.output.startswith(REGION_PREFIX) or not config.source_store.startswith(REGION_PREFIX):
        raise ValueError("All large reads and writes must stay in us-central2")
    set_jagged_array_read_cache_bytes(256_000_000)
    manifest = freeze(config)
    summaries = sample(config, manifest)
    tokenizer_info = fit_tokenizer(config, summaries, local)
    tokenizer = load_tokenizer(str(local))
    assert tokenizer.vocab_size == 16384
    train = f"{config.output}/train"
    if StoragePath(f"{train}/validated.json").exists():
        train_stats = read_json(f"{train}/validated.json")
    else:
        train_stats = write_cache(
            train, (text for summary in summaries for text in raw_texts(summary)), tokenizer, config.seed
        )
    if train_stats["tokens"] < 1_600_000_000:
        raise ValueError("Cache must exceed the full 1,476,919,296-token run budget with margin")
    cache_map = {"train": train}
    eval_stats = {}
    source_components = json.loads(SOURCE_CONFIG.read_text())["data"]["components"]
    for name in manifest["evaluation"]["sources"]:
        matches = [key for key in source_components if key.startswith(name + "-")]
        if len(matches) != 1:
            raise ValueError(f"Cannot map evaluation component {name}: {matches}")
        root = f"{config.output}/evaluation/{name}"
        url = f"{root}/validation"
        if StoragePath(f"{url}/validated.json").exists():
            stats = read_json(f"{url}/validated.json")
        else:
            subset = {"sources": {name: manifest["evaluation"]["sources"][name]}}
            stats = write_cache(url, (text for _, text in eval_records(subset)), tokenizer)
        cache_map[matches[0]] = root
        eval_stats[name] = stats
        LOGGER.info("eval %s: %s", name, stats)
    if set(cache_map) != set(source_components):
        raise ValueError("Incomplete frozen-recipe cache coverage")
    baseline = {
        "status": "validated",
        "kind": manifest["baseline_kind"],
        "config": asdict(config),
        "tokenizer": tokenizer_info,
        "cache_map": cache_map,
        "train": train_stats,
        "evaluation": eval_stats,
    }
    write_json(f"{config.output}/baseline.json", baseline)
    LOGGER.info("BASELINE COMPLETE: %s/baseline.json", config.output)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-tokenizer", type=Path, default=Path("/tmp/belay-tpu-tokenizer"))
    args = parser.parse_args()
    args.local_tokenizer.mkdir(parents=True, exist_ok=True)
    build(BuildConfig(output=args.output, revision=args.revision), args.local_tokenizer)


if __name__ == "__main__":
    main()
