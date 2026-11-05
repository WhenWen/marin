# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail closed when the frozen comparison's caches disappear or change."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.store.cache import CACHE_LAYOUT_CONSOLIDATED, CacheLedger, ShardedCacheLayout
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

CHECK_INTERVAL = 1000


class CacheIntegrityError(RuntimeError):
    def __init__(self, report):
        self.report = report
        failed = {name: row["error"] for name, row in report.items() if "error" in row}
        super().__init__(f"Frozen dataset cache integrity failed: {failed}")


def required_caches(data: LmDataConfig) -> dict[str, str]:
    """Resolve the frozen recipe's positive-weight train and zero-weight eval caches."""
    if not isinstance(data.train_weights, dict):
        raise ValueError("The comparison requires its frozen constant mixture weights")
    paths = {}
    for name, component in data.components.items():
        if not isinstance(component, DatasetComponent) or component.cache_dir is None or component.flat_cache:
            raise ValueError(f"Expected an explicit split cache for {name}")
        # In this recipe all zero-weight components are the Paloma/uncheatable
        # evaluation suites; training components need not have validation caches.
        split = "train" if data.train_weights.get(name, 0) > 0 else "validation"
        paths[f"{split}/{name}"] = str(StoragePath(component.cache_dir) / split)
    return paths


def _read_metadata(path: str) -> bytes:
    fs, inner = url_to_fs(path)
    fs.invalidate_cache(inner)
    return StoragePath(path).read_bytes()


def _cache_fingerprint(root: str) -> dict:
    ledger_path = ShardedCacheLayout.parse(root).ledger
    content = _read_metadata(ledger_path)
    ledger = CacheLedger.from_json(content.decode())
    if not ledger.is_finished or ledger.total_num_rows <= 0:
        raise ValueError(f"Cache is incomplete or empty: {ledger_path}")
    if ledger.layout != CACHE_LAYOUT_CONSOLIDATED:
        raise ValueError(f"Expected the frozen consolidated cache layout: {root}")
    objects = {ledger_path: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}}
    # These metadata objects define the flat token array and row offsets read by
    # JaggedArrayStore. Reading them again also catches a deleted cache prefix.
    for field in ("data", "offsets"):
        path = str(StoragePath(root) / "input_ids" / field / "zarr.json")
        content = _read_metadata(path)
        metadata = json.loads(content)
        objects[path] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "shape": metadata["shape"],
        }
    return {"root": root, "rows": ledger.total_num_rows, "objects": objects}


def check_caches(paths: dict[str, str], expected: dict | None = None) -> dict:
    """Read required metadata afresh and reject missing or changed cache identities."""
    report = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {name: pool.submit(_cache_fingerprint, root) for name, root in paths.items()}
        for name, future in futures.items():
            try:
                row = future.result()
                if expected is not None and row != expected[name]:
                    raise ValueError(f"Cache metadata changed since startup: {paths[name]}")
                report[name] = row
            except (OSError, ValueError, KeyError) as exc:
                report[name] = {"root": paths[name], "error": str(exc)}
    if any("error" in row for row in report.values()):
        raise CacheIntegrityError(report)
    return report


@jax.jit
def batch_statistics(tokens, loss_weight):
    """Reduce batch diagnostics without copying token contents off the device."""
    active = loss_weight > 0
    return {
        "token_count": jnp.asarray(tokens.size),
        "zero_token_fraction": jnp.mean(tokens == 0),
        "repeated_token_fraction": jnp.mean(tokens[..., 1:] == tokens[..., :-1]),
        "active_weight_fraction": jnp.mean(active),
        "weight_sum": jnp.sum(loss_weight),
        "finite_weights": jnp.all(jnp.isfinite(loss_weight)),
        "negative_weights": jnp.sum(loss_weight < 0),
        "active_zero_sequences": jnp.sum(jnp.all(tokens == 0, axis=-1) & jnp.any(active, axis=-1)),
    }


def validate_batch_statistics(statistics):
    """Reject unwritten TensorStore fill sequences and unusable loss weights."""
    if (
        not statistics["finite_weights"]
        or statistics["negative_weights"] > 0
        or statistics["weight_sum"] <= 0
        or statistics["active_zero_sequences"] > 0
    ):
        raise ValueError(f"Invalid training batch; possible missing cache chunks: {statistics}")
