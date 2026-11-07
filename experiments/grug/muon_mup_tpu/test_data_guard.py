# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from types import SimpleNamespace

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.store.cache import CacheLedger, ShardedCacheLayout
from levanter.store.jagged_array import _no_cache_read_context
from levanter.store.tree_store import TreeStore
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.grug.muon_mup_tpu.collect import collect_attempt
from experiments.grug.muon_mup_tpu.data_guard import (
    CacheIntegrityError,
    batch_statistics,
    check_caches,
    validate_batch_statistics,
)
from experiments.grug.muon_mup_tpu.train import persist_json_artifact


def write_cache(root):
    tokens = np.arange(1, 17, dtype=np.int32)
    store = TreeStore.open({"input_ids": tokens}, str(root), mode="w")
    store.extend([{"input_ids": tokens}])
    ledger = CacheLedger(total_num_rows=1, shard_rows={"part": 1}, is_finished=True)
    root.joinpath("shard_ledger.json").write_text(ledger.to_json())


@pytest.mark.parametrize("damage", ["delete", "replace"])
def test_guard_rejects_cache_disappearance_or_replacement_after_startup(tmp_path, damage):
    root = tmp_path / "train"
    write_cache(root)
    paths = {"train/corpus": str(root)}
    baseline = check_caches(paths)
    assert baseline["train/corpus"]["rows"] == 1
    if damage == "delete":
        shutil.rmtree(root)
    else:
        ledger_path = ShardedCacheLayout.parse(str(root)).ledger
        ledger = CacheLedger.load(str(root))
        ledger.total_num_rows = 2
        with open(ledger_path, "w") as stream:
            stream.write(ledger.to_json())
    with pytest.raises(CacheIntegrityError) as error:
        check_caches(paths, baseline)
    assert "train/corpus" in error.value.report
    assert "error" in error.value.report["train/corpus"]


def test_deleted_tensorstore_chunk_is_rejected_as_zero_fill_batch(tmp_path):
    root = tmp_path / "train"
    write_cache(root)
    token_store = TreeStore.open({"input_ids": np.zeros(1, np.int32)}, str(root), mode="r")
    original = np.asarray(token_store.tree["input_ids"].data[:16].read().result())
    statistics = jax.device_get(batch_statistics(jnp.asarray(original[None, :]), jnp.ones((1, 16))))
    validate_batch_statistics(statistics)
    assert statistics["zero_token_fraction"] == 0
    # TensorStore accepts absent chunks as unwritten zeros even though the
    # ledger and Zarr metadata still exist. Metadata checks alone cannot catch it.
    shutil.rmtree(root / "input_ids" / "data" / "c")
    with _no_cache_read_context():
        reopened = TreeStore.open({"input_ids": np.zeros(1, np.int32)}, str(root), mode="r")
        tokens = np.asarray(reopened.tree["input_ids"].data[:16].read().result())
    np.testing.assert_array_equal(tokens, np.zeros(16, np.int32))
    statistics = jax.device_get(batch_statistics(jnp.asarray(tokens[None, :]), jnp.ones((1, 16))))
    with pytest.raises(ValueError, match="missing cache chunks"):
        validate_batch_statistics(statistics)


def test_sharded_cache_guard_rejects_missing_child_metadata(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    for name in ("part-0", "part-1"):
        write_cache(root / name)
    ledger = CacheLedger(
        total_num_rows=2,
        shard_rows={"part-0": 1, "part-1": 1},
        is_finished=True,
        finished_shards=["part-0", "part-1"],
        layout="sharded",
    )
    (root / "shard_ledger.json").write_text(ledger.to_json())
    paths = {"train/hero": str(root)}
    baseline = check_caches(paths)
    assert baseline["train/hero"]["rows"] == 2
    (root / "part-1" / "input_ids" / "data" / "zarr.json").unlink()
    with pytest.raises(CacheIntegrityError):
        check_caches(paths, baseline)


def test_completed_attempt_with_invalid_data_review_is_excluded(tmp_path):
    identity = {
        "wandb_url": "https://wandb.ai/entity/project/runs/run",
        "data_sha256": "recipe",
        "seed": 0,
        "steps": 2,
    }
    config = {
        "model": {},
        "optimizer": {},
        "trainer": {"data_seed": 0, "trainer": {"train_batch_size": 16, "mp": "bf16", "jax_config": {}}},
        "eval": {},
        "data": {"components": {"paloma/test": {}, "uncheatable_eval/test": {}}},
    }
    files = {
        "identity.json": identity,
        "resolved-config.json": config,
        "complete.json": {"steps": 2, "final_evaluation_complete": True},
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    run = SimpleNamespace(
        state="finished",
        summary={"global_step": 1, "train/loss": 0.0},
        scan_history=lambda **kwargs: [{"_step": 1, "eval/paloma/test/bpb": 1.2, "eval/uncheatable_eval/test/bpb": 1.1}],
    )
    api = SimpleNamespace(run=lambda _: run)
    fs = fsspec.filesystem("file")
    assert collect_attempt(api, fs, str(tmp_path))["eligible"]
    (tmp_path / "validity-review.json").write_text(
        json.dumps({"eligible_for_optimizer_comparison": False, "reason": "Training cache was deleted"})
    )
    record = collect_attempt(api, fs, str(tmp_path))
    assert not record["eligible"]
    assert record["ineligible_reason"] == "Training cache was deleted"
    assert record["state"] == "finished"


def test_batch_diagnostics_reduce_across_explicit_data_shards():
    run_on_cpu_devices(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import pytest
        from jax.sharding import Mesh, AxisType, PartitionSpec as P
        from haliax.partitioning import set_mesh
        from experiments.grug.muon_mup_tpu.data_guard import batch_statistics, validate_batch_statistics

        mesh = Mesh(np.array(jax.devices()).reshape(1, 4, 1, 1),
                    ('replica_dcn', 'data', 'expert', 'model'), axis_types=(AxisType.Explicit,) * 4)
        tokens = np.tile(np.arange(4096, dtype=np.int32), (16, 1))
        weights = np.ones_like(tokens, dtype=np.float32)
        weights[:, -1] = 0
        with set_mesh(mesh):
            sharded_tokens = jax.sharding.reshard(jnp.asarray(tokens), P('data', None))
            sharded_weights = jax.sharding.reshard(jnp.asarray(weights), P('data', None))
            statistics = jax.device_get(batch_statistics(sharded_tokens, sharded_weights))
            validate_batch_statistics(statistics)
            assert statistics['token_count'] == 16 * 4096
            assert statistics['weight_sum'] == np.sum(weights)
            assert statistics['active_weight_fraction'] == np.mean(weights > 0)
            assert statistics['zero_token_fraction'] == np.mean(tokens == 0)
            assert statistics['repeated_token_fraction'] == 0
            # Only the last data shard contains the missing-chunk sequence.
            tokens[-1] = 0
            sharded_tokens = jax.sharding.reshard(jnp.asarray(tokens), P('data', None))
            statistics = jax.device_get(batch_statistics(sharded_tokens, sharded_weights))
            assert statistics['active_zero_sequences'] == 1
            with pytest.raises(ValueError):
                validate_batch_statistics(statistics)
        """,
        device_count=4,
    )


def test_nested_diagnostic_evidence_is_persisted_to_both_sinks(tmp_path):
    output = tmp_path / "local" / "attempt"
    destination = tmp_path / "artifacts" / "attempt"
    name = "data-integrity/step-0.json"
    value = {"step": 0, "passed": True, "caches": {"train/corpus": {"rows": 1}}}
    persist_json_artifact(output, str(destination), name, value)
    local_bytes = (output / name).read_bytes()
    assert json.loads(local_bytes) == value
    assert (destination / name).read_bytes() == local_bytes
    # A later failure replaces the same evidence consistently in both sinks.
    value["passed"] = False
    persist_json_artifact(output, str(destination), name, value)
    assert json.loads((output / name).read_bytes()) == value
    assert (destination / name).read_bytes() == (output / name).read_bytes()
