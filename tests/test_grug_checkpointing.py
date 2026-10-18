# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from levanter.checkpoint import Checkpointer, CheckpointInterval, latest_checkpoint_path

from experiments.grug.checkpointing import (
    init_weights_only_from_checkpoint,
    initialize_or_restore_grug_state,
    restore_grug_state_from_checkpoint,
)


@register_dataclass
@dataclass(frozen=True)
class _TestGrugState:
    step: jax.Array
    params: jax.Array
    opt_state: jax.Array
    ema_params: jax.Array | None
    pending_qb_betas: jax.Array


def _write_checkpoint_metadata(checkpoint_dir: Path, *, step: int, timestamp: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"step": step, "timestamp": timestamp, "is_temporary": True}
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _save_checkpoint(checkpoint_root: Path, state: _TestGrugState) -> str:
    checkpointer = Checkpointer(
        base_path=str(checkpoint_root),
        save_interval=timedelta(seconds=1),
        step_policies=[CheckpointInterval(every=1)],
    )
    checkpointer.on_step(tree=state, step=int(state.step), force=True)
    checkpointer.wait_until_finished()
    return latest_checkpoint_path(str(checkpoint_root))


def test_fresh_run_allocates_state_once(tmp_path: Path):
    allocations = 0

    def initialize():
        nonlocal allocations
        allocations += 1
        return {"step": jnp.array(0), "params": jnp.array([2.0, 3.0])}

    def no_template():
        raise AssertionError("A fresh run must not construct a restore template")

    state = initialize_or_restore_grug_state(
        initialize=initialize,
        restore_template=no_template,
        checkpoint_search_paths=[str(tmp_path / "absent")],
        load_checkpoint_setting=None,
        mesh=None,
        allow_partial=False,
    )
    assert allocations == 1
    assert int(state["step"]) == 0
    np.testing.assert_array_equal(state["params"], [2.0, 3.0])


def test_full_resume_preserves_sharding_and_pinned_host_without_initializing(tmp_path: Path):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    device = NamedSharding(mesh, P("data"))
    host = device.with_memory_kind("pinned_host")

    @jax.jit
    def initializer():
        return _TestGrugState(
            step=jnp.array(0),
            params=jax.device_put(jnp.zeros(8), device),
            opt_state=jax.device_put(jnp.zeros(8), host),
            ema_params=None,
            pending_qb_betas=jax.device_put(jnp.zeros(8), device),
        )

    def no_initialize():
        raise AssertionError("Full resume must not allocate a random initialized state")

    with jax.set_mesh(mesh):
        source = _TestGrugState(
            step=jnp.array(17),
            params=jax.device_put(jnp.arange(8, dtype=jnp.float32), device),
            opt_state=jax.device_put(jnp.arange(8, dtype=jnp.float32) + 20, host),
            ema_params=None,
            pending_qb_betas=jax.device_put(jnp.ones(8), device),
        )
        _save_checkpoint(tmp_path, source)
        restored = initialize_or_restore_grug_state(
            initialize=no_initialize,
            restore_template=lambda: initializer.lower().out_info,
            checkpoint_search_paths=[str(tmp_path)],
            load_checkpoint_setting=None,
            mesh=mesh,
            allow_partial=False,
        )
    assert int(restored.step) == 17
    np.testing.assert_array_equal(restored.params, source.params)
    np.testing.assert_array_equal(restored.opt_state, source.opt_state)
    assert restored.params.sharding == device
    assert restored.opt_state.sharding == host


def test_partial_resume_initializes_missing_leaves_once(tmp_path: Path):
    checkpointer = Checkpointer(
        base_path=str(tmp_path), save_interval=timedelta(seconds=1), step_policies=[CheckpointInterval(every=1)]
    )
    checkpointer.on_step(tree={"step": jnp.array(7), "params": jnp.array([4.0, 5.0])}, step=7, force=True)
    checkpointer.wait_until_finished()
    allocations = 0

    def initialize():
        nonlocal allocations
        allocations += 1
        return {"step": jnp.array(0), "params": jnp.zeros(2), "missing": jnp.array([8.0, 9.0])}

    def no_template():
        raise AssertionError("Partial resume needs concrete values for missing leaves")

    restored = initialize_or_restore_grug_state(
        initialize=initialize,
        restore_template=no_template,
        checkpoint_search_paths=[str(tmp_path)],
        load_checkpoint_setting=None,
        mesh=None,
        allow_partial=True,
    )
    assert allocations == 1
    assert int(restored["step"]) == 7
    np.testing.assert_array_equal(restored["params"], [4.0, 5.0])
    np.testing.assert_array_equal(restored["missing"], [8.0, 9.0])


@pytest.mark.parametrize("metadata", ['{"step": 4}', "unreadable-json"])
def test_existing_broken_checkpoint_never_initializes_fresh(tmp_path: Path, metadata: str):
    directory = tmp_path / "step-4"
    directory.mkdir()
    (directory / "metadata.json").write_text(metadata)

    def no_initialize():
        raise AssertionError("An existing broken checkpoint must not start from scratch")

    def missing_data(*args, **kwargs):
        raise FileNotFoundError("checkpoint arrays missing")

    with pytest.raises(FileNotFoundError, match="Could not load"):
        initialize_or_restore_grug_state(
            initialize=no_initialize,
            restore_template=lambda: {"params": jax.ShapeDtypeStruct((8,), jnp.float32)},
            checkpoint_search_paths=[str(tmp_path)],
            load_checkpoint_setting=None,
            mesh=None,
            allow_partial=False,
            _load_fn=missing_data,
        )


@pytest.mark.parametrize(
    ("additional_weight_fields", "loads_router_state"),
    [
        ((), False),
        (("pending_qb_betas",), True),
    ],
    ids=["base", "moe"],
)
def test_initialize_from_loads_only_phase_weight_state(
    tmp_path: Path,
    additional_weight_fields: tuple[str, ...],
    loads_router_state: bool,
):
    source_state = _TestGrugState(
        step=jnp.array(100),
        params=jnp.array([10, 20]),
        opt_state=jnp.array([30, 40]),
        ema_params=jnp.array([50, 60]),
        pending_qb_betas=jnp.array([70, 80]),
    )
    source_root = tmp_path / "source"
    _save_checkpoint(source_root, source_state)
    fresh_state = _TestGrugState(
        step=jnp.array(0),
        params=jnp.array([1, 2]),
        opt_state=jnp.array([3, 4]),
        ema_params=jnp.array([5, 6]),
        pending_qb_betas=jnp.array([7, 8]),
    )

    initialized = init_weights_only_from_checkpoint(
        fresh_state,
        str(source_root),
        mesh=None,
        allow_partial=False,
        additional_weight_fields=additional_weight_fields,
    )

    assert int(initialized.step) == 0
    assert jnp.array_equal(initialized.params, source_state.params)
    assert jnp.array_equal(initialized.opt_state, fresh_state.opt_state)
    assert jnp.array_equal(initialized.ema_params, source_state.params)
    expected_router_state = source_state.pending_qb_betas if loads_router_state else fresh_state.pending_qb_betas
    assert jnp.array_equal(initialized.pending_qb_betas, expected_router_state)


def test_initialize_from_preserves_own_run_resume_precedence(tmp_path: Path):
    source_state = _TestGrugState(
        step=jnp.array(100),
        params=jnp.array([10, 20]),
        opt_state=jnp.array([30, 40]),
        ema_params=None,
        pending_qb_betas=jnp.array([50, 60]),
    )
    source_checkpoint = _save_checkpoint(tmp_path / "source", source_state)
    own_state = _TestGrugState(
        step=jnp.array(5),
        params=jnp.array([11, 21]),
        opt_state=jnp.array([31, 41]),
        ema_params=None,
        pending_qb_betas=jnp.array([51, 61]),
    )
    own_checkpoint_root = tmp_path / "own"
    _save_checkpoint(own_checkpoint_root, own_state)
    fresh_state = _TestGrugState(
        step=jnp.array(0),
        params=jnp.array([1, 2]),
        opt_state=jnp.array([3, 4]),
        ema_params=None,
        pending_qb_betas=jnp.array([5, 6]),
    )

    resumed = restore_grug_state_from_checkpoint(
        fresh_state,
        checkpoint_search_paths=[str(own_checkpoint_root)],
        load_checkpoint_setting=None,
        mesh=None,
        allow_partial=False,
    )
    initialized = init_weights_only_from_checkpoint(
        resumed,
        source_checkpoint,
        mesh=None,
        allow_partial=False,
        additional_weight_fields=("pending_qb_betas",),
    )

    assert int(initialized.step) == 5
    assert jnp.array_equal(initialized.params, own_state.params)
    assert jnp.array_equal(initialized.opt_state, own_state.opt_state)
    assert jnp.array_equal(initialized.pending_qb_betas, own_state.pending_qb_betas)


def test_restore_prefers_highest_step_over_latest_timestamp(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint_metadata(checkpoint_root / "step-100", step=100, timestamp="2026-03-17T00:00:00")
    _write_checkpoint_metadata(checkpoint_root / "step-20", step=20, timestamp="2026-03-17T12:00:00")

    attempted: list[str] = []

    def fake_load(state, path, *, axis_mapping, mesh, allow_partial):
        attempted.append(path)
        return {"loaded_from": path}

    loaded = restore_grug_state_from_checkpoint(
        {"state": "init"},
        checkpoint_search_paths=[str(checkpoint_root)],
        load_checkpoint_setting=True,
        mesh=None,
        allow_partial=False,
        _load_fn=fake_load,
    )

    assert attempted == [str(checkpoint_root / "step-100")]
    assert loaded == {"loaded_from": str(checkpoint_root / "step-100")}


def test_restore_falls_back_to_older_checkpoint_when_latest_fails(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint_metadata(checkpoint_root / "step-100", step=100, timestamp="2026-03-17T10:00:00")
    _write_checkpoint_metadata(checkpoint_root / "step-90", step=90, timestamp="2026-03-17T09:00:00")

    attempted: list[str] = []

    def fake_load(state, path, *, axis_mapping, mesh, allow_partial):
        attempted.append(path)
        if path.endswith("step-100"):
            raise FileNotFoundError(path)
        return {"loaded_from": path}

    loaded = restore_grug_state_from_checkpoint(
        {"state": "init"},
        checkpoint_search_paths=[str(checkpoint_root)],
        load_checkpoint_setting=None,
        mesh=None,
        allow_partial=False,
        _load_fn=fake_load,
    )

    assert attempted == [
        str(checkpoint_root / "step-100"),
        str(checkpoint_root / "step-100"),
        str(checkpoint_root / "step-90"),
    ]
    assert loaded == {"loaded_from": str(checkpoint_root / "step-90")}


def test_restore_raises_when_required_and_no_checkpoint_loads(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint_metadata(checkpoint_root / "step-100", step=100, timestamp="2026-03-17T10:00:00")

    def fake_load(state, path, *, axis_mapping, mesh, allow_partial):
        raise FileNotFoundError(path)

    with pytest.raises(FileNotFoundError, match="Could not load a checkpoint"):
        restore_grug_state_from_checkpoint(
            {"state": "init"},
            checkpoint_search_paths=[str(checkpoint_root)],
            load_checkpoint_setting=True,
            mesh=None,
            allow_partial=False,
            _load_fn=fake_load,
        )


def test_restore_raises_when_only_unreadable_checkpoints_present(tmp_path: Path):
    """When checkpoints are present but all fail to load, raise instead of silently starting from step 0."""
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint_metadata(checkpoint_root / "step-100", step=100, timestamp="2026-03-17T10:00:00")

    def fake_load(state, path, *, axis_mapping, mesh, allow_partial):
        raise FileNotFoundError(path)

    with pytest.raises(FileNotFoundError, match="none could be loaded"):
        restore_grug_state_from_checkpoint(
            {"state": "init"},
            checkpoint_search_paths=[str(checkpoint_root)],
            load_checkpoint_setting=None,
            mesh=None,
            allow_partial=False,
            _load_fn=fake_load,
        )


def test_launch_with_no_checkpoints(tmp_path: Path):
    """Launching with load_checkpoint_setting=None and no existing checkpoints starts training from scratch"""

    def fake_load(state, path, *, axis_mapping, mesh, allow_partial):
        raise FileNotFoundError(path)

    restored = restore_grug_state_from_checkpoint(
        {"state": "init"},
        checkpoint_search_paths=[str(tmp_path / "checkpoints")],
        load_checkpoint_setting=None,
        mesh=None,
        allow_partial=False,
        _load_fn=fake_load,
    )

    assert restored == {"state": "init"}


def test_restore_discovers_candidates_across_search_paths(tmp_path: Path):
    permanent_root = tmp_path / "checkpoints"
    temp_root = tmp_path / "checkpoints-temp"

    _write_checkpoint_metadata(permanent_root / "step-100", step=100, timestamp="2026-03-17T00:00:00")
    _write_checkpoint_metadata(temp_root / "step-150", step=150, timestamp="2026-03-17T06:00:00")

    attempted: list[str] = []

    def fake_load(state, path, *, axis_mapping, mesh, allow_partial):
        attempted.append(path)
        return {"loaded_from": path}

    loaded = restore_grug_state_from_checkpoint(
        {"state": "init"},
        checkpoint_search_paths=[str(permanent_root), str(temp_root)],
        load_checkpoint_setting=True,
        mesh=None,
        allow_partial=False,
        _load_fn=fake_load,
    )

    # step-150 from temp root should be preferred (highest step).
    assert attempted == [str(temp_root / "step-150")]
    assert loaded == {"loaded_from": str(temp_root / "step-150")}


def test_restore_respects_explicit_checkpoint_path_as_single_search_path(tmp_path: Path):
    permanent_root = tmp_path / "checkpoints"
    temp_root = tmp_path / "checkpoints-temp"
    explicit_checkpoint = permanent_root / "step-100"

    _write_checkpoint_metadata(explicit_checkpoint, step=100, timestamp="2026-03-17T00:00:00")
    _write_checkpoint_metadata(temp_root / "step-150", step=150, timestamp="2026-03-17T06:00:00")

    attempted: list[str] = []

    def fake_load(state, path, *, axis_mapping, mesh, allow_partial):
        attempted.append(path)
        return {"loaded_from": path}

    loaded = restore_grug_state_from_checkpoint(
        {"state": "init"},
        checkpoint_search_paths=[str(explicit_checkpoint)],
        load_checkpoint_setting=True,
        mesh=None,
        allow_partial=False,
        _load_fn=fake_load,
    )

    assert attempted == [str(explicit_checkpoint)]
    assert loaded == {"loaded_from": str(explicit_checkpoint)}


def test_restore_falls_back_from_temp_to_permanent(tmp_path: Path):
    permanent_root = tmp_path / "checkpoints"
    temp_root = tmp_path / "checkpoints-temp"

    _write_checkpoint_metadata(permanent_root / "step-100", step=100, timestamp="2026-03-17T00:00:00")
    _write_checkpoint_metadata(temp_root / "step-150", step=150, timestamp="2026-03-17T06:00:00")

    attempted: list[str] = []

    def fake_load(state, path, *, axis_mapping, mesh, allow_partial):
        attempted.append(path)
        if "step-150" in path:
            raise FileNotFoundError(path)
        return {"loaded_from": path}

    loaded = restore_grug_state_from_checkpoint(
        {"state": "init"},
        checkpoint_search_paths=[str(permanent_root), str(temp_root)],
        load_checkpoint_setting=None,
        mesh=None,
        allow_partial=False,
        _load_fn=fake_load,
    )

    # Should fall back to step-100 from permanent root
    assert loaded == {"loaded_from": str(permanent_root / "step-100")}


def test_restore_supports_legacy_wrapped_and_current_checkpoint_formats(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"

    checkpointer = Checkpointer(
        base_path=str(checkpoint_root),
        save_interval=timedelta(seconds=1),
        step_policies=[CheckpointInterval(every=1)],
    )

    template_state = {"step": jnp.array(0, dtype=jnp.int32), "value": jnp.array([0, 0], dtype=jnp.int32)}
    legacy_state = {"step": jnp.array(7, dtype=jnp.int32), "value": jnp.array([7, 7], dtype=jnp.int32)}
    current_state = {"step": jnp.array(8, dtype=jnp.int32), "value": jnp.array([8, 8], dtype=jnp.int32)}

    checkpointer.on_step(tree={"train_state": legacy_state}, step=7, force=True)
    checkpointer.wait_until_finished()

    loaded_legacy = restore_grug_state_from_checkpoint(
        template_state,
        checkpoint_search_paths=[str(checkpoint_root)],
        load_checkpoint_setting=True,
        mesh=None,
        allow_partial=False,
    )
    assert int(loaded_legacy["step"]) == 7
    assert jnp.array_equal(loaded_legacy["value"], legacy_state["value"])

    checkpointer.on_step(tree=current_state, step=8, force=True)
    checkpointer.wait_until_finished()

    loaded_current = restore_grug_state_from_checkpoint(
        template_state,
        checkpoint_search_paths=[str(checkpoint_root)],
        load_checkpoint_setting=True,
        mesh=None,
        allow_partial=False,
    )
    assert int(loaded_current["step"]) == 8
    assert jnp.array_equal(loaded_current["value"], current_state["value"])


def test_a_refused_candidate_aborts_the_restore_instead_of_falling_back(tmp_path: Path):
    """A template hook's refusal must not quietly skip to an older checkpoint.

    Falling back would resume from stale state while the newest checkpoint sits unread, which is
    exactly the silent outcome the hook exists to prevent.
    """
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint_metadata(checkpoint_root / "step-100", step=100, timestamp="2026-03-17T10:00:00")
    _write_checkpoint_metadata(checkpoint_root / "step-90", step=90, timestamp="2026-03-17T09:00:00")

    def refuse_newest(candidate: str):
        if candidate.endswith("step-100"):
            raise ValueError("wrong layout")
        return {"state": "init"}

    with pytest.raises(ValueError, match="wrong layout"):
        restore_grug_state_from_checkpoint(
            {"state": "init"},
            checkpoint_search_paths=[str(checkpoint_root)],
            load_checkpoint_setting=None,
            mesh=None,
            allow_partial=False,
            template_for_candidate=refuse_newest,
            _load_fn=lambda state, path, **kwargs: {"loaded_from": path},
        )


def test_an_unverifiable_candidate_is_skipped_like_an_unreadable_one(tmp_path: Path):
    """A template hook's ``FileNotFoundError`` (e.g. no manifest) falls through to an older candidate."""
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint_metadata(checkpoint_root / "step-100", step=100, timestamp="2026-03-17T10:00:00")
    _write_checkpoint_metadata(checkpoint_root / "step-90", step=90, timestamp="2026-03-17T09:00:00")

    def unverifiable_newest(candidate: str):
        if candidate.endswith("step-100"):
            raise FileNotFoundError("no manifest")
        return {"state": "init"}

    loaded = restore_grug_state_from_checkpoint(
        {"state": "init"},
        checkpoint_search_paths=[str(checkpoint_root)],
        load_checkpoint_setting=None,
        mesh=None,
        allow_partial=False,
        template_for_candidate=unverifiable_newest,
        _load_fn=lambda state, path, **kwargs: {"loaded_from": path},
    )

    assert loaded["loaded_from"].endswith("step-90")
