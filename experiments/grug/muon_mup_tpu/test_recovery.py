# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from levanter.checkpoint import CheckpointerConfig

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.muon_mup_tpu.optimizer import MuonMDConfig, MuownConfig, NorMuonHConfig, RowNormalizedMuonHConfig
from experiments.grug.muon_mup_tpu.recovery import batch_fingerprint, prepare_recovery


def test_batch_fingerprint_preserves_order_across_shards():
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    values = np.arange(8 * len(jax.devices()), dtype=np.int32).reshape(-1, 8)
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    with jax.set_mesh(mesh):
        actual = batch_fingerprint(jax.device_put(values, sharding))
        reordered = batch_fingerprint(jax.device_put(values[:, ::-1].copy(), sharding))
    assert actual == {
        "shape": list(values.shape),
        "dtype": "int32",
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    # Sum and XOR checksums cannot distinguish this permutation of identical tokens.
    assert actual != reordered


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ContinuationState:
    step: jax.Array
    params: object
    opt_state: object
    rng: jax.Array


@pytest.mark.parametrize("optimizer", [MuownConfig, MuonMDConfig, NorMuonHConfig, RowNormalizedMuonHConfig])
def test_checkpoint_restores_gains_momentum_schedule_rng_and_position(tmp_path, optimizer):
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    with jax.set_mesh(mesh):
        tx = optimizer(learning_rate=0.003, warmup=0, min_lr_ratio=0.05).build(20)
        params = {"matrix": jnp.asarray(np.random.default_rng(1).normal(size=(2, 7, 4)), dtype=jnp.float32)}
        initial = ContinuationState(jnp.array(0), params, tx.init(params), jax.random.PRNGKey(3))

        @jax.jit
        def advance(state):
            key, next_key = jax.random.split(state.rng)
            # Indexed synthetic gradients exercise both a data cursor and RNG continuation.
            grads = {"matrix": jax.random.normal(key, (2, 7, 4)) + state.step * 0.01}
            updates, opt_state = tx.update(grads, state.opt_state, state.params)
            return ContinuationState(state.step + 1, optax.apply_updates(state.params, updates), opt_state, next_key)

        prefix = advance(advance(initial))
        checkpointer = CheckpointerConfig(base_path=str(tmp_path), append_run_id_to_base_path=False).create("unit")
        checkpointer.on_step(tree=prefix, step=2, force=True)
        checkpointer.wait_until_finished()
        restored = restore_grug_state_from_checkpoint(
            initial,
            checkpoint_search_paths=[str(tmp_path)],
            load_checkpoint_setting=True,
            mesh=mesh,
            allow_partial=False,
        )
        for actual, expected in zip(jax.tree.leaves(advance(restored)), jax.tree.leaves(advance(prefix)), strict=True):
            np.testing.assert_array_equal(actual, expected)
        # A missing durable checkpoint must not quietly produce the initial state.
        with pytest.raises(FileNotFoundError):
            restore_grug_state_from_checkpoint(
                initial,
                checkpoint_search_paths=[str(tmp_path / "lost")],
                load_checkpoint_setting=True,
                mesh=mesh,
                allow_partial=False,
            )


@dataclass
class TestTrainerConfig:
    __test__ = False
    seed: int = 0
    id: str = "attempt-a"


@dataclass
class TestTrainer:
    __test__ = False
    trainer: TestTrainerConfig


@dataclass
class TestConfig:
    __test__ = False
    trainer: TestTrainer
    optimizer: str = "muown"
    stop_after_steps: int | None = None


def test_recovery_contract_allows_new_attempt_but_rejects_changed_science(tmp_path):
    original = TestConfig(TestTrainer(TestTrainerConfig()))
    root = str(tmp_path)
    assert not prepare_recovery(original, root, "first")
    # A new process and a longer prefix retain the same experiment identity.
    continuation = replace(original, stop_after_steps=4, trainer=TestTrainer(TestTrainerConfig(id="attempt-b")))
    assert prepare_recovery(continuation, root, "second")
    with pytest.raises(ValueError, match="configuration/source mismatch"):
        prepare_recovery(replace(continuation, optimizer="muon_md"), root, "incompatible")
