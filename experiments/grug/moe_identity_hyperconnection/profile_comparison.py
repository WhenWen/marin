# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched July-baseline versus Identity-HC profiles at Gate 1 widths."""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic import MoeHeuristic as BaselineHeuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
)
from experiments.grug.moe.launch import (
    GrugMoeLaunchConfig as BaselineLaunchConfig,
)
from experiments.grug.moe.launch import (
    run_grug_moe_trial as run_baseline,
)
from experiments.grug.moe.train import GrugTrainerConfig as BaselineTrainerConfig
from experiments.grug.moe_identity_hyperconnection.heuristic import MoeHeuristic as IdentityHcHeuristic
from experiments.grug.moe_identity_hyperconnection.launch import (
    GrugMoeLaunchConfig as IdentityHcLaunchConfig,
)
from experiments.grug.moe_identity_hyperconnection.launch import (
    run_grug_moe_trial as run_identity_hc,
)
from experiments.grug.moe_identity_hyperconnection.train import GrugTrainerConfig as IdentityHcTrainerConfig

_SEQ_LEN = 8192
_TPU = "v5p-8"
_NUM_STEPS = 220
_GROUP = "MOE-JULY-IHC-perf-v2-issue-7409"
_POINTS: tuple[tuple[int, int, int], ...] = (
    (512, 16, 10_980),
    (768, 32, 16_875),
)


def _profiler() -> ProfilerConfig:
    return ProfilerConfig(enabled=True, start_step=20, num_steps=50, perfetto_link=False)


def _local_checkpointer(run_id: str) -> CheckpointerConfig:
    return CheckpointerConfig(
        base_path=f"/tmp/{run_id}/checkpoints",
        save_interval=None,
        keep=None,
        append_run_id_to_base_path=False,
    )


def _build_baseline_step(hidden_dim: int, batch_size: int, gate_steps: int) -> ExecutorStep:
    heuristic = BaselineHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(hidden_dim, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )
    tokens = float(gate_steps * batch_size * _SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)
    run_id = f"MOE-JULY-IHC-PERF2-BASE-d{hidden_dim}"
    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_baseline,
        config=BaselineLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(_NUM_STEPS),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=["moe", "july_baseline", "identity_hc_perf", "baseline", f"d{hidden_dim}"],
                group=_GROUP,
                name=None,
            ),
            optimizer=versioned(optimizer),
            profiler=versioned(_profiler()),
            grug_trainer=versioned(BaselineTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
            eval=None,
            checkpointer=versioned(_local_checkpointer(run_id)),
        ),
    )


def _build_identity_hc_step(hidden_dim: int, batch_size: int, gate_steps: int) -> ExecutorStep:
    heuristic = IdentityHcHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(hidden_dim, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
        num_residual_streams=4,
        hyperconnection_alpha_init=0.01,
        hyperconnection_remat_layers=2,
    )
    tokens = float(gate_steps * batch_size * _SEQ_LEN)
    optimizer = heuristic.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=_SEQ_LEN)
    run_id = f"MOE-JULY-IHC-PERF2-CAND-d{hidden_dim}"
    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_identity_hc,
        config=IdentityHcLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(_NUM_STEPS),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=["moe", "july_baseline", "identity_hc_perf", "candidate", f"d{hidden_dim}"],
                group=_GROUP,
                name=None,
            ),
            optimizer=versioned(optimizer),
            profiler=versioned(_profiler()),
            grug_trainer=versioned(IdentityHcTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
            eval=None,
            checkpointer=versioned(_local_checkpointer(run_id)),
        ),
    )


if __name__ == "__main__":
    steps: list[ExecutorStep] = []
    for dim, batch, gate_steps in _POINTS:
        steps.append(_build_baseline_step(dim, batch, gate_steps))
        steps.append(_build_identity_hc_step(dim, batch, gate_steps))
    executor_main(
        steps=steps,
        description="Matched v5p-8 profiles for July baseline versus four-stream Identity HC at d512/d768.",
    )
