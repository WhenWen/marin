# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AMUSE + MuonH launcher for the May-arch MoE (env-var driven), d=512.

Builds the **May-arch** MoE model via ``MoeMuonHHeuristic`` (heuristic_v2) and a
MuonH base optimizer, then wraps it with our schedule-free ``GrugMoeAmuseConfig``
(time-varying Schedule-Free; arxiv 2605.22432). The AMUSE X (averaged) sequence
is mirrored into ``ema_params`` so the X-model eval is logged as
``eval/ema/paloma/c4_en/loss`` (eval_ema=True).

Mirrors ``amuse_sweep_d512.py`` from the kitchen-sink checkout, but targets the
May-arch model/heuristic and TARGET's ``GrugMoeLaunchConfig`` / ``run_grug_moe_trial``
/ ``executor_main`` launch path (there is no ``direct_launch`` here).

Hyperparameter axes (env vars):

  - ``AMUSE_BETA``         β_1 / constant β. Default 0.9.
  - ``AMUSE_C_WARMUP``     c-warmup steps. Default 400.
  - ``AMUSE_LR_MULT``      peak LR multiplier vs heuristic. Default 1.0.
  - ``AMUSE_WARMUP``       LR warmup fraction. Default 0.01.
  - ``AMUSE_R_WEIGHTING``  r in w_t = max_lr^p · t^r. Default 0.0.
  - ``AMUSE_BETA_SCHEDULE`` "constant" | "amuse" | "sfplus". Default "constant".
  - ``AMUSE_T0``           T_0 for the "amuse" growing-β schedule. Default 500.
  - ``AMUSE_RHO``          β-growth rate for the "amuse" schedule. Default 0.8.
  - ``AMUSE_LR_POWER``     if set, power-law LR decay (lr~token^-power). Default unset.
  - ``AMUSE_SEED``         RNG seed. Default 0.
  - ``AMUSE_BATCH`` / ``AMUSE_STEPS``  direct (batch, steps) override for a long
        "hero" run; base LR is recomputed at the ORIGINAL budget's token count
        (budget-independence). Both must be set together. Default off.
  - ``AMUSE_BUDGET`` / ``AMUSE_TARGET_STEPS``  compute budget + target steps used
        by the heuristic to size (batch, steps). Defaults = the d512 cell.
  - ``AMUSE_CKPT_KEEP_EVERY``  permanent-checkpoint interval. Default 1000.
  - ``AMUSE_TPU`` / ``AMUSE_PREEMPTIBLE`` / ``AMUSE_ZONE`` / ``AMUSE_REGION``  TPU
        capacity for the TRAINING task (set on its ResourceConfig, not the parent).
  - ``AMUSE_RUN_TAG``      wandb run/group suffix. Default "anchor".
  - ``AMUSE_SWEEP_GROUP``  wandb group name. Default "amuse-mayarch-r1".

Submit:

    AMUSE_BETA=0.95 AMUSE_RUN_TAG="beta-0.95" AMUSE_REGION=us-east5 \\
    .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -e AMUSE_BETA -e AMUSE_C_WARMUP -e AMUSE_LR_MULT -e AMUSE_WARMUP \\
      -e AMUSE_R_WEIGHTING -e AMUSE_BETA_SCHEDULE -e AMUSE_T0 -e AMUSE_RHO \\
      -e AMUSE_LR_POWER -e AMUSE_SEED -e AMUSE_BATCH -e AMUSE_STEPS \\
      -e AMUSE_BUDGET -e AMUSE_TARGET_STEPS -e AMUSE_CKPT_KEEP_EVERY \\
      -e AMUSE_TPU -e AMUSE_PREEMPTIBLE -e AMUSE_ZONE -e AMUSE_REGION \\
      -e AMUSE_RUN_TAG -e AMUSE_SWEEP_GROUP \\
      -- python -m experiments.grug.moe.amuse_mayarch_d512
"""

import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic_v2 import SEQ_LEN, MoeMuonHHeuristic, build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    _resolve_run_id,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeAmuseConfig, PowerLawTokenDecay
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_HIDDEN_DIM: int = 512
# Budget + target-steps drive the heuristic's (batch, steps) selection. Default
# = the d512 compute-optimal cell from launch.py (bs=32, steps=10_980).
_BUDGET: float = float(os.environ.get("AMUSE_BUDGET", "3.82e17"))
_TARGET_STEPS: int = int(os.environ.get("AMUSE_TARGET_STEPS", str(2**14)))

# Capacity for the TRAINING task (not the launcher). AMUSE_PREEMPTIBLE=0 routes the
# training sub-task to the reserved pool; AMUSE_ZONE/AMUSE_REGION pin placement so
# data + checkpoints stay in-region (region alone is needed when a zone is pinned —
# a zone pin without the matching region is unschedulable).
_TPU: str = os.environ.get("AMUSE_TPU", "v5p-8")
_AMUSE_PREEMPTIBLE: bool = os.environ.get("AMUSE_PREEMPTIBLE", "1") not in ("0", "false", "False")
_AMUSE_ZONE: str = os.environ.get("AMUSE_ZONE", "")
_AMUSE_REGION: str = os.environ.get("AMUSE_REGION", "")

# AMUSE recipe HPs.
_AMUSE_BETA: float = float(os.environ.get("AMUSE_BETA", "0.9"))
_AMUSE_C_WARMUP: int = int(os.environ.get("AMUSE_C_WARMUP", "400"))
_AMUSE_LR_MULT: float = float(os.environ.get("AMUSE_LR_MULT", "1.0"))
_AMUSE_WARMUP: float = float(os.environ.get("AMUSE_WARMUP", "0.01"))
_AMUSE_R_WEIGHTING: float = float(os.environ.get("AMUSE_R_WEIGHTING", "0.0"))
# β_t schedule: "constant" (β=β1), "amuse" (β1 until T0 then grows toward 1 at
# rate rho — budget-independent since T0 is an absolute step count), or "sfplus".
_AMUSE_BETA_SCHEDULE: str = os.environ.get("AMUSE_BETA_SCHEDULE", "constant")
_AMUSE_T0: int = int(os.environ.get("AMUSE_T0", "500"))
_AMUSE_RHO: float = float(os.environ.get("AMUSE_RHO", "0.8"))
_AMUSE_WD: float = float(os.environ.get("AMUSE_WEIGHT_DECAY", "0.0"))  # decoupled WD on the AdamW group
# Curvature-corrected Muon (0 = plain MuonH). Riemannian inner-solve on the Muon direction.
_CURV_LAMBDA: float = float(os.environ.get("CURV_LAMBDA", "0.0"))
_CURV_K: int = int(os.environ.get("CURV_K", "10"))
_CURV_MAXBT: int = int(os.environ.get("CURV_MAXBT", "10"))
_CURV_TWO_SIDED: bool = os.environ.get("CURV_TWO_SIDED", "1") not in ("0", "false", "False")
_CURV_CONSTRAINT: str = os.environ.get("CURV_CONSTRAINT", "stiefel")
_AMUSE_SEED: int = int(os.environ.get("AMUSE_SEED", "0"))
# If set, replace the constant LR with a power-law decay (lr~token^-power, anchored at peak).
_AMUSE_LR_POWER_ENV: str = os.environ.get("AMUSE_LR_POWER", "")
_AMUSE_LR_POWER: float | None = float(_AMUSE_LR_POWER_ENV) if _AMUSE_LR_POWER_ENV else None
_AMUSE_Z_NORMALIZE: bool = os.environ.get("AMUSE_Z_NORMALIZE", "1") not in ("0", "false", "False")
_AMUSE_Y_NORMALIZE: bool = os.environ.get("AMUSE_Y_NORMALIZE", "0") not in ("0", "false", "False")
_RUN_TAG: str = os.environ.get("AMUSE_RUN_TAG", "anchor")
_SWEEP_GROUP: str = os.environ.get("AMUSE_SWEEP_GROUP", "amuse-mayarch-r1")
# Direct (batch, steps) override for the long "hero" run: exact control over batch
# and duration, with the token-scaled base LR recomputed from the heuristic at the
# ORIGINAL budget's token count (budget-independence — only the batch changes).
_AMUSE_BATCH: int = int(os.environ.get("AMUSE_BATCH", "0"))
_AMUSE_STEPS: int = int(os.environ.get("AMUSE_STEPS", "0"))
# Permanent-checkpoint interval. Raise for long runs to avoid writing many multi-GB
# checkpoints to GCS (storage is a major cost driver).
_AMUSE_CKPT_KEEP_EVERY: int = int(os.environ.get("AMUSE_CKPT_KEEP_EVERY", "1000"))

_HEURISTIC = MoeMuonHHeuristic()


def _build_launch() -> tuple[str, GrugMoeLaunchConfig]:
    # build_from_heuristic returns an AdamH config; we only need its (model, batch,
    # steps) sizing and rebuild the base as a MuonH config below (heuristic_v2's
    # build_muonh_config) so AMUSE wraps the May Recipe MuonH base optimizer.
    model, _adamh_base, batch_size, num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )
    # Tokens used to scale the heuristic LR. By default the heuristic's own token
    # count for the selected (batch, steps).
    tokens = float(batch_size * SEQ_LEN * num_steps)

    if _AMUSE_BATCH and _AMUSE_STEPS:
        # Budget-INDEPENDENT LR: hold the heuristic's token count at the ORIGINAL
        # (batch, steps) — where the recipe was validated — so the base LR is
        # unchanged across durations; only the batch changes (μP). Using the hero's
        # long token count would scale LR down with the budget (compute-optimal
        # scaling) = budget-dependent, breaking the recipe's validated performance.
        orig_tokens = float(batch_size * SEQ_LEN * num_steps)
        batch_size = _AMUSE_BATCH
        num_steps = _AMUSE_STEPS
        tokens = orig_tokens

    base_optimizer = _HEURISTIC.build_muonh_config(batch_size, tokens, _HIDDEN_DIM, seq_len=SEQ_LEN)

    # Wrap the MuonH base with AMUSE. β constant by default; c-warmup configurable;
    # LR constant after warmup (AMUSE removes the decay phase) unless AMUSE_LR_POWER.
    optimizer = GrugMoeAmuseConfig(
        learning_rate=base_optimizer.learning_rate * _AMUSE_LR_MULT,
        adam_lr=base_optimizer.adam_lr * _AMUSE_LR_MULT,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=_AMUSE_WARMUP,
        weight_decay=_AMUSE_WD,
        # Curvature-corrected Muon (0 = plain MuonH).
        curvature_lambda=_CURV_LAMBDA,
        curvature_inner_steps=_CURV_K,
        curvature_maxbt=_CURV_MAXBT,
        curvature_two_sided=_CURV_TWO_SIDED,
        curvature_constraint=_CURV_CONSTRAINT,
        # SF interpolation.
        amuse_beta1=_AMUSE_BETA,
        amuse_beta_schedule=_AMUSE_BETA_SCHEDULE,
        amuse_rho=_AMUSE_RHO,  # β-growth rate (ignored under "constant")
        amuse_warmup_steps=_AMUSE_T0,  # T0 for the "amuse" growing-β schedule
        amuse_beta_final=0.965,  # ignored under "constant"
        amuse_c_warmup_steps=_AMUSE_C_WARMUP,
        amuse_r_weighting=_AMUSE_R_WEIGHTING,
        amuse_weight_lr_power=2.0,
        amuse_z_normalize=_AMUSE_Z_NORMALIZE,
        amuse_y_normalize=_AMUSE_Y_NORMALIZE,
        # Muon internals — match the May Recipe MuonH base.
        muon_momentum=base_optimizer.momentum,
        muon_nesterov=base_optimizer.nesterov,
        backend_steps=base_optimizer.backend_steps,
        muon_epsilon=base_optimizer.muon_epsilon,
        coefficient_type=base_optimizer.coefficient_type,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,  # May Recipe 1pct-noclip
        # AMUSE default: NO LR decay (warmup + flat). If AMUSE_LR_POWER is set, use a
        # power-law decay lr~token^-power (anchored at peak) instead.
        lr_schedule=PowerLawTokenDecay(exponent=_AMUSE_LR_POWER) if _AMUSE_LR_POWER is not None else "constant",
        decay=None,
    )

    znorm_s = "Z" if _AMUSE_Z_NORMALIZE else "z"
    ynorm_s = "Y" if _AMUSE_Y_NORMALIZE else "y"
    lrpow_s = f"-lrpow{_AMUSE_LR_POWER:g}" if _AMUSE_LR_POWER is not None else ""
    wd_s = f"-wd{_AMUSE_WD:g}" if _AMUSE_WD else ""
    sched_s = f"-sch{_AMUSE_BETA_SCHEDULE}T{_AMUSE_T0}rho{_AMUSE_RHO:g}" if _AMUSE_BETA_SCHEDULE != "constant" else ""
    suffix = (
        f"sweep-{_RUN_TAG}-b{_AMUSE_BETA:.3g}-c{_AMUSE_C_WARMUP}"
        f"-lr{_AMUSE_LR_MULT:.3g}-w{_AMUSE_WARMUP:.3g}-r{_AMUSE_R_WEIGHTING:.3g}"
        f"-{znorm_s}{ynorm_s}{lrpow_s}{sched_s}"
        f"{f'-s{_AMUSE_SEED}' if _AMUSE_SEED else ''}{wd_s}"
    )
    run_id = _resolve_run_id(f"amuse-mayarch-d{_HIDDEN_DIM}-{_BUDGET:.2e}-{suffix}".replace("+", ""))
    name = f"grug/amuse-mayarch-d{_HIDDEN_DIM}-{suffix}"

    launch = GrugMoeLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=ResourceConfig.with_tpu(
            _TPU,
            preemptible=_AMUSE_PREEMPTIBLE,
            **({"zone": _AMUSE_ZONE} if _AMUSE_ZONE else {}),
            **({"regions": [_AMUSE_REGION]} if _AMUSE_REGION else {}),
        ),
        steps=versioned(num_steps),
        batch_size=versioned(batch_size),
        seed=versioned(_AMUSE_SEED),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="marin-community",
            project="marin_moe",
            tags=[
                "moe",
                "amuse",
                "amuse_mayarch",
                "may",
                f"d{_HIDDEN_DIM}",
                _SWEEP_GROUP,
                f"tag={_RUN_TAG}",
            ],
            group=_SWEEP_GROUP,
            name=None,
        ),
        optimizer=versioned(optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
                ema_beta=None,  # AMUSE supplies the X eval model; no separate EMA.
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=500,
                max_eval_batches=8,
                eval_current=True,  # Y eval
                eval_ema=True,  # X eval (the AMUSE averaged sequence)
            )
        ),
        checkpoint_keep_every=_AMUSE_CKPT_KEEP_EVERY,
    )
    return name, launch


def build_step() -> ExecutorStep:
    name, launch = _build_launch()
    return ExecutorStep(name=name, fn=run_grug_moe_trial, config=launch)


if __name__ == "__main__":
    executor_main(
        steps=[build_step()],
        description=(
            f"AMUSE+MuonH May-arch d{_HIDDEN_DIM} — tag={_RUN_TAG!r} "
            f"beta={_AMUSE_BETA} c_warm={_AMUSE_C_WARMUP} LRx{_AMUSE_LR_MULT} warmup={_AMUSE_WARMUP}"
        ),
    )
