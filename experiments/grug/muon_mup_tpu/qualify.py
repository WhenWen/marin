# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-model numerical and full optimizer-step gates on the actual training batch."""

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from levanter.kernels.pallas.fused_cross_entropy_loss import fused_cross_entropy_loss_and_logsumexp_penalty

from experiments.grug.fast_track import train
from experiments.grug.muon_mup_tpu.tpu_execution import (
    D768_CE_BLOCK_SIZES,
    execution_mode,
)

logger = logging.getLogger(__name__)
MODES = ("reference", "histogram", "tpu")
# The KDA limits were set before measurement. On October 28 the user accepted
# the measured histogram rounding differences and removed its bitwise-only gate.
HISTOGRAM_LIMITS = {"gradient_rel_l2": 1e-5, "update_rel_l2": 1e-3}
LIMITS = {"loss_abs": 2e-4, "gradient_rel_l2": 0.005, "leaf_rel_l2": 0.02, "update_rel_l2": 0.005}


class QualificationComplete(Exception):
    """Exit the native loop after the requested diagnostic, without final evaluation."""


@jax.jit
def differences(reference, candidate):
    rows = []
    for a, b in zip(jax.tree.leaves(reference), jax.tree.leaves(candidate), strict=True):
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        delta = a - b
        rows.append(
            jnp.stack(
                (
                    jnp.sum(a * a),
                    jnp.sum(b * b),
                    jnp.sum(delta * delta),
                    jnp.max(jnp.abs(delta)),
                    jnp.sum(jnp.abs(delta)),
                    jnp.asarray(a.size, jnp.float32),
                    jnp.all(jnp.isfinite(b)).astype(jnp.float32),
                )
            )
        )
    return jnp.stack(rows)


def summarize(reference, candidate):
    rows = np.asarray(jax.device_get(differences(reference, candidate)))
    names = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(reference)[0]]
    relative = np.sqrt(rows[:, 2] / np.maximum(rows[:, 0], 1e-30))
    # Tiny leaves are still reported, but use an absolute RMS floor in the per-leaf gate.
    gated_relative = np.sqrt(rows[:, 2] / np.maximum(rows[:, 0], rows[:, 5] * 1e-16))
    return {
        "relative_l2": float(np.sqrt(rows[:, 2].sum() / max(rows[:, 0].sum(), 1e-30))),
        "max_abs": float(rows[:, 3].max()),
        "mean_abs": float(rows[:, 4].sum() / rows[:, 5].sum()),
        "finite": bool(rows[:, 6].all()),
        "max_leaf_relative_l2": float(gated_relative.max()),
        "leaves": [
            {"name": name, "relative_l2": float(rel), "max_abs": float(row[3])}
            for name, rel, row in zip(names, relative, rows, strict=True)
        ],
    }


def qualify_d768_cross_entropy(persist):
    """Check TPU CE loss and both gradients at the failed shard shape against XLA."""
    if jax.default_backend() != "tpu":
        raise RuntimeError("The d768 CE qualification must run on TPU")
    persist("ce-stage.json", {"stage": "prepare_inputs"})
    # Generate diagnostic inputs on the host rather than compiling a separate
    # 98-million-element TPU random-number program before testing the CE kernel.
    hidden = jnp.asarray(np.random.default_rng(41).standard_normal((4096, 768), dtype=np.float32).astype(jnp.bfloat16))
    head_host = np.random.default_rng(42).standard_normal((768, 128256), dtype=np.float32)
    head_host *= 0.5 / np.sqrt(768)
    head = jnp.asarray(head_host.astype(jnp.bfloat16))
    del head_host
    # Cover the partial final vocabulary tile and masked tokens.
    labels = (jnp.arange(4096, dtype=jnp.int32) * 31).at[-2].set(128255)
    weight = (jnp.arange(4096) % 7 != 0).astype(jnp.float32)
    mesh = jax.sharding.get_mesh()

    def loss(x, w, implementation):
        def shard_loss(shard_x, shard_w, shard_labels, shard_weight):
            return fused_cross_entropy_loss_and_logsumexp_penalty(
                shard_x,
                shard_labels,
                shard_w,
                weight=shard_weight,
                dtype=jnp.float32,
                implementation=implementation,
                block_sizes=D768_CE_BLOCK_SIZES,
            )

        # Pallas requires manual partitioning. Replicate this diagnostic so
        # every shard retains the failing 4096-token shape, without dividing B.
        return jax.shard_map(
            shard_loss,
            mesh=mesh,
            in_specs=(jax.P(),) * 4,
            out_specs=jax.P(),
            check_vma=False,
        )(x, w, labels, weight)

    evaluate = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)), static_argnums=2)
    persist("ce-stage.json", {"stage": "xla_reference"})
    reference_loss, reference_grads = jax.block_until_ready(evaluate(hidden, head, "xla"))
    persist("ce-stage.json", {"stage": "pallas_candidate"})
    candidate_loss, candidate_grads = jax.block_until_ready(evaluate(hidden, head, "pallas_tpu"))
    gradient_reports = [summarize(a, b) for a, b in zip(reference_grads, candidate_grads, strict=True)]
    loss_abs = abs(float(candidate_loss) - float(reference_loss))
    passed = loss_abs <= LIMITS["loss_abs"] and all(
        g["finite"] and g["relative_l2"] <= LIMITS["gradient_rel_l2"] for g in gradient_reports
    )
    report = {
        "shape": [4096, 768, 128256],
        "reference": "xla",
        "loss_abs": loss_abs,
        "loss": float(candidate_loss),
        "gradients": gradient_reports,
        "passed": passed,
        "loss_abs_limit": LIMITS["loss_abs"],
        "gradient_rel_l2_limit": LIMITS["gradient_rel_l2"],
    }
    persist("ce-qualification.json", report)
    if jax.process_index() == 0:
        wandb.log({"qualification/ce/loss_abs": loss_abs, "qualification/ce/passed": int(passed)}, step=0)
    if not passed:
        raise ValueError(f"d768 CE numerical qualification failed: {report}")
    evaluate.clear_cache()


def qualify(state, batch, optimizer, mp, z_loss_weight, native_factory, persist, modes=MODES):
    if not modes or modes[0] != "reference":
        raise ValueError("Qualification modes must start with reference")
    if jax.process_index() == 0 and (wandb.run is None or wandb.run.offline or wandb.run.disabled):
        raise RuntimeError("Qualification requires online W&B before optimizer updates")
    if jax.process_index() == 0:
        wandb.log({"preflight/logging_ready": 1, "preflight/training_updates": 0}, step=0)
    report = {
        "limits": LIMITS,
        "histogram_limits": HISTOGRAM_LIMITS,
        "acceptance_policy": "user-approved histogram rounding, 2026-10-28",
        "modes": {},
        "timing": "2 warmups + 8 synchronized full steps at fixed state/batch",
    }
    persist("qualification.json", report)
    # Advance once so the checked optimizer update has nonzero warmup LR and real momentum/QB state.
    with execution_mode("reference"):
        native = native_factory(optimizer, mp, z_loss_weight=z_loss_weight)
        baseline_step = jax.jit(native.__wrapped__).lower(state, batch).compile()
        state, metrics, _ = baseline_step(state, batch)
        jax.block_until_ready(state)
        if jax.process_index() == 0:
            wandb.log({"qualification/bootstrap_loss": float(metrics["train/loss"])}, step=1)
    del native
    jax.clear_caches()
    qb_params = train._apply_qb_betas(state.params, state.pending_qb_betas)
    z_loss = z_loss_weight if z_loss_weight > 0 else None

    def gradient(params, batch):
        return train._loss_and_grads(params, batch, mp, z_loss)

    reference_grads = reference_updates = reference_aux = reference_loss = None
    update_step = None
    for index, mode in enumerate(modes):
        logger.info("QUALIFICATION numerical mode=%s", mode)
        # Clear nested jit traces, so no candidate accidentally reuses reference code.
        jax.clear_caches()
        with execution_mode(mode):
            (loss, aux), grads = jax.jit(gradient)(qb_params, batch)
            jax.block_until_ready(grads)
            if update_step is None:
                update_step = jax.jit(optimizer.update).lower(grads, state.opt_state, qb_params).compile()
            updates = update_step(grads, state.opt_state, qb_params)[0]
            jax.block_until_ready(updates)
        loss = float(loss)
        if mode == "reference":
            reference_loss, reference_aux = loss, aux
            reference_grads, reference_updates = grads, updates
            report["modes"][mode] = {"loss": loss, "numerical_pass": bool(np.isfinite(loss))}
        else:
            grad_diff = summarize(reference_grads, grads)
            update_diff = summarize(reference_updates, updates)
            aux_diff = summarize(reference_aux, aux)
            loss_diff = abs(loss - reference_loss)
            if mode == "histogram":
                passed = (
                    loss_diff == 0
                    and grad_diff["finite"]
                    and update_diff["finite"]
                    and grad_diff["relative_l2"] <= HISTOGRAM_LIMITS["gradient_rel_l2"]
                    and update_diff["relative_l2"] <= HISTOGRAM_LIMITS["update_rel_l2"]
                    and aux_diff["max_abs"] == 0
                )
            else:
                passed = (
                    loss_diff <= LIMITS["loss_abs"]
                    and grad_diff["finite"]
                    and update_diff["finite"]
                    and grad_diff["relative_l2"] <= LIMITS["gradient_rel_l2"]
                    and grad_diff["max_leaf_relative_l2"] <= LIMITS["leaf_rel_l2"]
                    and update_diff["relative_l2"] <= LIMITS["update_rel_l2"]
                )
            report["modes"][mode] = {
                "loss": loss,
                "loss_abs": loss_diff,
                "gradients": grad_diff,
                "updates": update_diff,
                "aux": aux_diff,
                "numerical_pass": bool(passed),
            }
            del grads, updates, aux
        persist("qualification.json", report)
        if jax.process_index() == 0:
            wandb.log(
                {
                    f"qualification/{mode}/loss": loss,
                    f"qualification/{mode}/numerical_pass": int(report["modes"][mode]["numerical_pass"]),
                },
                step=index + 2,
            )
    del reference_grads, reference_updates, reference_aux
    # Measure each complete step including routing, backward, NS/Adam, and QB state.
    for index, mode in enumerate(modes):
        logger.info("QUALIFICATION throughput mode=%s", mode)
        jax.clear_caches()
        with execution_mode(mode):
            native = native_factory(optimizer, mp, z_loss_weight=z_loss_weight)
            step = baseline_step if mode == "reference" else jax.jit(native.__wrapped__)
            times = []
            for iteration in range(10):
                start = time.perf_counter()
                result = step(state, batch)
                jax.block_until_ready(result)
                elapsed = time.perf_counter() - start
                if iteration >= 2:
                    times.append(elapsed)
                if jax.process_index() == 0:
                    wandb.log(
                        {
                            f"qualification/{mode}/step_seconds": elapsed,
                            f"qualification/{mode}/train_loss": float(result[1]["train/loss"]),
                        },
                        step=5 + index * 10 + iteration,
                    )
                del result
            report["modes"][mode]["median_step_seconds"] = float(np.median(times))
            report["modes"][mode]["step_seconds"] = times
            report["modes"][mode]["tokens_per_second"] = int(np.prod(batch.tokens.shape)) / np.median(times)
        del native, step
        persist("qualification.json", report)
    baseline = report["modes"]["reference"]["median_step_seconds"]
    eligible = [
        mode
        for mode in modes[1:]
        if report["modes"][mode]["numerical_pass"] and report["modes"][mode]["median_step_seconds"] < baseline * 0.9
    ]
    report["selected"] = (
        min(eligible, key=lambda mode: report["modes"][mode]["median_step_seconds"]) if eligible else "reference"
    )
    persist("qualification.json", report)
    logger.info("QUALIFICATION selected=%s", report["selected"])
    raise QualificationComplete()


def profile_steps(state, batch, optimizer, mp, z_loss_weight, native_factory, directory, count, persist):
    """Capture device operations from warmed full training steps on the real batch."""
    if jax.process_index() == 0 and (wandb.run is None or wandb.run.offline or wandb.run.disabled):
        raise RuntimeError("Profiling requires online W&B")
    if jax.process_index() == 0:
        wandb.log({"preflight/logging_ready": 1, "preflight/training_updates": 0}, step=0)
    step = native_factory(optimizer, mp, z_loss_weight=z_loss_weight)
    records = []

    def advance(state, index):
        start = time.perf_counter()
        state, metrics, _ = step(state, batch)
        jax.block_until_ready(state)
        record = {
            "update": index + 1,
            "train/loss": float(metrics["train/loss"]),
            "step_seconds": time.perf_counter() - start,
        }
        records.append(record)
        persist("profile-losses.json", records)
        logging.getLogger("levanter.json_logger").info("%s", record)
        if jax.process_index() == 0:
            wandb.log(record, step=index + 1)
        return state

    for index in range(2):
        state = advance(state, index)
    with jax.profiler.trace(str(directory)):
        for index in range(2, count + 2):
            with jax.profiler.StepTraceAnnotation("train", step_num=index):
                state = advance(state, index)
    raise QualificationComplete()
