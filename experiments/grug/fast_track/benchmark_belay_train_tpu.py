# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure native base16 train steps on fixed random tokens, not LM quality.

This includes optimizer updates, router balancing, EMA storage and inline
watch metrics. It excludes data loading, checkpoints and held-out evaluation.
"""

import argparse
import atexit
import gc
import json
import math
import os
import statistics
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import draccus
import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import jmp
import matplotlib as mpl
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from levanter.callbacks.profiler import ProfileOptionsConfig
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig, _finalize_iris_jax_after_clean_exit, _unregister_iris_exit_handler
from levanter.grug._moe import ep_ragged_all_to_all
from levanter.grug.sharding import compact_grug_mesh
from levanter.optim.config import OptimizerConfig

from experiments.grug.fast_track import model as model_module
from experiments.grug.fast_track import qb_balancing
from experiments.grug.fast_track.belay_flops import training_flops_per_token
from experiments.grug.fast_track.belay_tpu import SOURCE_CONFIG, base16_model_config, require_tpu_rounding
from experiments.grug.fast_track.benchmark_kda_state_tpu import tpu_recurrence
from experiments.grug.fast_track.benchmark_projection_tpu import gathered_projection
from experiments.grug.fast_track.model import AttnResLayerBackward
from experiments.grug.fast_track.train import _flat_train_step, _make_train_step, _router_ties_active, initial_state
from experiments.grug.moe import kda

mpl.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class StepMeasurement:
    step: int
    phase: str
    loss: float
    seconds: float
    dropped_assignments: tuple[int, ...]
    useful_matrix_flops_per_token: float


def _independent_dispatch(source: jax.Array, previous_residuals: tuple[jax.Array, ...]) -> jax.Array:
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4, help="Timed steps after two warmup steps and one profiled step.")
    parser.add_argument(
        "--attention-remat",
        choices=("on", "off", "save-all"),
        default="on",
        help="Diagnostic inner attention checkpoint and residual-saving policy.",
    )
    parser.add_argument(
        "--ema-phase",
        choices=("early", "active"),
        default="early",
        help="Active enables EMA from the first diagnostic update; it does not simulate a trained checkpoint.",
    )
    parser.add_argument(
        "--layer-backward",
        choices=("production", "recompute", "save"),
        default="production",
        help="Diagnostic outer layer residual policy; inner attention checkpointing is separate.",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--kda-scan", choices=("production", "reference", "stacked", "stacked-split"), default="production"
    )
    parser.add_argument("--accepted-mask", choices=("production", "reference", "inverse-direct"), default="production")
    parser.add_argument("--kda-state", choices=("production", "pallas"), default="production")
    parser.add_argument("--projection-weights", choices=("production", "gather"), default="production")
    parser.add_argument("--moe-dispatch", choices=("serialized", "overlap"), default="serialized")
    parser.add_argument("--qb-quantile", choices=("reference", "block128"), default="reference")
    parser.add_argument(
        "--job-completion",
        choices=("on-exit", "deferred"),
        default="on-exit",
        help="Defer Iris job completion to a later process in a benchmark sequence; still shut down distributed JAX.",
    )
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    if args.steps < 1 or not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        parser.error("Use positive steps and a unique us-central2 GCS artifact prefix")
    require_tpu_rounding()
    source = json.loads(SOURCE_CONFIG.read_text())
    outer = source["trainer"]
    trainer = outer["trainer"]
    for key, value in trainer["jax_config"].items():
        jax.config.update(key, value)
    DistributedConfig().initialize()
    if args.job_completion == "deferred":
        _unregister_iris_exit_handler()
        atexit.register(_finalize_iris_jax_after_clean_exit)
    devices = jax.devices()
    if len(devices) != 16 or any(d.platform != "tpu" or "v4" not in d.device_kind for d in devices):
        raise RuntimeError("This frozen full-batch benchmark requires 16 TPU v4 chips")
    config = replace(
        base16_model_config(),
        attn_res_remat_attention=args.attention_remat != "off",
        attn_res_remat_attention_save_all=args.attention_remat == "save-all",
    )
    if args.layer_backward != "production":
        config = replace(config, attn_res_layer_backward=AttnResLayerBackward[args.layer_backward.upper()])
    if args.kda_scan != "production":
        scans = {"reference": "parallel", "stacked": "parallel_stacked", "stacked-split": "parallel_stacked_split"}
        config = replace(config, kda_xla_scan_impl=scans[args.kda_scan])
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    source["optimizer"]["type"] = "grug_fast_track_muonh_v1"
    optimizer = draccus.decode(OptimizerConfig, source["optimizer"]).build(trainer["num_train_steps"])
    rank = jax.process_index()
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / Path(args.artifact_prefix).name / f"native-rank-{rank}"
    output.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(f"{args.artifact_prefix}/rank-{rank}")
    rows: list[StepMeasurement] = []
    report = {
        "kernel": "base16_native_train_step",
        "implementation": "native_grug_tpu",
        "purpose": "fixed random-token throughput diagnostic; not LM quality",
        "shape": [128, 4096, 512],
        "dtype": str(policy),
        "backend": "tpu",
        "device_type": devices[0].device_kind,
        "device_count": len(devices),
        "block_sizes": {"inkling_query": 128},
        "attn_res_layer_backward": config.attn_res_layer_backward.value,
        "attn_res_remat_attention": config.attn_res_remat_attention,
        "qb_quantile": args.qb_quantile,
        "attn_res_remat_attention_save_all": config.attn_res_remat_attention_save_all,
        "kda_scan": args.kda_scan,
        "kda_state": args.kda_state,
        "projection_weights": args.projection_weights,
        "moe_dispatch": args.moe_dispatch,
        "job_completion": args.job_completion,
        "kda_xla_scan_impl": config.kda_xla_scan_impl,
        "accepted_mask": args.accepted_mask,
        "git_sha": args.revision,
        "seed": trainer["seed"],
        "data_seed": 1,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
        "jax_config": trainer["jax_config"],
        "watch": trainer["watch"],
        "recipe_ema_start_step": trainer["num_train_steps"] - outer["ema_last_steps"],
        "ema_start_step": 0 if args.ema_phase == "active" else trainer["num_train_steps"] - outer["ema_last_steps"],
        "ema_phase": args.ema_phase,
        "timing_protocol": "all-host barrier before each update; block all outputs; reporting outside timer",
        "flop_convention": (
            "base16 useful matrix/convolution training FLOPs; excludes recomputation and elementwise work"
        ),
        "nominal_training_flop_breakdown": training_flops_per_token((1024,) * 4, batch_size=128),
        "peak_bf16_flops_per_chip": 275e12,
        "status": "initializing",
        "error": None,
    }

    def save_report():
        path = output / "report.json"
        path.write_text(json.dumps({**report, "steps": [asdict(row) for row in rows]}, indent=2) + "\n")
        fs.put_file(str(path), f"{prefix}/report.json")

    def save_curve():
        if rank != 0:
            return
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        ax.plot(
            [r.step for r in rows],
            [r.loss for r in rows],
            color="#1F77B4",
            marker="o",
            linewidth=2,
            markersize=4,
            label=f"base16, seed {trainer['seed']}, EMA {args.ema_phase}",
        )
        ax.set(xlabel="Optimizer update", ylabel="Random-token training loss")
        ax.grid(color="#B0B0B0", linestyle="--", alpha=0.65)
        ax.set_axisbelow(True)
        ax.legend()
        fig.tight_layout()
        for extension in ("pdf", "png"):
            path = output / f"loss-curve.{extension}"
            fig.savefig(path, dpi=220)
            fs.put_file(str(path), f"{prefix}/{path.name}")
        plt.close(fig)

    save_report()
    mesh = compact_grug_mesh(expert_axis_size=16, replica_axis_size=1)
    with jax.set_mesh(mesh):
        model_key = jax.random.split(jax.random.PRNGKey(trainer["seed"]), 2)[1]
        state = jax.jit(
            lambda key: initial_state(config, optimizer=optimizer, mp=policy, key=key, ema_beta=outer["ema_beta"])
        )(model_key)
        tokens = jax.sharding.reshard(
            jax.random.randint(jax.random.key(1), (128, 4096), 0, config.vocab_size),
            P(("replica_dcn", "data", "expert"), None),
        )
        segments = jax.sharding.reshard(
            jnp.broadcast_to(jnp.arange(4096) // 1024, tokens.shape),
            P(("replica_dcn", "data", "expert"), None),
        )
        batch = jax.vmap(GrugLmExample.causal)(tokens, segment_ids=segments)
        step_fn = _make_train_step(
            optimizer,
            policy,
            z_loss_weight=outer["z_loss_weight"],
            ema_beta=outer["ema_beta"],
            ema_start_step=report["ema_start_step"],
            watch_config=draccus.decode(WatchConfig, trainer["watch"]),
            grad_accum_microbatches=outer["grad_accum_microbatches"],
        )
        leaves, treedef = jax.tree.flatten(state)
        del state
        step_fn = _flat_train_step(step_fn, treedef)
        router_ties = _router_ties_active(config, 0)
        print("Compiling native base16 train step", flush=True)
        start = time.monotonic()
        with ExitStack() as contexts:
            if args.qb_quantile == "block128":
                contexts.enter_context(
                    patch.object(qb_balancing, "histogram_quantile", partial(qb_balancing.histogram_quantile, block_bins=128))
                )
            if args.moe_dispatch == "overlap":
                contexts.enter_context(patch.object(ep_ragged_all_to_all, "_serialize_dispatch", _independent_dispatch))
            if args.projection_weights == "gather":
                contexts.enter_context(patch.object(model_module, "_proj", gathered_projection))
            if args.kda_state == "pallas":
                contexts.enter_context(patch.object(kda, "_parallel_chunk_recurrence", tpu_recurrence))
            if args.accepted_mask != "production":
                implementation = (
                    ep_ragged_all_to_all._accepted_assignments_inverse
                    if args.accepted_mask == "inverse-direct"
                    else ep_ragged_all_to_all._accepted_assignments_scatter
                )
                contexts.enter_context(patch.object(ep_ragged_all_to_all, "_accepted_assignments", implementation))
            compiled = step_fn.lower(leaves, batch, loop_active=None, router_tie_active=router_ties).compile()
        report["compile_time"] = time.monotonic() - start
        memory = compiled.memory_analysis()
        report["compiler_memory_bytes"] = {
            "arguments": memory.argument_size_in_bytes,
            "outputs": memory.output_size_in_bytes,
            "temporary": memory.temp_size_in_bytes,
            "aliases": memory.alias_size_in_bytes,
        }
        if rank == 0:
            hlo = output / "optimized-hlo.txt"
            hlo.write_text(compiled.as_text())
            fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
        report["status"] = "compiled"
        save_report()
        gc.collect()
        gc.disable()

        def step(phase):
            nonlocal leaves
            # Per-host artifact uploads and rank-zero plotting can finish at
            # different times. Keep that skew outside the measured collectives.
            multihost_utils.sync_global_devices(f"native-step-{len(rows) + 1}")
            start = time.monotonic()
            leaves, metrics, watch = jax.block_until_ready(compiled(leaves, batch))
            elapsed = time.monotonic() - start
            loss = float(metrics["train/loss"])
            dropped = tuple(int(n) for n in jax.device_get(metrics["moe/dropped_assignments"]))
            flops = training_flops_per_token((1024,) * 4, batch_size=128, dropped_assignments=dropped)
            row = StepMeasurement(
                step=len(rows) + 1,
                phase=phase,
                loss=loss,
                seconds=elapsed,
                dropped_assignments=dropped,
                useful_matrix_flops_per_token=sum(flops.values()),
            )
            rows.append(row)
            print(json.dumps({"rank": rank, **asdict(row)}), flush=True)
            if not math.isfinite(loss):
                save_report()
                raise RuntimeError("Nonfinite loss in native train-step benchmark")
            del metrics, watch

        step("warmup")
        save_report()
        save_curve()
        step("warmup")
        save_report()
        save_curve()
        profile = output / "profile"
        options = ProfileOptionsConfig(
            host_tracer_level=1, python_tracer_level=0, enable_hlo_proto=True
        ).build_jax_profile_options()
        with jax.profiler.trace(str(profile), profiler_options=options):
            with jax.profiler.StepTraceAnnotation("base16_native_train_step", step_num=2):
                step("profile")
        save_report()
        save_curve()
        for path in profile.rglob("*"):
            if path.is_file():
                fs.put_file(str(path), f"{prefix}/profile/{path.relative_to(profile)}")
        for _ in range(args.steps):
            step("timed")
            save_report()
            save_curve()
        # Inspect parameters, optimizer buffers and EMA after timing; EMA does
        # not feed the training loss, so finite losses alone do not check it.
        finite_state = jax.jit(lambda xs: jnp.stack([jnp.all(jnp.isfinite(x)) for x in xs]).all())(leaves)
        report["finite_state"] = bool(finite_state)
        if not report["finite_state"]:
            save_report()
            raise RuntimeError("Nonfinite parameter, optimizer or EMA state")
        samples = [r.seconds for r in rows if r.phase == "timed"]
        report.update(status="completed", steady_state_time=statistics.median(samples))
        save_report()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
