# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed-state base16 gradients before optimizer updates on TPU v4."""

import argparse
import functools
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import jmp
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track import model as model_module
from experiments.grug.fast_track.attn_res_saved_diagnostic import saved_attn_res_layer
from experiments.grug.fast_track.belay_tpu import SOURCE_CONFIG, base16_model_config, require_tpu_rounding
from experiments.grug.fast_track.train import _apply_qb_betas, _loss_and_grads, _router_ties_active


def _gradient_stats(reference, candidate):
    difference = jnp.abs(candidate - reference)
    return (
        jnp.max(difference, initial=0.0),
        jnp.max(jnp.abs(reference), initial=0.0),
        jnp.sum(candidate != reference, dtype=jnp.int32),
        jnp.sum(difference > 1e-5 + 1e-5 * jnp.abs(reference), dtype=jnp.int32),
        jnp.all(jnp.isfinite(reference)) & jnp.all(jnp.isfinite(candidate)),
    )


def _compare_gradients(reference, candidate, layer_map):
    named = jax.tree_util.tree_leaves_with_path(reference)
    names = [jax.tree_util.keystr(path, simple=True, separator=".") for path, _ in named]
    ref_leaves, ref_tree = jax.tree.flatten(reference)
    cand_leaves, cand_tree = jax.tree.flatten(candidate)
    if ref_tree != cand_tree or not all(eqx.is_inexact_array(x) for x in ref_leaves + cand_leaves):
        raise ValueError("Gradient structures must match and contain only inexact arrays")
    layers = [layer_map.get(name.split(".")[0], ()) for name in names]
    for array, indices in zip(ref_leaves, layers, strict=True):
        if indices and array.shape[0] != len(indices):
            raise ValueError("Stacked gradient's leading dimension differs from its layer mapping")

    def reduce_rows(ref_arrays, cand_arrays):
        return tuple(
            (_gradient_stats(a, b), jax.vmap(_gradient_stats)(a, b) if indices else None)
            for a, b, indices in zip(ref_arrays, cand_arrays, layers, strict=True)
        )

    reduced = jax.device_get(jax.jit(reduce_rows)(ref_leaves, cand_leaves))
    rows = []

    def append(name, shape, values, layer=None):
        maximum, reference_maximum, unequal, outside_tolerance, finite = values
        rows.append(
            {
                "parameter": name,
                "layer": layer,
                "shape": list(shape),
                "max_abs_difference": float(maximum),
                "max_abs_reference": float(reference_maximum),
                "unequal_count": int(unequal),
                "outside_tolerance_count": int(outside_tolerance),
                "finite": bool(finite),
            }
        )

    for name, array, indices, (whole, per_layer) in zip(names, ref_leaves, layers, reduced, strict=True):
        append(name, array.shape, whole)
        if indices:
            assert per_layer is not None
            for index, layer in enumerate(indices):
                append(name, array.shape[1:], tuple(value[index] for value in per_layer), layer)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        parser.error("Artifacts must remain in us-central2")
    require_tpu_rounding()
    source = json.loads(SOURCE_CONFIG.read_text())
    outer = source["trainer"]
    trainer = outer["trainer"]
    for key, value in trainer["jax_config"].items():
        jax.config.update(key, value)
    DistributedConfig().initialize()
    devices = jax.devices()
    if len(devices) != 16 or any(d.platform != "tpu" or "v4" not in d.device_kind for d in devices):
        raise RuntimeError("This diagnostic requires 16 TPU v4 chips")
    config = base16_model_config()
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    rank = jax.process_index()
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / f"gradient-rank-{rank}"
    output.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(f"{args.artifact_prefix.rstrip('/')}/rank-{rank}")
    report = {
        "purpose": "fixed-state derivative localization; no optimizer updates or throughput claim",
        "git_sha": args.revision,
        "candidate_origin": "d1a10c8d4",
        "shape": [128, 4096, 512],
        "seed": trainer["seed"],
        "data_seed": 1,
        "dtype": str(policy),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_config": trainer["jax_config"],
        "comparison": "global numerical equality, not bitwise identity; distributed reductions",
        "tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "status": "initializing",
        "error": None,
    }

    def save_report():
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        fs.put_file(str(path), f"{prefix}/report.json")

    save_report()
    try:
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=16, replica_axis_size=1)):
            key = jax.random.split(jax.random.PRNGKey(trainer["seed"]), 2)[1]
            model = eqx.filter_jit(lambda key: policy.cast_to_param(model_module.Transformer.init(config, key=key)))(key)
            stack_names = ("stacked_blocks", "kda_blocks", "stacked_blocks_tail", "kda_blocks_tail")
            present = [name for name in stack_names if getattr(model, name) is not None]
            layer_map = dict(zip(present, model.stack_layer_indices(), strict=True))
            report["layer_map"] = layer_map
            report["kda_xla_scan_impl"] = config.kda_xla_scan_impl
            report["inner_attention_remat"] = config.attn_res_remat_attention
            spec = P(("replica_dcn", "data", "expert"), None)
            tokens = jax.sharding.reshard(
                jax.random.randint(jax.random.key(1), (128, 4096), 0, config.vocab_size), spec
            )
            segments = jax.sharding.reshard(jnp.broadcast_to(jnp.arange(4096) // 1024, tokens.shape), spec)
            batch = jax.vmap(GrugLmExample.causal)(tokens, segment_ids=segments)
            pending = jnp.zeros((config.num_layers, config.qb_num_experts))
            z_loss = outer["z_loss_weight"] if outer["z_loss_weight"] > 0 else None
            router_ties = _router_ties_active(config, 0)

            def make_evaluator():
                def evaluate(params, batch, pending):
                    effective = _apply_qb_betas(params, pending)
                    return _loss_and_grads(
                        effective, batch, policy, z_loss, jnp.array(0, jnp.int32), router_tie_active=router_ties
                    )

                return eqx.filter_jit(evaluate)

            def compile_evaluator(name):
                report["status"] = f"{name}_compiling"
                save_report()
                print(f"Compiling {name} fixed-state gradient evaluator", flush=True)
                start = time.monotonic()
                compiled = make_evaluator().lower(model, batch, pending).compile()
                memory = compiled.compiled.memory_analysis()
                report[name] = {
                    "compile_time": time.monotonic() - start,
                    "compiler_memory_bytes": {
                        "arguments": memory.argument_size_in_bytes,
                        "outputs": memory.output_size_in_bytes,
                        "temporary": memory.temp_size_in_bytes,
                        "aliases": memory.alias_size_in_bytes,
                    },
                }
                if rank == 0:
                    hlo = output / f"{name}-optimized-hlo.txt"
                    hlo.write_text(compiled.compiled.as_text())
                    fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
                save_report()
                return compiled

            reference_fn = compile_evaluator("reference")
            start = time.monotonic()
            (reference_loss, reference_metrics), reference = jax.block_until_ready(reference_fn(model, batch, pending))
            report["reference"]["execution_time"] = time.monotonic() - start
            report["reference"]["loss"] = float(reference_loss)
            report["status"] = "reference_completed"
            save_report()
            start = time.monotonic()
            (repeat_loss, repeat_metrics), repeated = jax.block_until_ready(reference_fn(model, batch, pending))
            repeat_time = time.monotonic() - start
            control = _compare_gradients(reference, repeated, layer_map)
            report["control_repeat"] = {"loss": float(repeat_loss), "execution_time": repeat_time, "rows": control}
            report["control_exact"] = bool(jnp.isfinite(reference_loss) & (reference_loss == repeat_loss)) and all(
                row["unequal_count"] == 0 and row["finite"] for row in control
            )
            report["status"] = "control_completed"
            save_report()
            if not report["control_exact"]:
                raise RuntimeError("Reference repeat is nonfinite or not numerically exact")
            del repeated, reference_metrics, repeat_metrics
            candidate_backward = functools.partial(
                saved_attn_res_layer,
                layer_fn=model_module._attn_res_layer,
                differentiable_stats=model_module._DIFFERENTIABLE_STATS,
            )
            with patch.object(model_module, "_attn_res_layer_remat", candidate_backward):
                candidate_fn = compile_evaluator("candidate")
            start = time.monotonic()
            (candidate_loss, candidate_metrics), candidate = jax.block_until_ready(candidate_fn(model, batch, pending))
            report["candidate"]["execution_time"] = time.monotonic() - start
            report["candidate"]["loss"] = float(candidate_loss)
            report["finite_losses"] = bool(jnp.isfinite(reference_loss) & jnp.isfinite(candidate_loss))
            report["loss_exact"] = bool(reference_loss == candidate_loss)
            report["status"] = "candidate_completed"
            save_report()
            rows = _compare_gradients(reference, candidate, layer_map)
            report["gradient_rows"] = rows
            report["gradients_exact"] = all(row["unequal_count"] == 0 for row in rows)
            report["finite_gradients"] = all(row["finite"] for row in rows)
            report["status"] = "completed"
            save_report()
            print(json.dumps({k: report[k] for k in ("status", "loss_exact", "gradients_exact", "finite_gradients")}))
            del candidate_metrics
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save_report()
        raise


if __name__ == "__main__":
    main()
