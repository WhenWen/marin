# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare two logical MoE chunks with merged physical transport on TPU.

Small fixtures use independent NumPy FP64 expert products and analytic
gradients. Full-size calls measure output plus backward without training.
"""

import argparse
import json
import math
import os
import statistics
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from unittest.mock import patch

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import ml_dtypes
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug._moe import ep_ragged_all_to_all as ragged
from levanter.grug._moe.common import CapacityDrops
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.diagnose_kda_state_precision import errors
from experiments.grug.fast_track.moe_transport_tpu import coalesced_routed_experts, trimmed_chunk_plans

_NAMES = ("output", "dx", "dweights", "dup", "ddown")
_SPECS = (P("expert", None), P("expert", None), P("expert"), P("expert", None, None), P("expert", None, None))


def relu_squared(x):
    return jnp.square(jnp.maximum(x, 0))


def accepted_routes(selected, valid, experts, capacity_factor, shards):
    """Allocate each receiver's logical chunks in expert, sender, assignment order."""
    tokens, topk = selected.shape
    local_tokens, local_experts = tokens // shards, experts // shards
    physical = max(local_experts, math.ceil(capacity_factor * local_tokens * topk))
    logical = min(physical, max(local_experts, math.ceil(int(valid.sum()) * topk * capacity_factor / shards)))
    chunk_capacity = max(local_experts // 2, (logical + 1) // 2)
    accepted = np.zeros_like(selected, dtype=bool)
    for receiver in range(shards):
        for chunk in range(2):
            remaining = chunk_capacity
            start = receiver * local_experts + chunk * (local_experts // 2)
            for expert in range(start, start + local_experts // 2):
                for sender in range(shards):
                    sl = slice(sender * local_tokens, (sender + 1) * local_tokens)
                    indices = np.flatnonzero(((selected[sl] == expert) & valid[sl, None]).ravel())
                    kept = indices[:remaining]
                    accepted[sl].flat[kept] = True
                    remaining -= len(kept)
    return accepted


def oracle(x, weights, valid, up, down, selected, cotangent, capacity_factor, shards):
    """Evaluate represented inputs with FP64 products and explicit derivatives."""
    accepted = accepted_routes(selected, valid, up.shape[0], capacity_factor, shards)
    x, weights, up, down, cotangent = (
        np.asarray(a, np.float64)
        for a in (x, weights, up.astype(ml_dtypes.bfloat16), down.astype(ml_dtypes.bfloat16), cotangent)
    )
    output = np.zeros((len(x), down.shape[-1]), np.float64)
    dx, dw, du, dd = (np.zeros_like(a) for a in (x, weights, up, down))
    for expert in range(len(up)):
        token, route = np.nonzero((selected == expert) & accepted)
        a = x[token] @ up[expert]
        h = np.maximum(a, 0) ** 2
        y = h @ down[expert]
        scale = weights[token, route, None]
        dy = cotangent[token] * scale
        da = (dy @ down[expert].T) * (2 * np.maximum(a, 0))
        np.add.at(output, token, y * scale)
        np.add.at(dx, token, da @ up[expert].T)
        dw[token, route] = np.sum(y * cotangent[token], axis=-1)
        du[expert] = x[token].T @ da
        dd[expert] = h.T @ dy
    dropped = int(valid.sum()) * selected.shape[1] - int(accepted.sum())
    return (output, dx, dw, du, dd), dropped


def evaluate(x, weights, valid, up, down, selected, cotangent, *, mesh, capacity_factor):
    def local(a, w, mask, u, d, ids):
        return ragged._moe_mlp_ep_ragged_a2a_local(
            a,
            ids,
            w,
            mask,
            u.astype(a.dtype),
            d.astype(a.dtype),
            activation_fn=relu_squared,
            num_experts=up.shape[0],
            capacity_factor=capacity_factor,
            token_sharding_axes=("expert",),
            expert_chunks=2,
        )

    forward = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(*_SPECS, P("expert", None)),
        out_specs=(P("expert", None), CapacityDrops(P(), P())),
        # Installed TPU Megablox does not annotate Pallas output variation.
        check_vma=False,
    )
    output, pullback, drops = jax.vjp(
        lambda a, w, u, d: forward(a, w, valid, u, d, selected), x, weights, up, down, has_aux=True
    )
    return (output, *pullback(cotangent)), drops


def fixture(seed, distribution, output_width):
    rng = np.random.default_rng(seed)
    tokens, width, intermediate, experts, topk = 512, 128, 128, 64, 4
    selected = np.stack([rng.choice(experts, topk, replace=False) for _ in range(tokens)]).astype(np.int32)
    valid = np.ones(tokens, bool)
    if distribution == "imbalanced":
        selected[:] = np.arange(topk)
    elif distribution == "sparse":
        valid[::3] = False
    elif distribution == "empty":
        valid[:] = False
    x = rng.normal(size=(tokens, width)).astype(ml_dtypes.bfloat16)
    weights = rng.uniform(0.05, 0.4, size=(tokens, topk)).astype(np.float32)
    up = (rng.normal(size=(experts, width, intermediate)) / np.sqrt(width)).astype(np.float32)
    down = (rng.normal(size=(experts, intermediate, output_width)) / np.sqrt(intermediate)).astype(np.float32)
    ct = rng.normal(size=(tokens, output_width)).astype(ml_dtypes.bfloat16)
    return x, weights, valid, up, down, selected, ct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--candidate", choices=("coalesced", "trimmed"), default="coalesced")
    args = parser.parse_args()
    names = ("reference", args.candidate)
    require_tpu_rounding()
    if args.candidate == "trimmed" and (
        (jax.__version__, jaxlib.__version__) != ("0.11.1", "0.11.1") or jax.config.jax_enable_pgle
    ):
        raise ValueError("The trimmed screen requires JAX/JAXLIB 0.11.1 and PGLE disabled")
    DistributedConfig().initialize()
    if len(jax.devices()) != 16 or any(d.device_kind != "TPU v4" for d in jax.devices()):
        raise ValueError("Use a v4-32 slice")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Keep artifacts in us-central2")
    rank = jax.process_index()
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / f"moe-transport-rank-{rank}"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(f"{args.artifact_prefix}/rank-{rank}")
    report = {
        "revision": args.revision,
        "status": "running",
        "rank": rank,
        "numerics": [],
        "timings": [],
        "full_shape_checks": [],
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jax_enable_pgle": jax.config.jax_enable_pgle,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "logical_chunks": 2,
        "physical_chunks": {"reference": 2, args.candidate: 1 if args.candidate == "coalesced" else 2},
        "candidate": args.candidate,
    }

    def save():
        path = folder / "moe-transport.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    def compile_fn(arrays, capacity_factor, candidate):
        context = nullcontext()
        if candidate:
            context = (
                patch.object(ragged, "_routed_experts", coalesced_routed_experts)
                if args.candidate == "coalesced"
                else patch.object(ragged, "_chunk_plans", trimmed_chunk_plans)
            )
        fn = jax.jit(partial(evaluate, mesh=mesh, capacity_factor=capacity_factor))
        with context:
            return fn.lower(*arrays).compile()

    mesh = compact_grug_mesh(expert_axis_size=16, replica_axis_size=1)
    specs = (*_SPECS, P("expert", None), P("expert", None))
    save()
    with jax.set_mesh(mesh):
        small_compiled = {}
        for seed in (0, 1, 2):
            for distribution in ("uniform", "sparse", "imbalanced", "empty"):
                for output_width in (128, 256):
                    host = fixture(seed, distribution, output_width)
                    if rank == 0:
                        path = folder / f"fixture-{seed}-{distribution}-{output_width}.npz"
                        np.savez(
                            path,
                            **dict(
                                zip(("x", "weights", "valid", "up", "down", "selected", "cotangent"), host, strict=True)
                            ),
                        )
                        fs.put_file(str(path), f"{prefix}/{path.name}")
                    for capacity in (0.5, 1.05):
                        expected, dropped = oracle(*host, capacity, 16)
                        arrays = tuple(
                            jax.make_array_from_callback(a.shape, NamedSharding(mesh, spec), lambda index, a=a: a[index])
                            for a, spec in zip(host, specs, strict=True)
                        )
                        row = {
                            "seed": seed,
                            "distribution": distribution,
                            "output_width": output_width,
                            "capacity_factor": capacity,
                            "expected_drops": dropped,
                            "implementations": {},
                        }
                        reference_values = None
                        for name in names:
                            cache_key = (output_width, capacity, name)
                            if cache_key not in small_compiled:
                                small_compiled[cache_key] = compile_fn(arrays, capacity, name != "reference")
                            result, drops = jax.block_until_ready(small_compiled[cache_key](*arrays))
                            gathered = tuple(multihost_utils.process_allgather(a, tiled=True) for a in result)
                            if name == "reference":
                                reference_values = gathered
                            else:
                                row["exact_match"] = all(
                                    np.array_equal(a, b) for a, b in zip(gathered, reference_values, strict=True)
                                )
                            row["implementations"][name] = {
                                "tensors": {
                                    key: errors(a, b) for key, a, b in zip(_NAMES, gathered, expected, strict=True)
                                },
                                "drops": int(drops.sender_dropped),
                                "receiver_drops": int(drops.receiver_dropped),
                            }
                        a, b = (row["implementations"][name] for name in names)
                        row["passed"] = (
                            a["drops"] == b["drops"] == dropped and a["receiver_drops"] == b["receiver_drops"] == 0
                        )
                        row["passed"] &= all(
                            a["tensors"][key]["finite"]
                            and b["tensors"][key]["finite"]
                            and all(
                                b["tensors"][key][metric] <= 1.25 * a["tensors"][key][metric] + 1e-3
                                for metric in ("max_abs_error", "relative_l2")
                            )
                            for key in _NAMES
                        )
                        if args.candidate == "trimmed":
                            row["passed"] &= row["exact_match"]
                        report["numerics"].append(row)
                        save()
                        print(json.dumps(row), flush=True)
        if not all(row["passed"] for row in report["numerics"]):
            report["status"] = "numerical-screen-failed"
            save()
            return
        small_compiled.clear()
        for output_width in (256, 512):
            shapes = (
                (524288, 256),
                (524288, 8),
                (524288,),
                (512, 256, 384),
                (512, 384, output_width),
                (524288, 8),
                (524288, output_width),
            )
            dtypes = (jnp.bfloat16, jnp.float32, jnp.bool_, jnp.float32, jnp.float32, jnp.int32, jnp.bfloat16)
            arrays = []
            for index, (shape, spec, dtype) in enumerate(zip(shapes, specs, dtypes, strict=True)):
                sharding = NamedSharding(mesh, spec)
                if index == 2:
                    a = jnp.ones(shape, dtype, out_sharding=sharding)
                elif index == 1:
                    a = jnp.full(shape, 1 / 8, dtype, out_sharding=sharding)
                elif index == 5:
                    ids = ((np.arange(shape[0])[:, None] * 17 + np.arange(8)[None, :] * 31) % 512).astype(np.int32)
                    a = jax.make_array_from_callback(shape, sharding, lambda index, ids=ids: ids[index])
                else:
                    a = jax.random.normal(jax.random.key(10 + index), shape, dtype, out_sharding=sharding)
                    if index in (3, 4):
                        a = a / np.sqrt(shape[1])
                arrays.append(a)
            compiled = {}
            reference_result = reference_drops = None
            for name in names:
                start = time.monotonic()
                compiled[name] = compile_fn(arrays, 1.05, name != "reference")
                elapsed = time.monotonic() - start
                for _ in range(2):
                    result, drops = jax.block_until_ready(compiled[name](*arrays))
                finite = bool(jax.jit(lambda xs: jnp.stack([jnp.isfinite(a).all() for a in xs]).all())(result))
                if name == "reference":
                    reference_result, reference_drops = result, drops
                    exact_match = True
                    drops_match = True
                else:
                    exact_match = bool(
                        jax.jit(lambda xs, ys: jnp.stack([jnp.array_equal(x, y) for x, y in zip(xs, ys)]).all())(
                            result, reference_result
                        )
                    )
                    drops_match = int(drops.sender_dropped) == int(reference_drops.sender_dropped) and int(
                        drops.receiver_dropped
                    ) == int(reference_drops.receiver_dropped)
                report["full_shape_checks"].append(
                    {
                        "output_width": output_width,
                        "implementation": name,
                        "finite": finite,
                        "exact_match": exact_match,
                        "drops_match": drops_match,
                        "drops": int(drops.sender_dropped),
                        "receiver_drops": int(drops.receiver_dropped),
                        "compile_seconds": elapsed,
                        "compiler_temporary_bytes": compiled[name].memory_analysis().temp_size_in_bytes,
                    }
                )
                if rank == 0:
                    path = folder / f"width-{output_width}-{name}-hlo.txt"
                    path.write_text(compiled[name].as_text())
                    fs.put_file(str(path), f"{prefix}/{path.name}")
                save()
                if not finite:
                    raise ValueError("Nonfinite full-size output or gradient")
                if args.candidate == "trimmed" and not (exact_match and drops_match):
                    raise ValueError("Trimmed full-size output/gradient or dropped-route mismatch")
            del reference_result, reference_drops
            for repeat in range(8):
                for name in (names if repeat % 2 == 0 else names[::-1]):
                    multihost_utils.sync_global_devices(f"moe-{output_width}-{repeat}-{name}")
                    start = time.monotonic()
                    jax.block_until_ready(compiled[name](*arrays))
                    report["timings"].append(
                        {
                            "output_width": output_width,
                            "implementation": name,
                            "repeat": repeat,
                            "seconds": time.monotonic() - start,
                        }
                    )
            save()
        report["median_seconds"] = {
            str(width): {
                name: statistics.median(
                    r["seconds"] for r in report["timings"] if r["output_width"] == width and r["implementation"] == name
                )
                for name in names
            }
            for width in (256, 512)
        }
        report["status"] = "completed"
        save()


if __name__ == "__main__":
    main()
