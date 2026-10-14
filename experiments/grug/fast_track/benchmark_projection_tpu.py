# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure explicit weight gathering for distributed Belay projections.

The small fixtures compare outputs and both input gradients with NumPy FP64.
The full-token fixtures measure forward plus backward without training updates.
"""

import argparse
import functools
import json
import os
import statistics
import time
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track.belay_tpu import base16_model_config, require_tpu_rounding
from experiments.grug.fast_track.diagnose_kda_state_precision import errors
from experiments.grug.fast_track.model import _proj as production_projection

_BATCH_SPEC = P(("replica_dcn", "data", "expert"), None, None)


def gathered_projection(cfg, x, w, out_sharding=None):
    """Gather small weights for the projection, retaining their stored sharding."""
    return production_projection(cfg, x, jax.sharding.reshard(w, P(None, None)), out_sharding)


def auxiliary_projection(cfg, x, w, out_sharding=None):
    """The direct matrix product used by Inkling and KDA auxiliary projections."""
    return jnp.einsum("bsd,dk->bsk", x, w)


def gathered_auxiliary_projection(cfg, x, w, out_sharding=None):
    return auxiliary_projection(cfg, x, jax.sharding.reshard(w, P(None, None)), out_sharding)


def projection_and_gradients(x, w, cotangent, *, projection, config):
    output, pullback = jax.vjp(lambda a, b: projection(config, a, b.astype(a.dtype), out_sharding=_BATCH_SPEC), x, w)
    dx, dw = pullback(cotangent)
    return output, dx, dw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--family", choices=("main", "auxiliary"), default="main")
    args = parser.parse_args()
    require_tpu_rounding()
    DistributedConfig().initialize()
    devices = jax.devices()
    if len(devices) != 16 or any(d.device_kind != "TPU v4" for d in devices):
        raise ValueError("This probe requires v4-32 with 16 chips")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Keep artifacts in us-central2")
    rank = jax.process_index()
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / f"projection-rank-{rank}"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(f"{args.artifact_prefix}/rank-{rank}")
    report = {
        "kernel": "distributed_projection",
        "revision": args.revision,
        "git_sha": args.revision,
        "family": args.family,
        "backend": "tpu",
        "device_type": devices[0].device_kind,
        "block_sizes": "XLA selected",
        "jax_version": jax.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
        "rank": rank,
        "device_count": len(devices),
        "oracle": "NumPy FP64 matrix products on identical represented inputs",
        "small_shape_prefix": [16, 64],
        "full_input_shape": [128, 4096, 512],
        "input_and_cotangent_dtype": "bfloat16",
        "stored_weight_dtype": "float32",
        "numerics": [],
        "timings": [],
        "full_shape_checks": [],
        "status": "running",
    }

    def save():
        path = folder / "projection.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    config = base16_model_config()
    mesh = compact_grug_mesh(expert_axis_size=16, replica_axis_size=1)
    if args.family == "auxiliary":
        implementations = (("production", auxiliary_projection), ("gather", gathered_auxiliary_projection))
        weight_specs = (("rows", P(("data", "expert"), None)),)
        small_shapes = ((512, 96), (512, 128))
        full_shapes = small_shapes
        seeds = (0, 1, 2)
    else:
        implementations = (("production", production_projection), ("gather", gathered_projection))
        weight_specs = (("rows", P(("data", "expert"), None)), ("columns", P(None, ("data", "expert"))))
        small_shapes = ((128, 512), (512, 512))
        full_shapes = ((512, 512),)
        seeds = (0, 1)
    with jax.set_mesh(mesh):
        for seed in seeds:
            for width, output_width in small_shapes:
                rng = np.random.default_rng(seed)
                x_host = rng.normal(size=(16, 64, width)).astype(ml_dtypes.bfloat16)
                w_host = (0.02 * rng.normal(size=(width, output_width))).astype(np.float32)
                ct_host = rng.normal(size=(16, 64, output_width)).astype(ml_dtypes.bfloat16)
                x64 = x_host.astype(np.float64)
                w64 = w_host.astype(ml_dtypes.bfloat16).astype(np.float64)
                ct64 = ct_host.astype(np.float64)
                oracle = (
                    x64 @ w64,
                    ct64 @ w64.T,
                    x64.reshape(-1, width).T @ ct64.reshape(-1, 512),
                )
                for sharding_name, weight_spec in weight_specs:
                    arrays = tuple(
                        jax.make_array_from_callback(a.shape, NamedSharding(mesh, spec), lambda index, a=a: a[index])
                        for a, spec in ((x_host, _BATCH_SPEC), (w_host, weight_spec), (ct_host, _BATCH_SPEC))
                    )
                    row = {
                        "seed": seed,
                        "width": width,
                        "output_width": output_width,
                        "weight_sharding": sharding_name,
                        "implementations": {},
                    }
                    for name, projection in implementations:
                        fn = jax.jit(functools.partial(projection_and_gradients, projection=projection, config=config))
                        result = jax.block_until_ready(fn(*arrays))
                        gathered = tuple(multihost_utils.process_allgather(a, tiled=True) for a in result)
                        row["implementations"][name] = {
                            key: errors(actual, expected)
                            for key, actual, expected in zip(("output", "dx", "dw"), gathered, oracle, strict=True)
                        }
                    row["passed"] = all(
                        row["implementations"]["production"][key]["finite"]
                        and row["implementations"]["gather"][key]["finite"]
                        and all(
                            row["implementations"]["gather"][key][metric]
                            <= 1.25 * row["implementations"]["production"][key][metric] + 1e-3
                            for metric in ("max_abs_error", "relative_l2")
                        )
                        for key in ("output", "dx", "dw")
                    )
                    report["numerics"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
        full_cases = tuple(
            (sharding_name, weight_spec, width, output_width)
            for sharding_name, weight_spec in weight_specs
            for width, output_width in full_shapes
        )
        for sharding_name, weight_spec, width, output_width in full_cases:
            case = f"{sharding_name}-{width}-{output_width}"
            x, w, ct = (
                jax.random.normal(jax.random.key(seed), shape, dtype, out_sharding=NamedSharding(mesh, spec))
                for seed, shape, dtype, spec in (
                    (10, (128, 4096, width), jnp.bfloat16, _BATCH_SPEC),
                    (11, (width, output_width), jnp.float32, weight_spec),
                    (12, (128, 4096, output_width), jnp.bfloat16, _BATCH_SPEC),
                )
            )
            compiled = {}
            for name, projection in implementations:
                fn = jax.jit(functools.partial(projection_and_gradients, projection=projection, config=config))
                start = time.monotonic()
                compiled[name] = fn.lower(x, w, ct).compile()
                compile_time = time.monotonic() - start
                for _ in range(2):
                    result = jax.block_until_ready(compiled[name](x, w, ct))
                finite = jax.jit(lambda xs: jnp.stack([jnp.all(jnp.isfinite(a)) for a in xs]).all())(result)
                report["full_shape_checks"].append(
                    {
                        "case": case,
                        "weight_sharding": sharding_name,
                        "implementation": name,
                        "finite": bool(finite),
                        "compile_time": compile_time,
                        "temporary_bytes": compiled[name].memory_analysis().temp_size_in_bytes,
                    }
                )
                if rank == 0:
                    path = folder / f"{case}-{name}-hlo.txt"
                    path.write_text(compiled[name].as_text())
                    fs.put_file(str(path), f"{prefix}/{path.name}")
            # Alternate order between paired rounds to expose timing drift.
            for repeat in range(8):
                order = ("production", "gather") if repeat % 2 == 0 else ("gather", "production")
                for name in order:
                    multihost_utils.sync_global_devices(f"projection-{case}-{repeat}-{name}")
                    start = time.monotonic()
                    result = jax.block_until_ready(compiled[name](x, w, ct))
                    seconds = time.monotonic() - start
                    report["timings"].append(
                        {
                            "case": case,
                            "weight_sharding": sharding_name,
                            "implementation": name,
                            "repeat": repeat,
                            "seconds": seconds,
                        }
                    )
                    del result
                    save()
            save()
            for name, _ in implementations:
                samples = [r["seconds"] for r in report["timings"] if r["case"] == case and r["implementation"] == name]
                print(json.dumps({"case": case, "implementation": name, "median": statistics.median(samples)}))
        report["status"] = "completed"
        report["numerical_gate_passed"] = all(row["passed"] for row in report["numerics"]) and all(
            row["finite"] for row in report["full_shape_checks"]
        )
        save()


if __name__ == "__main__":
    main()
