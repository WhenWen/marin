# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the complete base16 d512 forward/backward on TPU, without a dataset.

Random-token numerical validation is not a Belay training result. The model,
global batch and sequence length match the frozen run; no optimizer updates run.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import jmp
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.callbacks.profiler import ProfileOptionsConfig
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track.belay_tpu import SOURCE_CONFIG, base16_model_config
from experiments.grug.fast_track.model import Transformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, help="Override only the diagnostic batch; default is the frozen 128.")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--revision", help="Exact source commit for benchmark provenance.")
    parser.add_argument("--artifact-prefix", help="Unique gs://marin-us-central2/ path for durable diagnostics.")
    args = parser.parse_args()
    if args.benchmark_steps < 0 or args.profile_steps < 0:
        parser.error("Step counts must be nonnegative")
    if (args.benchmark_steps or args.profile_steps) and not args.revision:
        parser.error("Benchmarks require --revision")
    if args.artifact_prefix and not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        parser.error("Artifacts must remain in us-central2")
    DistributedConfig().initialize()
    devices = jax.devices()
    if any(d.platform != "tpu" or "v4" not in d.device_kind for d in devices):
        raise RuntimeError("Full-model validation requires TPU v4")
    config = base16_model_config()
    source = json.loads(SOURCE_CONFIG.read_text())
    batch = source["trainer"]["trainer"]["train_batch_size"] if args.batch_size is None else args.batch_size
    if batch <= 0 or batch % len(devices):
        raise ValueError("Diagnostic batch must be positive and divisible by the TPU chip count")
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    mesh = compact_grug_mesh(expert_axis_size=len(devices), replica_axis_size=1)
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "random-token forward/backward; not a training result",
        "batch_size": batch,
        "sequence_length": config.max_seq_len,
        "chips": len(devices),
        "seed": 0,
        "kernel": "base16_forward_backward",
        "implementation": "native_grug_tpu",
        "shape": [batch, config.max_seq_len, config.hidden_dim],
        "dtype": "params=float32,compute=bfloat16,output=bfloat16",
        "backend": "tpu",
        "device_type": devices[0].device_kind,
        "device_count": len(devices),
        "block_sizes": {"inkling_query": 128},
        "git_sha": args.revision,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
        "error": None,
        "status": "initializing",
    }
    rank = jax.process_index()
    report_path = output / f"full-model-rank-{rank}.json"

    def save_report():
        payload = json.dumps(report, indent=2) + "\n"
        report_path.write_text(payload)
        if args.artifact_prefix:
            with fsspec.open(f"{args.artifact_prefix}/rank-{rank}/report.json", "w") as stream:
                stream.write(payload)

    save_report()
    with jax.set_mesh(mesh):
        print(f"Initializing base16 d512 on {len(devices)} TPU v4 chips, batch={batch}, seq=4096", flush=True)
        model = eqx.filter_jit(lambda key: Transformer.init(config, key=key))(jax.random.key(0))
        spec = P(("replica_dcn", "data", "expert"), None)
        tokens = jax.sharding.reshard(
            jax.random.randint(jax.random.key(1), (batch, config.max_seq_len), 0, config.vocab_size), spec
        )
        weights = jax.sharding.reshard(jnp.ones(tokens.shape, jnp.float32), spec)
        segments = jax.sharding.reshard(jnp.broadcast_to(jnp.arange(config.max_seq_len) // 1024, tokens.shape), spec)
        mask = AttentionMask.causal().with_segment_ids(segments)

        def loss(model, tokens, weights, mask):
            return policy.cast_to_compute(model).next_token_loss(tokens, weights, mask=mask)

        start = time.monotonic()
        print("Compiling and executing forward/backward", flush=True)
        compiled = eqx.filter_jit(eqx.filter_value_and_grad(loss)).lower(model, tokens, weights, mask).compile()
        report["compile_time"] = time.monotonic() - start
        loss_value, grads = jax.block_until_ready(compiled(model, tokens, weights, mask))
        report["compile_and_first_execute_seconds"] = time.monotonic() - start
        finite = jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads) if eqx.is_array(x)]).all()
        report.update(
            loss=float(loss_value), finite_gradients=bool(finite), compile_and_execute_seconds=time.monotonic() - start
        )
        report["status"] = "validated"
        save_report()
        print(json.dumps(report), flush=True)
        if not report["finite_gradients"] or not bool(jnp.isfinite(loss_value)):
            raise RuntimeError("Nonfinite base16 loss or gradients")
        del loss_value, grads
        if args.benchmark_steps or args.profile_steps:
            memory = compiled.compiled.memory_analysis()
            report["compiler_memory_bytes"] = {
                "arguments": memory.argument_size_in_bytes,
                "outputs": memory.output_size_in_bytes,
                "temporary": memory.temp_size_in_bytes,
                "aliases": memory.alias_size_in_bytes,
            }
            for _ in range(2):
                result = jax.block_until_ready(compiled(model, tokens, weights, mask))
                del result
            samples = []
            for step in range(args.benchmark_steps):
                start = time.monotonic()
                result = jax.block_until_ready(compiled(model, tokens, weights, mask))
                elapsed = time.monotonic() - start
                samples.append(elapsed)
                del result
                print(json.dumps({"rank": rank, "benchmark_step": step, "seconds": elapsed}), flush=True)
            report["steady_state_samples"] = samples
            report["steady_state_time"] = statistics.median(samples) if samples else None
            report["status"] = "benchmarked"
            save_report()
        if args.profile_steps:
            profile_dir = output / f"profile-rank-{rank}"
            options = ProfileOptionsConfig(
                host_tracer_level=1, python_tracer_level=0, enable_hlo_proto=True
            ).build_jax_profile_options()
            with jax.profiler.trace(str(profile_dir), profiler_options=options):
                for step in range(args.profile_steps):
                    with jax.profiler.StepTraceAnnotation("base16_forward_backward", step_num=step):
                        result = jax.block_until_ready(compiled(model, tokens, weights, mask))
                        del result
            if rank == 0:
                (profile_dir / "optimized-hlo.txt").write_text(compiled.compiled.as_text())
            if args.artifact_prefix:
                fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
                for path in profile_dir.rglob("*"):
                    if path.is_file():
                        fs.put_file(str(path), f"{prefix}/rank-{rank}/profile/{path.relative_to(profile_dir)}")
            report["status"] = "profiled"
            save_report()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
