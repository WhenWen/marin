# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify causal buckets on native packed data without optimizer updates."""

import argparse
import asyncio
import functools
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.tracker as tracker
import numpy as np
from levanter.schedule import BatchSchedule
from levanter.trainer import _initialize_global_tracker

from experiments.grug.belay_tpu import train
from experiments.grug.belay_tpu.attention_kernel_check import dense, evaluate, pallas
from experiments.grug.belay_tpu.belay_ckpt1_tpu import DATA_ROOT, build_config, require_tpu_rounding, verified_tokenizer
from experiments.grug.belay_tpu.causal_attention import causal_attention_local
from experiments.grug.belay_tpu.kernel_check import measure_errors

NATIVE_SEGMENT_HASHES = {
    5187: "fd0677160aa633b25048eca52cd66b79ce27aa80eb0d9984203d1978cea130c0",
    5188: "62ef91b020abfab9abbe459056463ded694f540f78359f62c3a5d5652e4e8881",
}
WARMUPS = 3
PAIRED_SAMPLES = 40


def causal(q, k, v, band, ids):
    return causal_attention_local(q, k, v, band, ids, ids, 512)


async def native_segments(config):
    """Verify two complete native batches and select eight sequences per chip."""
    with jax.default_device(jax.devices("cpu")[0]):
        native = config.trainer.trainer
        schedule = BatchSchedule(native.train_batch_size)
        key, _ = jax.random.split(jax.random.PRNGKey(native.seed), 2)
        if config.trainer.data_seed is not None:
            key = jax.random.PRNGKey(config.trainer.data_seed)
        dataset = train.build_train_dataset(config.data, max_seq_len=4096, batch_schedule=schedule, key=key)
        result = {}
        for step, expected_hash in NATIVE_SEGMENT_HASHES.items():
            offset = schedule.global_data_offset_by_step(step)
            examples = await dataset.get_batch(list(range(offset, offset + 128)))
            ids = np.stack([np.asarray(example.attn_mask.segment_ids[0], np.int32) for example in examples])
            assert ids.shape == (128, 4096)
            assert all(
                np.array_equal(example.attn_mask.segment_ids[0], example.attn_mask.segment_ids[1])
                for example in examples
            )
            assert hashlib.sha256(ids.tobytes()).hexdigest() == expected_hash
            result[f"native_step_{step}"] = np.repeat(ids[:8], 4, axis=0)
        return result


def timed_pair(compiled, arguments):
    """Alternate order and retain every timed sample after fixed warmup."""
    samples = {name: [] for name in compiled}
    names = list(compiled)
    for index in range(WARMUPS + PAIRED_SAMPLES):
        order = names if index % 2 == 0 else names[::-1]
        for name in order:
            start = time.perf_counter()
            jax.block_until_ready(compiled[name](*arguments))
            elapsed = time.perf_counter() - start
            if index >= WARMUPS:
                samples[name].append(elapsed)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logging-only", action="store_true")
    args = parser.parse_args()
    require_tpu_rounding()
    run_id = os.environ["BELAY_RUN_ID"]
    config = build_config(
        run_id,
        f"{DATA_ROOT}/kaiyuew/belay-ckpt1-20261025/{run_id}",
        verified_tokenizer(),
        os.environ["BELAY_TPU"],
        0,
        local_experts=True,
        kernel_profile="kda-owner-attention-v1",
        pipeline_depth=1,
    )
    if args.logging_only:
        _initialize_global_tracker(config.trainer.trainer.tracker, run_id)
        tracker.log_configuration(config)
        tracker.log_summary(
            {
                "execution/logging_only_pass": True,
                "execution/optimizer_updates": 0,
                "kernel_check/source_commit": os.environ["BELAY_SOURCE_COMMIT"],
            }
        )
        tracker.current_tracker().finish()
        print(json.dumps({"stage": "logging_only_pass", "optimizer_updates": 0}), flush=True)
        return
    config.trainer.trainer.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Causal attention qualification requires TPU")
    tracker.log_configuration(config)
    tracker.log_summary({"execution/optimizer_updates": 0})
    native = asyncio.run(native_segments(config))
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    device = jax.local_devices()[0]
    functions = {
        name: jax.jit(functools.partial(evaluate, kernel))
        for name, kernel in (("dense_reference", dense), ("selected_pallas", pallas), ("causal_bucket512", causal))
    }
    all_numerical_pass, all_target_speed_pass = True, True
    scenarios = [
        ("small_full", np.zeros((2, 512), np.int32)),
        ("small_strong_packed", np.broadcast_to(np.arange(384) // 37, (2, 384)).astype(np.int32)),
        ("full_length", np.zeros((32, 4096), np.int32)),
        *native.items(),
    ]
    with jax.default_device(device), jax.default_matmul_precision("highest"):
        for scenario, host_ids in scenarios:
            for dtype, tolerance in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
                if host_ids.shape[1] == 4096 and dtype == jnp.float32:
                    continue
                rng = np.random.default_rng(452)
                shape = (*host_ids.shape, 128)
                scale = 4 if scenario == "small_strong_packed" else 1
                q, k = (jnp.asarray(scale * rng.standard_normal(shape, dtype=np.float32), dtype) for _ in range(2))
                v = jnp.asarray(rng.standard_normal(shape, dtype=np.float32), dtype)
                band = jnp.asarray(0.2 * rng.standard_normal((*host_ids.shape, 1152), dtype=np.float32), dtype)
                cotangent = jnp.asarray(rng.standard_normal(shape, dtype=np.float32), dtype)
                arguments = ((q, k, v, band), jnp.asarray(host_ids), cotangent)
                compiled = {name: function.lower(*arguments).compile() for name, function in functions.items()}
                expected = jax.block_until_ready(compiled.pop("dense_reference")(*arguments))
                errors = {
                    name: measure_errors(jax.block_until_ready(function(*arguments)), expected, tolerance)
                    for name, function in compiled.items()
                }
                numerical_pass = all(
                    np.isfinite(leaf["normalized_max"]) and leaf["normalized_max"] <= 1
                    for leaves in errors.values()
                    for leaf in leaves
                )
                all_numerical_pass &= numerical_pass
                samples = timed_pair(compiled, arguments)
                medians = {name: statistics.median(values) for name, values in samples.items()}
                speedup = 1 - medians["causal_bucket512"] / medians["selected_pallas"]
                if host_ids.shape[1] == 4096:
                    all_target_speed_pass &= speedup >= 0.05
                record = dict(
                    stage="causal_attention_operator",
                    scenario=scenario,
                    dtype=str(jnp.dtype(dtype)),
                    shape=shape,
                    bucket_size=512,
                    band_width=1152,
                    seed=452,
                    rank=jax.process_index(),
                    device=device.device_kind,
                    single_device=True,
                    optimizer_updates=0,
                    source_commit=os.environ["BELAY_SOURCE_COMMIT"],
                    native_segment_hashes=NATIVE_SEGMENT_HASHES,
                    numerical_pass=numerical_pass,
                    tolerance=tolerance,
                    error_leaves=["output", "dq", "dk", "dv", "dband"],
                    errors=errors,
                    warmups=WARMUPS,
                    paired_samples=PAIRED_SAMPLES,
                    samples=samples,
                    median_seconds=medians,
                    mean_seconds={name: statistics.mean(values) for name, values in samples.items()},
                    elapsed_reduction_fraction=speedup,
                    temporary_bytes={
                        name: function.memory_analysis().temp_size_in_bytes for name, function in compiled.items()
                    },
                    xla_flags=os.environ.get("XLA_FLAGS", ""),
                    libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
                )
                with (output / "causal-attention-check.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
    tracker.log_summary(
        {
            "kernel_check/numerical_pass": all_numerical_pass,
            "kernel_check/all_target_medians_faster_by_5_percent": all_target_speed_pass,
            "kernel_check/promotable": all_numerical_pass and all_target_speed_pass,
        }
    )
    tracker.current_tracker().finish()
    if not all_numerical_pass:
        raise AssertionError("Causal attention failed its unchanged output/gradient bounds; all evidence retained")


if __name__ == "__main__":
    main()
