# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure bit-exact MoE row gathering candidates on one TPU v4 chip."""

import argparse
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import jaxlib


def gather_rows(x, indices, *, packed, rows):
    width = x.shape[-1]
    if packed:
        x = jax.lax.bitcast_convert_type(x.reshape(x.shape[0], width // 2, 2), jnp.uint32)
    if rows is None:
        out = x[indices]
    else:
        out = jax.lax.map(lambda idx: x[idx], indices.reshape(-1, rows)).reshape(indices.size, x.shape[-1])
    if packed:
        out = jax.lax.bitcast_convert_type(out, jnp.bfloat16).reshape(indices.size, width)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    device = jax.local_devices()[0]
    if device.device_kind != "TPU v4":
        raise RuntimeError("This benchmark requires TPU v4")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        parser.error("Artifacts must stay in us-central2")
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    records = []
    with jax.default_device(device):
        for tokens, width in ((4096, 16), (32768, 512)):
            selected = jax.lax.top_k(jax.random.uniform(jax.random.key(1), (tokens, 512)), 8)[1]
            order = jnp.argsort(selected.reshape(-1), stable=True)
            for mode in ("dispatch", "permutation"):
                source_rows = tokens if mode == "dispatch" else tokens * 8
                if width == 16:
                    bits = jnp.arange(source_rows * width, dtype=jnp.uint16).reshape(source_rows, width)
                    x = jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
                else:
                    x = jax.random.normal(jax.random.key(0), (source_rows, width), jnp.bfloat16)
                indices = order // 8 if mode == "dispatch" else jnp.argsort(order)
                reference = jax.block_until_ready(x[indices])
                reference_bits = jax.lax.bitcast_convert_type(reference, jnp.uint16)
                for rows in (None, 128, 512, 4096, 16384):
                    for packed in (False, True):
                        name = f"{'packed' if packed else 'bf16'}_rows_{rows}"
                        record: dict[str, object] = {
                            "kernel": "moe_row_gather",
                            "implementation": name,
                            "shape": [source_rows, tokens * 8, width],
                            "mode": mode,
                            "dtype": "bfloat16",
                            "seed": 0,
                            "routing_seed": 1,
                            "index_distribution": "top-8 uniform scores over 512 experts, stable expert sort",
                            "backend": "tpu",
                            "device_type": device.device_kind,
                            "device_count": 1,
                            "block_sizes": {"rows": rows, "packed_pairs": packed},
                            "git_sha": args.revision,
                            "jax_version": jax.__version__,
                            "jaxlib_version": jaxlib.__version__,
                            "xla_flags": os.environ.get("XLA_FLAGS", ""),
                            "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                            "error": None,
                        }
                        try:
                            fn = partial(gather_rows, packed=packed, rows=rows)
                            start = time.monotonic()
                            compiled = jax.jit(fn).lower(x, indices).compile()
                            record["compile_time"] = time.monotonic() - start
                            actual = jax.block_until_ready(compiled(x, indices))
                            actual_bits = jax.lax.bitcast_convert_type(actual, jnp.uint16)
                            record["bit_mismatches"] = int(jnp.sum(actual_bits != reference_bits))
                            if record["bit_mismatches"]:
                                raise AssertionError("Gather changed bf16 bit patterns")
                            record["temporary_bytes"] = compiled.memory_analysis().temp_size_in_bytes
                            for _ in range(2):
                                jax.block_until_ready(compiled(x, indices))
                            samples = []
                            for _ in range(5):
                                start = time.monotonic()
                                jax.block_until_ready(compiled(x, indices))
                                samples.append(time.monotonic() - start)
                            record.update(steady_state_time=statistics.median(samples), steady_state_samples=samples)
                            if width == 512:
                                hlo = output / f"{mode}-{name}-hlo.txt"
                                hlo.write_text(compiled.as_text())
                                fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
                        except Exception as exc:
                            record["error"] = repr(exc)
                        records.append(record)
                        print(json.dumps(record), flush=True)
                        report = output / "gather-benchmark.json"
                        report.write_text(json.dumps(records, indent=2) + "\n")
                        fs.put_file(str(report), f"{prefix}/{report.name}")
    if any(r["error"] for r in records):
        raise RuntimeError("A gather candidate failed; see the retained report")


if __name__ == "__main__":
    main()
