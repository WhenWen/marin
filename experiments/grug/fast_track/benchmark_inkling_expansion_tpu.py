# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare aligned Inkling expansion implementations on TPU v4.

This isolates the bias expansion and its transpose; a win here is not a
whole-model speedup. Query offsets are multiples of the 128-row Inkling block.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, dense_rel_bias, rel_extent_of_band


def aligned_slice(band, query_offset, k_len):
    """Expand one aligned query block without an indexed gather."""
    padded = jnp.pad(band, ((0, 0), (0, 0), (0, 0), (k_len, k_len)))
    start = k_len + rel_extent_of_band(band) - query_offset
    return jax.lax.dynamic_slice_in_dim(padded, start, k_len, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    args = parser.parse_args()
    device = jax.local_devices()[0]
    if device.platform != "tpu" or "v4" not in device.device_kind:
        raise RuntimeError("This benchmark requires TPU v4")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        parser.error("Artifacts must stay in us-central2")
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    rows = []
    for batch, heads, keys, extent, offsets in (
        (1, 2, 256, 128, (0, 128)),
        (8, 6, 4096, 1024, (0, 1024, 3968)),
    ):
        for dtype in (jnp.float32, jnp.bfloat16):
            with jax.default_device(device):
                band = jax.random.normal(jax.random.key(0), (batch, heads, REL_BIAS_BLOCK, extent + REL_BIAS_BLOCK), dtype)
                cotangent = jax.random.normal(jax.random.key(1), (batch, heads, REL_BIAS_BLOCK, keys), dtype)

                def loss(band, cotangent, offset, *, candidate):
                    expanded = (
                        aligned_slice(band, offset, keys)
                        if candidate
                        else dense_rel_bias(band, REL_BIAS_BLOCK, keys, query_offset=offset)
                    )
                    return jnp.sum(expanded.astype(jnp.float32) * cotangent.astype(jnp.float32)), expanded

                for offset in offsets:
                    offset_array = jnp.asarray(offset, jnp.int32)
                    reference = None
                    for candidate in (False, True):
                        name = "aligned_pad_slice" if candidate else "gather_reference"
                        start = time.monotonic()
                        compiled = jax.jit(
                            jax.value_and_grad(loss, has_aux=True), static_argnames=("candidate",)
                        ).lower(band, cotangent, offset_array, candidate=candidate).compile()
                        compile_time = time.monotonic() - start
                        actual = jax.block_until_ready(compiled(band, cotangent, offset_array))
                        actual_host = [np.asarray(x).astype(np.float32) for x in jax.tree.leaves(actual)]
                        if reference is None:
                            reference = actual_host
                        errors = [np.abs(x - y) for x, y in zip(actual_host, reference, strict=True)]
                        for x, y in zip(actual_host, reference, strict=True):
                            np.testing.assert_allclose(x, y, atol=1e-5, rtol=1e-5)
                        del actual
                        samples = []
                        for _ in range(5):
                            start = time.monotonic()
                            result = jax.block_until_ready(compiled(band, cotangent, offset_array))
                            samples.append(time.monotonic() - start)
                            del result
                        row = {
                            "kernel": "inkling_bias_expansion_forward_backward",
                            "implementation": name,
                            "shape": [batch, heads, REL_BIAS_BLOCK, keys, extent],
                            "query_offset": offset,
                            "dtype": str(jnp.dtype(dtype)),
                            "backend": "tpu", "device_type": device.device_kind, "device_count": 1,
                            "block_sizes": {"query": REL_BIAS_BLOCK},
                            "compile_time": compile_time, "steady_state_time": statistics.median(samples),
                            "steady_state_samples": samples,
                            "max_abs_error": max(float(x.max()) for x in errors),
                            "mean_abs_errors": [float(x.mean()) for x in errors],
                            "error": None, "git_sha": args.revision,
                            "xla_flags": os.environ.get("XLA_FLAGS", ""),
                            "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                        }
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        report = output / "expansion-benchmark.json"
                        report.write_text(json.dumps(rows, indent=2) + "\n")
                        fs.put_file(str(report), f"{prefix}/{report.name}")
                        if keys == 4096 and offset == 1024:
                            hlo = output / f"{name}-{jnp.dtype(dtype)}-hlo.txt"
                            hlo.write_text(compiled.as_text())
                            fs.put_file(str(hlo), f"{prefix}/{hlo.name}")


if __name__ == "__main__":
    main()
