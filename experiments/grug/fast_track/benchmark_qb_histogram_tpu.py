# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure exact grouped QB histograms on TPU v4, without training a model."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


def flat_histogram(indices: jax.Array, n_bins: int) -> jax.Array:
    experts = indices.shape[1]
    flat = (jnp.arange(experts, dtype=jnp.int32)[None, :] * n_bins + indices).reshape(-1)
    return jnp.bincount(flat, length=experts * n_bins).reshape(experts, n_bins)


def grouped_histogram(indices: jax.Array, n_bins: int, group_size: int) -> jax.Array:
    """Use identical bins and integer sums, limiting each scatter to a group."""
    tokens, experts = indices.shape
    if experts % group_size:
        raise ValueError("Expert count must divide into equal histogram groups")
    groups = indices.T.reshape(experts // group_size, group_size, tokens)
    counts = jax.lax.map(lambda group: flat_histogram(group.T, n_bins), groups)
    return counts.reshape(experts, n_bins)


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
    with jax.default_device(device):
        for tokens, experts, bins in ((257, 16, 31), (32768, 512, 10000)):
            for distribution in ("uniform", "concentrated"):
                indices = jax.random.randint(jax.random.key(0), (tokens, experts), 0, bins)
                if distribution == "concentrated":
                    indices = jnp.where(indices < bins // 2, 0, bins - 1)
                reference = None
                for group in (None, 1, 8, 32, 128, 512):
                    if group is not None and experts % group:
                        continue
                    name = "flat_reference" if group is None else f"group_{group}"
                    fn = lambda x: flat_histogram(x, bins) if group is None else grouped_histogram(x, bins, group)
                    row = {
                        "kernel": "qb_integer_histogram",
                        "implementation": name,
                        "shape": [tokens, experts, bins],
                        "distribution": distribution,
                        "seed": 0,
                        "dtype": "int32",
                        "backend": "tpu",
                        "device_type": device.device_kind,
                        "device_count": 1,
                        "block_sizes": {"experts": group},
                        "git_sha": args.revision,
                        "jax_version": jax.__version__,
                        "jaxlib_version": jaxlib.__version__,
                        "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "error": None,
                    }
                    try:
                        start = time.monotonic()
                        compiled = jax.jit(fn).lower(indices).compile()
                        row["compile_time"] = time.monotonic() - start
                        actual = np.asarray(jax.block_until_ready(compiled(indices)))
                        if reference is None:
                            reference = actual
                        np.testing.assert_array_equal(actual, reference)
                        np.testing.assert_array_equal(actual.sum(axis=1), np.full(experts, tokens))
                        row["max_abs_error"] = int(np.abs(actual - reference).max())
                        memory = compiled.memory_analysis()
                        row["temporary_bytes"] = memory.temp_size_in_bytes
                        for _ in range(2):
                            jax.block_until_ready(compiled(indices))
                        samples = []
                        for _ in range(5):
                            start = time.monotonic()
                            jax.block_until_ready(compiled(indices))
                            samples.append(time.monotonic() - start)
                        row.update(steady_state_time=statistics.median(samples), steady_state_samples=samples)
                        if tokens == 32768 and distribution == "uniform":
                            hlo = output / f"{name}-hlo.txt"
                            hlo.write_text(compiled.as_text())
                            fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
                    except Exception as exc:
                        row["error"] = repr(exc)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    report = output / "histogram-benchmark.json"
                    report.write_text(json.dumps(rows, indent=2) + "\n")
                    fs.put_file(str(report), f"{prefix}/{report.name}")
    if any(row["error"] for row in rows):
        raise RuntimeError("One or more histogram candidates failed; see the retained report")


if __name__ == "__main__":
    main()
