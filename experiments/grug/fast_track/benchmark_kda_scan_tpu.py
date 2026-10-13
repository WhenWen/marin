# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare KDA affine scan reconstruction on one TPU v4 chip."""

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


def compose(left, right):
    ml, cl = left
    mr, cr = right
    return jnp.einsum("...ij,...jk->...ik", mr, ml), jnp.einsum("...ij,...jm->...im", mr, cl) + cr


def stacked_scan(elems):
    """Use JAX's pairwise scan order with stack/reshape interleaving on axis -3."""
    n = elems[0].shape[-3]
    if n < 2:
        return elems
    odd = stacked_scan(compose(tuple(x[..., :-1:2, :, :] for x in elems), tuple(x[..., 1::2, :, :] for x in elems)))
    left = tuple(x[..., :-1, :, :] for x in odd) if n % 2 == 0 else odd
    even = compose(left, tuple(x[..., 2::2, :, :] for x in elems))
    even = tuple(jnp.concatenate((x[..., :1, :, :], y), axis=-3) for x, y in zip(elems, even, strict=True))
    result = []
    for a, b in zip(even, odd, strict=True):
        paired = jnp.stack((a[..., : n // 2, :, :], b), axis=-3)
        paired = paired.reshape(*a.shape[:-3], 2 * (n // 2), *a.shape[-2:])
        result.append(jnp.concatenate((paired, a[..., -1:, :, :]), axis=-3) if n % 2 else paired)
    return tuple(result)


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
    rows = []
    with jax.default_device(device):
        for shape in ((1, 2, 7, 8, 8), (2, 4, 64, 64, 64), (8, 4, 256, 128, 128)):
            # Stable transitions, including resets, exercise the same affine algebra.
            m = 0.95 * jnp.eye(shape[-1]) + 0.001 * jax.random.normal(jax.random.key(0), shape)
            m = m.at[..., ::5, :, :].set(0.0)
            c = jax.random.normal(jax.random.key(1), shape)
            cot = jax.random.normal(jax.random.key(2), shape)
            inputs = (m, c)
            for mode in ("forward", "value_and_grad"):
                reference = None
                for name, scan in (("jax_reference", lambda xs: jax.lax.associative_scan(compose, xs, axis=-3)), ("stacked", stacked_scan)):
                    def run(xs, ct):
                        def objective(ys):
                            a, b = scan(ys)
                            return jnp.mean(a * ct) + jnp.mean(b * ct)

                        return scan(xs) if mode == "forward" else jax.value_and_grad(objective)(xs)

                    row: dict[str, object] = {
                        "kernel": "kda_affine_scan", "implementation": name, "shape": shape,
                        "pass_mode": mode, "dtype": "float32", "seed": 0,
                        "backend": "tpu", "device_type": device.device_kind, "device_count": 1,
                        "block_sizes": {}, "git_sha": args.revision, "jax_version": jax.__version__,
                        "jaxlib_version": jaxlib.__version__, "xla_flags": os.environ.get("XLA_FLAGS", ""),
                        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")}, "error": None,
                    }
                    try:
                        start = time.monotonic()
                        compiled = jax.jit(run).lower(inputs, cot).compile()
                        row["compile_time"] = time.monotonic() - start
                        actual = jax.block_until_ready(compiled(inputs, cot))
                        if reference is None:
                            reference = actual
                        pairs = list(zip(jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True))
                        row["max_abs_error"] = max(float(jnp.max(jnp.abs(a - b))) for a, b in pairs)
                        row["mean_abs_errors"] = [float(jnp.mean(jnp.abs(a - b))) for a, b in pairs]
                        if not all(bool(jnp.allclose(a, b, atol=1e-5, rtol=1e-5)) for a, b in pairs):
                            raise AssertionError("Value or gradient differs from JAX associative_scan")
                        row["temporary_bytes"] = compiled.memory_analysis().temp_size_in_bytes
                        for _ in range(2):
                            jax.block_until_ready(compiled(inputs, cot))
                        samples = []
                        for _ in range(5):
                            start = time.monotonic()
                            jax.block_until_ready(compiled(inputs, cot))
                            samples.append(time.monotonic() - start)
                        row.update(steady_state_time=statistics.median(samples), steady_state_samples=samples)
                        if shape[-1] == 128:
                            hlo = output / f"{name}-{mode}-hlo.txt"
                            hlo.write_text(compiled.as_text())
                            fs.put_file(str(hlo), f"{prefix}/{hlo.name}")
                    except Exception as exc:
                        row["error"] = repr(exc)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    report = output / "scan-benchmark.json"
                    report.write_text(json.dumps(rows, indent=2) + "\n")
                    fs.put_file(str(report), f"{prefix}/{report.name}")
    if any(row["error"] for row in rows):
        raise RuntimeError("A scan candidate failed; see the retained report")


if __name__ == "__main__":
    main()
