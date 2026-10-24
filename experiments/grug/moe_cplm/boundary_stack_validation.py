# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero-update checks and fixed-shape timing for boundary head mixing."""

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe import boundary_stack_kda as stack
from experiments.grug.moe import kda
from experiments.grug.moe_cplm.benchmark_frame_kda import write_record
from experiments.grug.moe_cplm.benchmark_head_stack_kda import benchmark, native, objective, operator_inputs
from experiments.grug.moe_cplm.benchmark_head_stack_kda import fixture as token_fixture


def fixture(length, rank=128, chunk=16, heads=4):
    xs, state = token_fixture(length, rank=rank, width=rank, mixed=True, heads=heads)
    return (xs[0][:, chunk - 1 :: chunk], *xs[1:]), state


def candidate(xs, state, chunk=16, mixed=False, interpret=False, reference=False):
    args, segments = operator_inputs(xs)
    out, final = stack.chunk_kda(
        *args,
        actions=xs[0],
        initial_state=state,
        use_qk_l2norm=False,
        segment_ids=segments,
        chunk_size=chunk,
        matmul_dtype=jnp.bfloat16 if mixed else jnp.float32,
        interpret=interpret,
        reference=reference,
    )
    return jnp.swapaxes(out, 1, 2), final


def literal(xs, initial, chunk=16):
    actions, q, k, v, g, beta, keep = xs
    tokens = tuple(jnp.swapaxes(x, 0, 1) for x in (q, k, v, g, beta, jax.lax.stop_gradient(keep)))

    def step(state, token):
        index, (q, k, v, g, beta, keep) = token
        state = state * (jnp.exp(g) * keep[:, None, None])[..., None]
        prediction = jnp.einsum("bhr,bhrv->bhv", k, state, precision=jax.lax.Precision.HIGHEST)
        state += (beta[..., None] * k)[..., None] * (v - prediction)[..., None, :]
        out = jnp.einsum("bhr,bhrv->bhv", q, state, precision=jax.lax.Precision.HIGHEST)
        state = jax.lax.cond(
            (index + 1) % chunk == 0,
            lambda s: (
                jnp.einsum(
                    "bij,bjrv->birv",
                    actions[:, index // chunk].reshape(-1, 4, 4),
                    s,
                    precision=jax.lax.Precision.HIGHEST,
                )
                if actions.shape[-1] == 16
                else jax.vmap(stack.mix)(
                    s, jnp.broadcast_to(actions[:, index // chunk, :, None], (s.shape[0], 3, s.shape[-1]))
                )
            ),
            lambda s: s,
            state,
        )
        return state, out

    final, out = jax.lax.scan(step, initial, (jnp.arange(q.shape[1]), tokens))
    return jnp.swapaxes(out, 0, 1), final


def evaluate(fn, xs, state):
    # Apply the strict reference policy to transposes introduced by AD too.
    with jax.default_matmul_precision("highest"):
        values = jax.jit(fn)(xs, state)
        grads = jax.jit(jax.value_and_grad(lambda x, s: objective(fn, x, s), argnums=(0, 1)))(xs, state)
    return values, grads


def high_precision_oracle(xs, state):
    # The token recurrence compounds one exponential rounding per token. Use
    # an independent CPU FP64 oracle rather than TPU approximate exponentials.
    with jax.default_device(jax.local_devices(backend="cpu")[0]), jax.enable_x64():
        xs, state = jax.tree.map(lambda x: jnp.asarray(np.asarray(x), jnp.float64), (xs, state))
        return jax.device_get(evaluate(literal, xs, state))


def compare(actual, expected, strict=True):
    result = []
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite boundary-stack result")
        result.append(
            {
                "max_abs": float(np.max(np.abs(a - b))),
                "relative_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
            }
        )
        if strict:
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
    return result


def validate_boundary_stack(rank=128, interpret=False, heads=4):
    xs, state = fixture(32, rank, heads=heads)
    exact = lambda x, s: candidate(x, s, interpret=interpret)  # noqa: E731
    mixed = lambda x, s: candidate(x, s, mixed=True, interpret=interpret)  # noqa: E731
    result = {"training_updates": 0, "heads": heads, "state_bytes": state.size * state.dtype.itemsize}
    oracle = high_precision_oracle(xs, state)
    actual = evaluate(exact, xs, state)
    scan = evaluate(lambda x, s: candidate(x, s, reference=True), xs, state)
    mixed_actual = evaluate(mixed, xs, state)
    mixed_scan = evaluate(lambda x, s: candidate(x, s, mixed=True, reference=True), xs, state)
    print(
        json.dumps(
            {
                "fp32_diagnostics": compare(actual, oracle, False),
                "fp32_fused_vs_scan": compare(actual, scan, False),
                "fp32_scan_vs_literal": compare(scan, oracle, False),
                "mixed_vjp_diagnostics": compare(mixed_actual, mixed_scan, False),
            }
        ),
        flush=True,
    )
    result["fp32"] = compare(actual, oracle)
    result["mixed_vs_literal"] = compare(mixed_actual, oracle, False)
    if any(row["relative_l2"] > 0.05 and row["max_abs"] > 1e-4 for row in result["mixed_vs_literal"]):
        raise ValueError("Mixed approximation exceeds the frozen 5% relative-L2 / 1e-4 absolute bound")
    result["mixed_vjp"] = compare(mixed_actual, mixed_scan)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    source = Path(__file__).parent
    settings = json.loads((source / "belay_recipe.json").read_text())["trainer"]["trainer"]["jax_config"]
    settings["jax_enable_pgle"] = False
    for key, value in settings.items():
        jax.config.update(key, value)
    if any(d.device_kind != "TPU v4" for d in jax.local_devices()):
        raise ValueError("Boundary-stack qualification requires TPU v4")
    identity = uuid.uuid4().hex
    directory, root = Path("/tmp/boundary-stack") / identity, args.artifact_root.rstrip("/") + "/" + identity
    directory.mkdir(parents=True)

    def record(name, value):
        write_record(directory, root, name + ".json", value)
        print(json.dumps({name: value}), flush=True)

    record(
        "manifest",
        {
            "training_updates": 0,
            "artifact_root": root,
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "device_type": jax.local_devices()[0].device_kind,
            "benchmark_device_count": 1,
            "allocated_chips": len(jax.devices()),
            "shape": {"batch": 1, "length": 8192, "heads": 4, "rank": 128, "width": 128},
            "block_sizes": {"chunk": 16},
            "dtype": "BF16 operands and FP32 state; FP32 qualification separately",
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
            "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/59",
            "source_hashes": {
                str(p.relative_to(source.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    source.parent / "moe/boundary_stack_kda.py",
                    source.parent / "moe/kda.py",
                    source.parent / "moe/kda_state_tpu.py",
                    source / "benchmark_head_stack_kda.py",
                    Path(__file__),
                )
            },
        },
    )
    # Import here avoids the training entrypoint importing its own benchmark.
    from experiments.grug.moe_cplm.train_belay import tpu_recurrence  # noqa: PLC0415

    with jax.default_device(jax.local_devices()[0]), patch.object(kda, "_parallel_chunk_recurrence", tpu_recurrence):
        record("checks", validate_boundary_stack())
        xs, state = fixture(128)
        stay = (jnp.broadcast_to(jnp.array([0, 0, 1], jnp.float32), xs[0].shape), *xs[1:])
        record("hard-stay", compare(candidate(stay, state, mixed=True), native(stay, state)))
        xs, state = fixture(8192)
        record("native", benchmark(native, xs, state))
        record("boundary-mixed", benchmark(lambda x, s: candidate(x, s, mixed=True), xs, state))


if __name__ == "__main__":
    main()
