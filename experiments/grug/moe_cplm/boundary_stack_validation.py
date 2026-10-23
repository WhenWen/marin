# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero-update checks and fixed-shape timing for boundary head mixing."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe import boundary_stack_kda as stack
from experiments.grug.moe import kda
from experiments.grug.moe_cplm.benchmark_frame_kda import write_record
from experiments.grug.moe_cplm.benchmark_head_stack_kda import benchmark, fixture as token_fixture
from experiments.grug.moe_cplm.benchmark_head_stack_kda import native, objective, operator_inputs


def fixture(length, rank=128, chunk=16):
    xs, state = token_fixture(length, rank=rank, width=rank, mixed=True)
    return (xs[0][:, chunk-1::chunk], *xs[1:]), state


def candidate(xs, state, chunk=16, mixed=False, interpret=False, reference=False):
    args, segments = operator_inputs(xs)
    out, final = stack.chunk_kda(
        *args, actions=xs[0], initial_state=state, use_qk_l2norm=False, segment_ids=segments,
        chunk_size=chunk, matmul_dtype=jnp.bfloat16 if mixed else jnp.float32,
        interpret=interpret, reference=reference,
    )
    return jnp.swapaxes(out, 1, 2), final


def literal(xs, initial, chunk=16):
    actions, q, k, v, g, beta, keep = xs
    tokens = tuple(jnp.swapaxes(x, 0, 1) for x in (q, k, v, g, beta, jax.lax.stop_gradient(keep)))

    def step(state, token):
        index, (q, k, v, g, beta, keep) = token
        state = state * (jnp.exp(g) * keep[:, None, None])[..., None]
        prediction = jnp.einsum("bhr,bhrv->bhv", k, state, precision=jax.lax.Precision.HIGHEST)
        state += (beta[..., None] * k)[..., None] * (v-prediction)[..., None, :]
        out = jnp.einsum("bhr,bhrv->bhv", q, state, precision=jax.lax.Precision.HIGHEST)
        state = jax.lax.cond((index+1) % chunk == 0,
                             lambda s: jax.vmap(stack.mix)(s, jnp.broadcast_to(
                                 actions[:, index//chunk, :, None], (s.shape[0], 3, s.shape[-1]))),
                             lambda s: s, state)
        return state, out

    final, out = jax.lax.scan(step, initial, (jnp.arange(q.shape[1]), tokens))
    return jnp.swapaxes(out, 0, 1), final


def evaluate(fn, xs, state):
    values = jax.jit(fn)(xs, state)
    grads = jax.jit(jax.value_and_grad(lambda x, s: objective(fn, x, s), argnums=(0, 1)))(xs, state)
    return values, grads


def compare(actual, expected, strict=True):
    result = []
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite boundary-stack result")
        result.append({"max_abs": float(np.max(np.abs(a-b))),
                       "relative_l2": float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))})
        if strict:
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
    return result


def validate_boundary_stack(rank=128, interpret=False):
    xs, state = fixture(32, rank)
    exact = lambda x, s: candidate(x, s, interpret=interpret)  # noqa: E731
    mixed = lambda x, s: candidate(x, s, mixed=True, interpret=interpret)  # noqa: E731
    result = {"training_updates": 0, "state_bytes": state.size * state.dtype.itemsize}
    oracle = evaluate(literal, xs, state)
    result["fp32"] = compare(evaluate(exact, xs, state), oracle)
    result["mixed_vs_literal"] = compare(evaluate(mixed, xs, state), oracle, False)
    result["mixed_vjp"] = compare(evaluate(mixed, xs, state),
                                   evaluate(lambda x, s: candidate(x, s, mixed=True, reference=True), xs, state))
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

    record("manifest", {"training_updates": 0, "artifact_root": root, "jax": jax.__version__,
                        "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/59",
                        "source_hashes": {str(p.relative_to(source.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (source.parent / "moe/boundary_stack_kda.py",
                                                    source.parent / "moe/kda.py", source.parent / "moe/kda_state_tpu.py",
                                                    Path(__file__))}})
    # Import here avoids the training entrypoint importing its own benchmark.
    from experiments.grug.moe_cplm.train_belay import tpu_recurrence  # noqa: PLC0415

    with jax.default_device(jax.local_devices()[0]), patch.object(kda, "_parallel_chunk_recurrence", tpu_recurrence):
        record("checks", validate_boundary_stack())
        xs, state = fixture(128)
        stay = (jnp.broadcast_to(jnp.array([0, 0, 1], jnp.float32), xs[0].shape), *xs[1:])
        record("hard-stay", compare(candidate(stay, state, mixed=True), native(stay, state), False))
        xs, state = fixture(8192)
        record("native", benchmark(native, xs, state))
        record("boundary-mixed", benchmark(lambda x, s: candidate(x, s, mixed=True), xs, state))


if __name__ == "__main__":
    main()
