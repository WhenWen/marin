# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero-update qualification of input-dependent orthogonal boundary mixing."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe import kda
from experiments.grug.moe_cplm.benchmark_frame_kda import write_record
from experiments.grug.moe_cplm.benchmark_head_stack_kda import benchmark, native
from experiments.grug.moe_cplm.boundary_stack_validation import candidate as matrix_candidate
from experiments.grug.moe_cplm.boundary_stack_validation import compare, evaluate
from experiments.grug.moe_cplm.boundary_stack_validation import fixture as stack_fixture
from experiments.grug.moe_cplm.boundary_stack_validation import literal as matrix_literal
from experiments.grug.moe_cplm.orthogonal_boundary_kda_model import cayley_matrix


def fixture(length, rank=128, chunk=16):
    xs, state = stack_fixture(length, rank, chunk)
    rng = np.random.default_rng(814 + length)
    coordinates = jnp.asarray(rng.normal(scale=0.15, size=(*xs[0].shape[:-1], 6)), jnp.float32)
    return (coordinates, *xs[1:]), state


def matrix_inputs(xs):
    matrix = cayley_matrix(xs[0])
    return (matrix.reshape(*matrix.shape[:-2], 16), *xs[1:])


def candidate(xs, state, chunk=16, mixed=False, interpret=False, reference=False):
    return matrix_candidate(matrix_inputs(xs), state, chunk, mixed, interpret, reference)


def literal(xs, state, chunk=16):
    return matrix_literal(matrix_inputs(xs), state, chunk)


def controller_candidate(inputs, state):
    boundary, weight, bias, xs = inputs
    coordinates = jnp.einsum("bcd,dk->bck", boundary, weight, precision=jax.lax.Precision.HIGHEST) + bias
    return candidate((coordinates, *xs[1:]), state, mixed=True)


def validate_orthogonal_boundary(rank=128, interpret=False):
    xs, state = fixture(32, rank)
    with jax.default_device(jax.local_devices(backend="cpu")[0]), jax.enable_x64():
        cpu_xs, cpu_state = jax.tree.map(lambda x: jnp.asarray(np.asarray(x), jnp.float64), (xs, state))
        oracle = jax.device_get(evaluate(literal, cpu_xs, cpu_state))
    exact = evaluate(lambda x, s: candidate(x, s, interpret=interpret), xs, state)
    mixed = evaluate(lambda x, s: candidate(x, s, mixed=True, interpret=interpret), xs, state)
    scan = evaluate(lambda x, s: candidate(x, s, mixed=True, reference=True), xs, state)
    diagnostics = {
        "fp32": compare(exact, oracle, False),
        "mixed_vs_literal": compare(mixed, oracle, False),
        "mixed_vjp": compare(mixed, scan, False),
    }
    print(json.dumps({"orthogonal_diagnostics": diagnostics}), flush=True)
    compare(exact, oracle)
    compare(mixed, scan)
    if any(r["relative_l2"] > 0.05 and r["max_abs"] > 1e-4 for r in diagnostics["mixed_vs_literal"]):
        raise ValueError("Mixed approximation exceeds the frozen 5% relative-L2 / 1e-4 absolute bound")
    matrix = np.asarray(cayley_matrix(xs[0]))
    error = float(np.max(np.abs(np.swapaxes(matrix, -1, -2) @ matrix - np.eye(4))))
    if error > 1e-5:
        raise ValueError("Cayley orthogonality exceeds 1e-5")
    return {
        "training_updates": 0,
        "state_bytes": state.size * state.dtype.itemsize,
        "orthogonality_max_abs": error,
        **diagnostics,
    }


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
        raise ValueError("Orthogonal boundary qualification requires TPU v4")
    identity = uuid.uuid4().hex
    directory = Path("/tmp/orthogonal-boundary") / identity
    root = args.artifact_root.rstrip("/") + "/" + identity
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
            "device_type": jax.local_devices()[0].device_kind,
            "benchmark_device_count": 1,
            "allocated_chips": len(jax.devices()),
            "shape": [1, 8192, 4, 128, 128],
            "chunk": 16,
            "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/59",
            "source_hashes": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in (
                    "../moe/boundary_stack_kda.py",
                    "orthogonal_boundary_kda_model.py",
                    "orthogonal_boundary_validation.py",
                    "boundary_stack_validation.py",
                )
            },
        },
    )
    # The trainer imports qualification only during startup, avoiding a cycle.
    from experiments.grug.moe_cplm.train_belay import tpu_recurrence  # noqa: PLC0415

    with jax.default_device(jax.local_devices()[0]), patch.object(kda, "_parallel_chunk_recurrence", tpu_recurrence):
        record("checks", validate_orthogonal_boundary())
        xs, state = fixture(128)
        identity_inputs = (jnp.zeros_like(xs[0]), *xs[1:])
        record("identity-native", compare(candidate(identity_inputs, state, mixed=True), native(xs, state)))
        xs, state = fixture(8192)
        record("native", benchmark(native, xs, state))
        record("orthogonal-mixed", benchmark(lambda x, s: candidate(x, s, mixed=True), xs, state))
        rng = np.random.default_rng(815)
        controller = (
            jnp.asarray(rng.normal(size=(1, 512, 512)), jnp.float32),
            jnp.asarray(rng.normal(scale=0.15 / 512**0.5, size=(512, 6)), jnp.float32),
            jnp.zeros(6, jnp.float32),
            xs,
        )
        record("controller-inclusive", benchmark(controller_candidate, controller, state))


if __name__ == "__main__":
    main()
