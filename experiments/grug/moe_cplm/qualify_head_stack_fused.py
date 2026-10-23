# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify one fixed C16 fused tree/state algorithm; no parameter sweep."""

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

from experiments.grug.moe import head_stack_kda as stack
from experiments.grug.moe import kda
from experiments.grug.moe.head_stack_state_tpu import chunked
from experiments.grug.moe.head_stack_tree import fused_prepare, prepare_tree
from experiments.grug.moe_cplm.benchmark_frame_kda import write_record
from experiments.grug.moe_cplm.benchmark_head_stack_kda import benchmark, fixture, native, objective
from experiments.grug.moe_cplm.train_belay import tpu_recurrence


def errors(actual, expected, *, strict):
    result = []
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite fused result")
        result.append({"max_abs": float(np.max(np.abs(a-b))),
                       "relative_l2": float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))})
        if strict:
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_pgle", False)
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
    if any(x.device_kind != "TPU v4" for x in jax.local_devices()):
        raise ValueError("Qualification requires TPU v4")
    identity = uuid.uuid4().hex
    root = args.artifact_root.rstrip("/") + "/" + identity
    directory = Path("/tmp/head-stack-fused") / identity
    directory.mkdir(parents=True)

    def record(name, result):
        write_record(directory, root, name + ".json", result)
        print(json.dumps({name: result}), flush=True)

    source = Path(__file__).parent
    record("manifest", {"training_updates": 0, "artifact_root": root, "chunk": 16,
                        "issue_url": "https://github.com/WhenWen/astra-d8-research/issues/59",
                        "jax": jax.__version__, "shape": [1, 8192, 4, 128, 128],
                        "source_hashes": {str(p.relative_to(source.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (source.parent / "moe/head_stack_tree.py",
                                                    source.parent / "moe/head_stack_state_tpu.py",
                                                    source.parent / "moe/head_stack_kda.py", Path(__file__))}})
    with jax.default_device(jax.local_devices()[0]), patch.object(kda, "_parallel_chunk_recurrence", tpu_recurrence):
        inputs, initial = fixture(32, mixed=True)
        split = stack._split(inputs, 16)
        for precision in ("fp32", "mixed"):
            print("Compile fused preparation and reverse: " + precision, flush=True)
            expected, pb = jax.vjp(lambda xs: prepare_tree(xs, precision), split)
            actual, fpb = jax.vjp(lambda xs: fused_prepare(xs, precision), split)
            dy = jax.tree.map(jnp.cos, expected)
            record("prep-" + precision, errors((actual, fpb(dy)), (expected, pb(dy)), strict=precision == "fp32"))
        print("Compile fused state stream and full reverse", flush=True)
        fn = lambda xs, s: chunked(xs, s, 16)  # noqa: E731
        actual = (jax.jit(fn)(inputs, initial),
                  jax.jit(jax.value_and_grad(lambda x, s: objective(fn, x, s), argnums=(0, 1)))(inputs, initial))
        expected = (stack.reference(inputs, initial),
                    jax.jit(jax.value_and_grad(lambda x, s: objective(stack.reference, x, s), argnums=(0, 1)))(
                        inputs, initial))
        record("full-fp32", errors(actual, expected, strict=True))
        mixed_fn = lambda xs, s: chunked(xs, s, 16, "mixed")  # noqa: E731
        mixed = (jax.jit(mixed_fn)(inputs, initial),
                 jax.jit(jax.value_and_grad(lambda x, s: objective(mixed_fn, x, s), argnums=(0, 1)))(inputs, initial))
        record("full-mixed-vs-fp32", errors(mixed, actual, strict=False))
        inputs, initial = fixture(8192)
        print("Fixed-shape timings", flush=True)
        record("native", benchmark(native, inputs, initial))
        record("fused-mixed", benchmark(mixed_fn, inputs, initial))


if __name__ == "__main__":
    main()
