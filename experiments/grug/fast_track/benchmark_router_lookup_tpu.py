# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare exact unique router selections with gather and bitwise reductions."""

import argparse
import hashlib
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
import numpy as np

from experiments.grug.fast_track.router_lookup import selected_logits

IMPLEMENTATIONS = ("reference", "broadcast", "unrolled")
DISTRIBUTIONS = ("uniform", "concentrated", "ties", "signed-zero")


def lookup(logits, selected, *, implementation):
    if implementation == "reference":
        return jnp.take_along_axis(logits, selected, axis=-1)
    return selected_logits(logits, selected, logits.shape[-1], implementation)


def run(logits, selected, cotangent, *, implementation, mode, expert_major):
    if expert_major:
        logits = logits.T
    fn = partial(lookup, selected=selected, implementation=implementation)
    if mode == "forward":
        return (fn(logits),)
    output, pullback = jax.vjp(fn, logits)
    gradient = pullback(cotangent)[0]
    return output, gradient.T if expert_major else gradient


def fixture(seed, tokens, experts, distribution):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(tokens, experts)).astype(np.float32)
    bias = rng.normal(size=(experts,)).astype(np.float32) * 0.2
    if distribution == "concentrated":
        bias[:8] += 32
    elif distribution == "ties":
        logits = np.round(logits)
        bias[:] = 0
    elif distribution == "signed-zero":
        logits[:, ::2] = 0.0
        logits[:, 1::2] = -0.0
        bias[:] = 0
    scores = logits + bias
    selected = np.argsort(-scores, axis=-1, kind="stable")[:, :8].astype(np.int32)
    assert np.all(np.diff(np.sort(selected, axis=-1), axis=-1) > 0)
    cotangent = rng.normal(size=(tokens, 8)).astype(np.float32)
    cotangent.flat[::17] = -0.0
    cotangent.flat[1::17] = 0.0
    rows = np.arange(tokens)[:, None]
    output = logits[rows, selected]
    gradient = np.zeros_like(logits)
    gradient[rows, selected] = np.where(cotangent == 0, np.float32(0), cotangent)
    return (logits, selected, cotangent), (output, gradient)


def check(values, expected):
    return [
        {
            "exact": bool(np.array_equal(np.asarray(a).view(np.uint32), b.view(np.uint32))),
            "finite": bool(np.isfinite(np.asarray(a)).all()),
            "max_abs_error": float(np.max(np.abs(np.asarray(a) - b))),
        }
        for a, b in zip(values, expected, strict=True)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--expert-major", action="store_true")
    args = parser.parse_args()
    device = jax.local_devices(backend="cpu" if args.cpu_check else "tpu")[0]
    if not args.cpu_check and device.device_kind != "TPU v4":
        raise ValueError("Use a TPU v4 chip")
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Artifacts must stay in us-central2")
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / "router-lookup"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(args.artifact_prefix)
    report = {
        "revision": args.revision,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "router_lookup_sha256": hashlib.sha256(Path(__file__).with_name("router_lookup.py").read_bytes()).hexdigest(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "device_kind": device.device_kind,
        "cpu_check": args.cpu_check,
        "expert_major": args.expert_major,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "status": "running",
        "numerics": [],
        "compilation": [],
        "timings": [],
    }

    def save():
        path = folder / "router-lookup.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    compiled = {}
    shapes = [(tokens, experts) for tokens in (32, 128) for experts in (16, 512)]
    if not args.cpu_check:
        shapes.append((32768, 512))
    save()
    with jax.default_device(device):
        for tokens, experts in shapes:
            for seed in (0, 1, 2):
                for distribution in DISTRIBUTIONS:
                    host, expected = fixture(seed, tokens, experts, distribution)
                    case = dict(tokens=tokens, experts=experts, seed=seed, distribution=distribution)
                    if tokens <= 128:
                        path = folder / f"fixture-{tokens}-{experts}-{seed}-{distribution}.npz"
                        np.savez(path, logits=host[0], selected=host[1], cotangent=host[2])
                        fs.put_file(str(path), f"{prefix}/{path.name}")
                    if args.expert_major:
                        host = (np.ascontiguousarray(host[0].T), *host[1:])
                        expected = (expected[0], expected[1].T)
                    arrays = tuple(jax.device_put(a, device) for a in host)
                    modes = ("forward", "forward-backward") if tokens == 32768 else ("forward-backward",)
                    for mode in modes:
                        row = case | {"mode": mode, "implementations": {}}
                        for name in IMPLEMENTATIONS:
                            key = (tokens, experts, name, mode)
                            if key not in compiled:
                                fn = jax.jit(
                                    partial(run, implementation=name, mode=mode, expert_major=args.expert_major)
                                )
                                start = time.monotonic()
                                executable = fn.lower(*arrays).compile()
                                compiled[key] = executable
                                report["compilation"].append(
                                    dict(
                                        tokens=tokens,
                                        experts=experts,
                                        implementation=name,
                                        mode=mode,
                                        seconds=time.monotonic() - start,
                                        temporary_bytes=executable.memory_analysis().temp_size_in_bytes,
                                    )
                                )
                                if tokens == 32768:
                                    path = folder / f"hlo-{name}-{mode}.txt"
                                    path.write_text(executable.as_text())
                                    fs.put_file(str(path), f"{prefix}/{path.name}")
                            values = jax.block_until_ready(compiled[key](*arrays))
                            row["implementations"][name] = check(values, expected[: len(values)])
                        row["passed"] = all(
                            m["exact"] and m["finite"] for ms in row["implementations"].values() for m in ms
                        )
                        report["numerics"].append(row)
                        save()
                        print(json.dumps(case | {"mode": mode, "passed": row["passed"]}), flush=True)
                        if tokens != 32768:
                            continue
                        for name in IMPLEMENTATIONS:
                            for _ in range(2):
                                jax.block_until_ready(compiled[(tokens, experts, name, mode)](*arrays))
                        for repeat in range(8):
                            names = IMPLEMENTATIONS if repeat % 2 == 0 else IMPLEMENTATIONS[::-1]
                            for name in names:
                                start = time.monotonic()
                                jax.block_until_ready(compiled[(tokens, experts, name, mode)](*arrays))
                                report["timings"].append(
                                    case
                                    | dict(
                                        mode=mode, implementation=name, repeat=repeat, seconds=time.monotonic() - start
                                    )
                                )
                        save()
    report["numerical_gate"] = all(r["passed"] for r in report["numerics"])
    report["qualification"] = {}
    if not args.cpu_check:
        for name in IMPLEMENTATIONS[1:]:
            cases = []
            for seed in (0, 1, 2):
                for mode in ("forward", "forward-backward"):
                    medians = {
                        implementation: statistics.median(
                            r["seconds"]
                            for r in report["timings"]
                            if r["seed"] == seed
                            and r["distribution"] == "uniform"
                            and r["mode"] == mode
                            and r["implementation"] == implementation
                        )
                        for implementation in ("reference", name)
                    }
                    ratio = medians[name] / medians["reference"]
                    cases.append(
                        dict(
                            seed=seed,
                            mode=mode,
                            medians=medians,
                            ratio=ratio,
                            passed=ratio <= (0.95 if mode == "forward" else 1.01),
                        )
                    )
            numerics_passed = all(
                m["exact"] and m["finite"]
                for r in report["numerics"]
                for impl in ("reference", name)
                for m in r["implementations"][impl]
            )
            report["qualification"][name] = dict(
                passed=numerics_passed and all(r["passed"] for r in cases), numerical_gate=numerics_passed, cases=cases
            )
    report["status"] = "completed"
    save()
    print(json.dumps({"numerical_gate": report["numerical_gate"], "qualification": report["qualification"]}), flush=True)
    if not report["numerical_gate"]:
        raise RuntimeError("Router lookup differs from the independent bitwise oracle")


if __name__ == "__main__":
    main()
