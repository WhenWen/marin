# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare two backward MoE returns with a bit-preserving paired return."""

import argparse
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import ml_dtypes
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig

from experiments.grug.fast_track.belay_tpu import require_tpu_rounding
from experiments.grug.fast_track.benchmark_moe_padded_transport_tpu import (
    DISTRIBUTIONS,
    output_oracle,
    payload,
    transport_fixture,
)
from experiments.grug.fast_track.moe_transport_tpu import (
    pack_return_buffers,
    paired_return_transport,
    unpack_return_buffers,
)

SCALAR_PATTERNS = np.asarray(
    [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0xFFC12345,
        0x7F800001,
        0xFF812345,
        0x3F800000,
        0xBF800000,
    ],
    dtype=np.uint32,
)


def scalar_payload(seed, peer, start, stop, width, dtype, *, initial=False):
    """Create deterministic raw FP32 bits, including nonfinite transport payloads."""
    assert width == 1 and dtype == np.float32
    rows = np.arange(start, stop, dtype=np.uint32)
    bits = rows * np.uint32(1664525) + np.uint32(1013904223)
    bits ^= np.uint32((seed * 1103515245 + peer * 22695477) % 2**32)
    bits ^= bits >> np.uint32(13)
    if initial:
        bits ^= np.uint32(0x91E10DA5)
    special = SCALAR_PATTERNS[(rows // 17 + peer + seed + int(initial)) % len(SCALAR_PATTERNS)]
    bits = np.where(rows % 17 == 0, special, bits)
    return bits.view(np.float32)[:, None]


def bit_check(actual, expected):
    actual = np.ascontiguousarray(actual)
    expected = np.ascontiguousarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    dtype = np.dtype(f"u{actual.dtype.itemsize}")
    different = actual.view(dtype) ^ expected.view(dtype)
    result = dict(mismatched_elements=int(np.count_nonzero(different)), max_xor_bits=int(different.max(initial=0)))
    if actual.dtype == np.float32:
        bits = expected.view(np.uint32)
        exponent = bits & np.uint32(0x7F800000)
        fraction = bits & np.uint32(0x007FFFFF)
        classes = {
            "nan": (exponent == 0x7F800000) & (fraction != 0),
            "infinity": (exponent == 0x7F800000) & (fraction == 0),
            "subnormal": (exponent == 0) & (fraction != 0),
            "zero": (exponent == 0) & (fraction == 0),
            "normal": (exponent != 0) & (exponent != 0x7F800000),
        }
        result["mismatch_classes"] = {
            label: int(np.count_nonzero(mask & (different != 0))) for label, mask in classes.items()
        }
        indices = np.flatnonzero(different)[:16]
        result["examples"] = [
            dict(
                index=int(i), expected_bits=hex(int(bits.flat[i])), actual_bits=hex(int(actual.view(np.uint32).flat[i]))
            )
            for i in indices
        ]
    return result


def cpu_packing_checks():
    results = []
    for rows in (17, 129, 1024):
        for width in (8, 16, 256, 512):
            for seed in (0, 1, 2):
                vector = payload(seed, 0, 0, rows, width, ml_dtypes.bfloat16)
                scalar = scalar_payload(seed, 0, 0, rows, 1, np.float32)
                expected = np.zeros((rows, 128 * ((width + 131) // 128)), dtype=np.uint16)
                expected[:, :width] = vector.view(np.uint16)
                scalar_bytes = np.ascontiguousarray(scalar).view(np.uint8).reshape(rows, 4)
                expected[:, width : width + 4] = scalar_bytes.astype(np.uint16) | np.uint16(0x3F00)
                packed = jax.jit(pack_return_buffers)(jnp.asarray(vector), jnp.asarray(scalar))
                permutation = np.random.default_rng(seed + rows).permutation(rows)
                restored = jax.jit(partial(unpack_return_buffers, width=width))(packed[permutation])
                checks = [
                    bit_check(np.asarray(packed), expected.view(ml_dtypes.bfloat16)),
                    bit_check(np.asarray(restored[0]), vector[permutation]),
                    bit_check(np.asarray(restored[1]), scalar[permutation]),
                ]
                results.append(dict(rows=rows, width=width, seed=seed, checks=checks))
                if any(c["mismatched_elements"] for c in checks):
                    raise ValueError(f"Packing mismatch: {results[-1]}")
    return results


def mapped_return(*arrays, mesh, implementation):
    def local(*values):
        vector, vector_output, scalar, scalar_output, *metadata = (x[0] for x in values)
        if implementation == "reference":
            result = (
                jax.lax.ragged_all_to_all(vector, vector_output, *metadata, axis_name="expert"),
                jax.lax.ragged_all_to_all(scalar, scalar_output, *metadata, axis_name="expert"),
            )
        else:
            result = paired_return_transport(vector, vector_output, scalar, scalar_output, *metadata, axis_name="expert")
        return tuple(x[None] for x in result)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(*(P("expert", None, None),) * 4, *(P("expert", None),) * 4),
        out_specs=(P("expert", None, None), P("expert", None, None)),
    )(*arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    if not args.artifact_prefix.startswith("gs://marin-us-central2/"):
        raise ValueError("Keep artifacts in us-central2")
    if not args.cpu_check:
        require_tpu_rounding()
        DistributedConfig().initialize()
    devices = jax.devices()
    if args.cpu_check:
        assert all(d.platform == "cpu" for d in devices)
    elif len(devices) != 16 or any(d.device_kind != "TPU v4" for d in devices):
        raise ValueError("Use one TPU v4-32 slice")
    rank = jax.process_index()
    name = "cpu-packing" if args.cpu_check else f"rank-{rank}"
    folder = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")) / f"paired-return-{name}"
    folder.mkdir(parents=True, exist_ok=True)
    fs, prefix = fsspec.core.url_to_fs(f"{args.artifact_prefix}/{name}")
    report = dict(
        revision=args.revision,
        rank=rank,
        backend=devices[0].platform,
        status="running",
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        source_hashes={
            filename: hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
            for filename in (
                Path(__file__).name,
                "benchmark_moe_padded_transport_tpu.py",
                "moe_padded_transport_tpu.py",
                "moe_transport_tpu.py",
            )
        },
        numerics=[],
        compilation=[],
        timings=[],
    )

    def save():
        path = folder / "paired-return.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fs.put_file(str(path), f"{prefix}/{path.name}")

    save()
    if args.cpu_check:
        report["numerics"] = cpu_packing_checks()
        report["status"] = "completed"
        save()
        print(json.dumps(report, indent=2))
        return

    mesh = Mesh(np.asarray(devices), ("expert",), axis_types=(AxisType.Explicit,))
    cache = {}
    with jax.set_mesh(mesh):
        for rows, experts, widths in ((1024, 64, (8, 16)), (262144, 512, (256, 512))):
            full = rows == 262144
            for seed in (0, 1, 2):
                for distribution in DISTRIBUTIONS:
                    if full and seed and distribution != "uniform":
                        continue
                    for chunk in (0, 1):
                        counts, clipped, _dispatch, metadata, capacity, _overflow = transport_fixture(
                            seed, distribution, 16, rows, experts, chunk
                        )
                        for width in widths:
                            case = dict(
                                rows=rows,
                                experts=experts,
                                width=width,
                                seed=seed,
                                distribution=distribution,
                                chunk=chunk,
                            )
                            arrays = []
                            for generator, channels, dtype in (
                                (payload, width, ml_dtypes.bfloat16),
                                (scalar_payload, 1, np.float32),
                            ):
                                for initial in (False, True):
                                    length = rows if initial else capacity

                                    def callback(
                                        index,
                                        *,
                                        generator=generator,
                                        channels=channels,
                                        dtype=dtype,
                                        length=length,
                                        initial=initial,
                                        seed=seed,
                                    ):
                                        return np.stack(
                                            [
                                                generator(
                                                    seed,
                                                    peer,
                                                    index[1].start or 0,
                                                    index[1].stop or length,
                                                    channels,
                                                    dtype,
                                                    initial=initial,
                                                )
                                                for peer in range(index[0].start, index[0].stop)
                                            ]
                                        )

                                    arrays.append(
                                        jax.make_array_from_callback(
                                            (16, length, channels),
                                            NamedSharding(mesh, P("expert", None, None)),
                                            callback,
                                        )
                                    )
                            arrays.extend(
                                jax.make_array_from_callback(
                                    a.shape, NamedSharding(mesh, P("expert", None)), lambda index, a=a: a[index]
                                )
                                for a in metadata
                            )
                            result_row = {
                                **case,
                                "metadata_sha256": (
                                    hashlib.sha256(
                                        b"".join(a.tobytes() for a in (counts, clipped, *metadata))
                                    ).hexdigest()
                                ),
                                "implementations": {},
                            }
                            for implementation in ("reference", "paired"):
                                key = (rows, width, implementation)
                                if key not in cache:
                                    fn = jax.jit(partial(mapped_return, mesh=mesh, implementation=implementation))
                                    start = time.monotonic()
                                    cache[key] = fn.lower(*arrays).compile()
                                    report["compilation"].append(
                                        dict(
                                            key=key,
                                            seconds=time.monotonic() - start,
                                            temporary_bytes=cache[key].memory_analysis().temp_size_in_bytes,
                                        )
                                    )
                                    if rank == 0:
                                        path = folder / ("-".join(map(str, key)) + "-hlo.txt")
                                        path.write_text(cache[key].as_text())
                                        fs.put_file(str(path), f"{prefix}/{path.name}")
                                actual = jax.block_until_ready(cache[key](*arrays))
                                checks = []
                                for value, generator, channels, dtype in (
                                    (actual[0], payload, width, ml_dtypes.bfloat16),
                                    (actual[1], scalar_payload, 1, np.float32),
                                ):
                                    shard_checks = []
                                    for shard in value.addressable_shards:
                                        expected = output_oracle(
                                            seed,
                                            shard.index[0].start,
                                            channels,
                                            dtype,
                                            counts,
                                            clipped,
                                            capacity,
                                            rows,
                                            "return",
                                            payload_fn=generator,
                                        )
                                        shard_checks.append(bit_check(np.asarray(shard.data)[0], expected))
                                    checks.append(
                                        dict(
                                            mismatched_elements=sum(c["mismatched_elements"] for c in shard_checks),
                                            max_xor_bits=max(c["max_xor_bits"] for c in shard_checks),
                                            shards=shard_checks,
                                        )
                                    )
                                result_row["implementations"][implementation] = checks
                            result_row["passed"] = all(
                                c["mismatched_elements"] == 0
                                for checks in result_row["implementations"].values()
                                for c in checks
                            )
                            report["numerics"].append(result_row)
                            save()
                            if not result_row["passed"]:
                                raise ValueError(f"Paired return mismatch: {result_row}")
                            if full and distribution == "uniform":
                                for implementation in ("reference", "paired"):
                                    for _ in range(4):
                                        jax.block_until_ready(cache[(rows, width, implementation)](*arrays))
                                for repeat in range(16):
                                    order = ("reference", "paired") if repeat % 2 == 0 else ("paired", "reference")
                                    for implementation in order:
                                        multihost_utils.sync_global_devices(
                                            f"{rows}-{width}-{seed}-{chunk}-{repeat}-{implementation}"
                                        )
                                        start = time.monotonic()
                                        jax.block_until_ready(cache[(rows, width, implementation)](*arrays))
                                        report["timings"].append(
                                            {
                                                **case,
                                                "implementation": implementation,
                                                "repeat": repeat,
                                                "seconds": time.monotonic() - start,
                                            }
                                        )
                                save()
                            if not full and rank == 0:
                                path = folder / ("fixture-" + "-".join(map(str, case.values())) + ".npz")
                                data = {
                                    "counts": counts,
                                    "clipped": clipped,
                                    **dict(
                                        zip(
                                            ("input_offsets", "send_sizes", "output_offsets", "recv_sizes"),
                                            metadata,
                                            strict=True,
                                        )
                                    ),
                                }
                                for label, generator, channels, dtype in (
                                    ("vector", payload, width, ml_dtypes.bfloat16),
                                    ("scalar", scalar_payload, 1, np.float32),
                                ):
                                    for initial in (False, True):
                                        length = rows if initial else capacity
                                        data[f"{label}_{'initial' if initial else 'source'}"] = np.stack(
                                            [
                                                generator(seed, peer, 0, length, channels, dtype, initial=initial)
                                                for peer in range(16)
                                            ]
                                        )
                                np.savez_compressed(path, **data)
                                fs.put_file(str(path), f"{prefix}/{path.name}")
                    print(
                        json.dumps(
                            dict(rows=rows, seed=seed, distribution=distribution, checked=len(report["numerics"]))
                        ),
                        flush=True,
                    )
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
