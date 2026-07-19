# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SparseCore-backed embedding lookup gradients for TPU training."""

import functools
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jaxtyping import Array, Float, Int

EmbeddingGradientImplementation = Literal["auto", "xla", "sparsecore"]

_SCATTER_WINDOW_SIZE = 128


def _shape_dtype_struct_with_mesh_metadata(value: Array) -> jax.ShapeDtypeStruct:
    """Describe an array while preserving shard-map variation metadata."""
    return jax.eval_shape(jnp.asarray, value)


def _mark_embedding_gradient_varying(value: Array, varying_axes: tuple[str, ...]) -> Array:
    """Mark a locally accumulated gradient as varying across table replicas."""
    return jax.lax.pcast(value, varying_axes, to="varying")


def _sparsecore_shape_is_supported(ids: Array, updates: Array) -> bool:
    sparse_core_info = plsc.get_sparse_core_info()
    num_sparse_workers = sparse_core_info.num_cores * sparse_core_info.num_subcores
    return (
        ids.ndim == 1
        and updates.ndim == 2
        and updates.shape[0] == ids.shape[0]
        and ids.shape[0] % (_SCATTER_WINDOW_SIZE * num_sparse_workers) == 0
    )


def embedding_scatter_add_reference(
    ids: Int[Array, "n"],  # noqa: F821
    updates: Float[Array, "n d"],
    *,
    num_rows: int,
) -> Float[Array, "v d"]:
    """Return the dense embedding gradient produced by indexed additions."""
    return jnp.zeros((num_rows, updates.shape[-1]), dtype=updates.dtype).at[ids].add(updates)


def _coalesce_embedding_updates(
    ids: Int[Array, "n"],  # noqa: F821
    updates: Float[Array, "n d"],
    *,
    num_rows: int,
) -> tuple[Int[Array, "n"], Float[Array, "n d"]]:  # noqa: F821
    """Sort and sum duplicate embedding updates for an overwrite scatter."""
    order = jnp.argsort(ids)
    sorted_ids = ids[order]
    sorted_updates = updates[order]
    starts = jnp.concatenate((jnp.ones((1,), dtype=jnp.bool_), sorted_ids[1:] != sorted_ids[:-1]))

    def combine(left, right):
        left_updates, left_has_start = left
        right_updates, right_has_start = right
        combined_updates = jnp.where(
            right_has_start[..., None],
            right_updates,
            left_updates + right_updates,
        )
        return combined_updates, left_has_start | right_has_start

    coalesced_updates, _ = jax.lax.associative_scan(combine, (sorted_updates, starts))
    ends = jnp.concatenate((sorted_ids[:-1] != sorted_ids[1:], jnp.ones((1,), dtype=jnp.bool_)))
    dummy_ids = num_rows + jnp.arange(ids.shape[0], dtype=ids.dtype)
    scatter_ids = jnp.where(ends, sorted_ids, dummy_ids)
    return scatter_ids, coalesced_updates


def _sparsecore_embedding_scatter_add(
    ids: Int[Array, "n"],  # noqa: F821
    updates: Float[Array, "n d"],
    *,
    num_rows: int,
) -> Float[Array, "v d"]:
    """Accumulate embedding updates with a coalesced SparseCore scatter."""
    if ids.ndim != 1:
        raise ValueError(f"embedding ids must be rank 1, got {ids.shape}")
    if updates.ndim != 2 or updates.shape[0] != ids.shape[0]:
        raise ValueError(f"embedding updates must have shape ({ids.shape[0]}, D), got {updates.shape}")

    sparse_core_info = plsc.get_sparse_core_info()
    if not _sparsecore_shape_is_supported(ids, updates):
        num_sparse_workers = sparse_core_info.num_cores * sparse_core_info.num_subcores
        raise ValueError(
            f"SparseCore embedding scatter requires the number of ids ({ids.shape[0]}) to be divisible by "
            f"{_SCATTER_WINDOW_SIZE * num_sparse_workers}"
        )
    grid_size = ids.shape[0] // _SCATTER_WINDOW_SIZE
    num_cores = sparse_core_info.num_cores
    num_subcores = sparse_core_info.num_subcores
    num_sparse_workers = num_cores * num_subcores
    windows_per_worker = grid_size // num_sparse_workers
    update_dtype = updates.dtype
    scatter_ids, coalesced_updates = _coalesce_embedding_updates(
        ids.astype(jnp.int32),
        updates.astype(jnp.float32),
        num_rows=num_rows,
    )
    scatter_ids = scatter_ids.reshape(1, -1)
    output_shape = (num_rows + ids.shape[0], updates.shape[-1])
    output = jnp.zeros(output_shape, dtype=jnp.float32)

    def worker_window(core, subcore, step):
        worker = core * num_subcores + subcore
        return worker * windows_per_worker + step

    def kernel(ids_vmem_ref, updates_vmem_ref, output_input_hbm_ref, output_hbm_ref):
        del output_input_hbm_ref
        pltpu.sync_copy(
            updates_vmem_ref,
            output_hbm_ref.at[ids_vmem_ref.at[0]],
        )

    scatter_bytes = ids.shape[0] * (
        jnp.dtype(jnp.int32).itemsize + 2 * updates.shape[-1] * jnp.dtype(jnp.float32).itemsize
    )
    result = pl.pallas_call(
        kernel,
        out_shape=_shape_dtype_struct_with_mesh_metadata(output),
        grid=(num_cores, num_subcores, windows_per_worker),
        in_specs=(
            pl.BlockSpec(
                (1, _SCATTER_WINDOW_SIZE),
                lambda core, subcore, step: (0, worker_window(core, subcore, step)),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(
                (_SCATTER_WINDOW_SIZE, updates.shape[-1]),
                lambda core, subcore, step: (worker_window(core, subcore, step), 0),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        input_output_aliases={2: 0},
        name="over_encoding_embedding_scatter_add",
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.CoreType.SC_VECTOR_SUBCORE,
            dimension_semantics=(pltpu.CORE_PARALLEL, pltpu.SUBCORE_PARALLEL, pltpu.PARALLEL),
            use_tc_tiling_on_sc=False,
        ),
        cost_estimate=pl.CostEstimate(
            flops=0,
            transcendentals=0,
            bytes_accessed=scatter_bytes,
        ),
    )(scatter_ids, coalesced_updates, output)
    return result[:num_rows].astype(update_dtype)


def embedding_scatter_add(
    ids: Int[Array, "n"],  # noqa: F821
    updates: Float[Array, "n d"],
    *,
    num_rows: int,
    implementation: EmbeddingGradientImplementation = "auto",
    varying_axes: tuple[str, ...] = (),
) -> Float[Array, "v d"]:
    """Accumulate embedding gradients with an explicit backend choice."""
    if implementation == "auto":
        if jax.default_backend() == "tpu" and _sparsecore_shape_is_supported(ids, updates):
            implementation = "sparsecore"
        else:
            implementation = "xla"
    if implementation == "xla":
        return embedding_scatter_add_reference(ids, updates, num_rows=num_rows)
    if implementation == "sparsecore":
        if jax.default_backend() != "tpu":
            raise ValueError("SparseCore embedding gradients require a TPU backend")
        gradient = _sparsecore_embedding_scatter_add(ids, updates, num_rows=num_rows)
        return _mark_embedding_gradient_varying(gradient, varying_axes)
    raise ValueError(f"Unknown embedding gradient implementation: {implementation}")


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def embedding_lookup(
    table: Float[Array, "v d"],
    ids: Int[Array, "..."],
    implementation: EmbeddingGradientImplementation = "auto",
    gradient_varying_axes: tuple[str, ...] = (),
) -> Float[Array, "... d"]:
    """Gather rows while dispatching the dense gradient to the selected backend."""
    return table[ids]


def _embedding_lookup_fwd(table, ids, implementation, gradient_varying_axes):
    del implementation, gradient_varying_axes
    return table[ids], (ids, table.shape)


def _embedding_lookup_bwd(implementation, gradient_varying_axes, residuals, output_gradient):
    ids, table_shape = residuals
    flat_ids = ids.reshape(-1)
    flat_updates = output_gradient.reshape((flat_ids.shape[0], table_shape[-1]))
    table_gradient = embedding_scatter_add(
        flat_ids,
        flat_updates,
        num_rows=table_shape[0],
        implementation=implementation,
        varying_axes=gradient_varying_axes,
    )
    return table_gradient, None


embedding_lookup.defvjp(_embedding_lookup_fwd, _embedding_lookup_bwd)
