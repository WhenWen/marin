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
    update_dtype = updates.dtype
    scatter_ids, coalesced_updates = _coalesce_embedding_updates(
        ids.astype(jnp.int32),
        updates.astype(jnp.float32),
        num_rows=num_rows,
    )
    scatter_ids = scatter_ids.reshape(1, -1)
    output = jnp.zeros((num_rows + ids.shape[0], updates.shape[-1]), dtype=jnp.float32)
    ids_ref = jax.new_ref(scatter_ids, memory_space=pltpu.HBM)
    updates_ref = jax.new_ref(coalesced_updates, memory_space=pltpu.HBM)
    output_ref = jax.new_ref(output, memory_space=pltpu.HBM)
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=sparse_core_info.num_cores,
        num_subcores=sparse_core_info.num_subcores,
    )

    @pl.core_map(
        mesh,
        name="over_encoding_embedding_scatter_add",
    )
    def kernel():
        def scatter_body(ids_vmem_ref, updates_vmem_ref):
            pltpu.sync_copy(
                updates_vmem_ref,
                output_ref.at[ids_vmem_ref.at[0]],
            )

        pltpu.emit_pipeline(
            scatter_body,
            grid=(grid_size,),
            in_specs=(
                pl.BlockSpec(
                    (1, _SCATTER_WINDOW_SIZE),
                    lambda step: (0, step),
                ),
                pl.BlockSpec(
                    (_SCATTER_WINDOW_SIZE, updates.shape[-1]),
                    lambda step: (step, 0),
                ),
            ),
            out_specs=(),
            core_axis_name=("core", "subcore"),
            dimension_semantics=(pltpu.PARALLEL,),
        )(ids_ref, updates_ref)

    return jax.freeze(output_ref)[:num_rows].astype(update_dtype)


def embedding_scatter_add(
    ids: Int[Array, "n"],  # noqa: F821
    updates: Float[Array, "n d"],
    *,
    num_rows: int,
    implementation: EmbeddingGradientImplementation = "auto",
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
        return _sparsecore_embedding_scatter_add(ids, updates, num_rows=num_rows)
    raise ValueError(f"Unknown embedding gradient implementation: {implementation}")


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def embedding_lookup(
    table: Float[Array, "v d"],
    ids: Int[Array, "..."],
    implementation: EmbeddingGradientImplementation = "auto",
) -> Float[Array, "... d"]:
    """Gather rows while dispatching the dense gradient to the selected backend."""
    return table[ids]


def _embedding_lookup_fwd(table, ids, implementation):
    del implementation
    return table[ids], (ids, table.shape)


def _embedding_lookup_bwd(implementation, residuals, output_gradient):
    ids, table_shape = residuals
    flat_ids = ids.reshape(-1)
    flat_updates = output_gradient.reshape((flat_ids.shape[0], table_shape[-1]))
    table_gradient = embedding_scatter_add(
        flat_ids,
        flat_updates,
        num_rows=table_shape[0],
        implementation=implementation,
    )
    return table_gradient, None


embedding_lookup.defvjp(_embedding_lookup_fwd, _embedding_lookup_bwd)
