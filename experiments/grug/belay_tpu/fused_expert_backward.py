# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fuse all four expert backward products within each device-local row tile."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata
from jaxtyping import Array, Float
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

from experiments.grug.belay_tpu.fused_row_relu2 import fused_row_products


def _dot(left, right):
    return lax.dot_general(left, right, dimension_numbers=(((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)


def _tile_backward_math(x, pre, dy, up, down):
    d_hidden = _dot(dy, down.T).astype(x.dtype)
    derivative = (jnp.asarray(2, x.dtype) * pre).astype(x.dtype)
    d_pre = jnp.where(pre > 0, (d_hidden * derivative).astype(x.dtype), 0)
    post = jnp.square(jnp.maximum(pre, 0)).astype(x.dtype)
    return _dot(d_pre, up.T).astype(x.dtype), _dot(x.T, d_pre), _dot(post.T, dy)


@partial(jax.jit, static_argnames=("partitions", "tile_rows", "interpret"))
def fused_expert_gradients(
    x: Float[Array, "rows width"],
    pre: Float[Array, "rows hidden"],
    dy: Float[Array, "rows width"],
    up: Float[Array, "experts width hidden"],
    down: Float[Array, "experts hidden width"],
    groups: jax.Array,
    *,
    partitions: int = 2,
    tile_rows: int = 512,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return input and expert-weight gradients without materializing d_pre.

    Inputs are fully device-local, including the expert bank. Row partitions
    own disjoint complete tiles. Only experts crossing partition boundaries
    have multiple partial gradients. Products accumulate in FP32 and each
    partial is stored in the weight dtype before the boundary sum. No
    collectives run in this wrapper.
    """
    if not interpret and jax.default_backend() != "tpu":
        raise ValueError("Fused expert gradients require TPU or explicit interpreter mode")
    m, width = x.shape
    experts, up_width, hidden = up.shape
    if partitions < 1 or m % partitions:
        raise ValueError("Expected complete row partitions")
    part_rows = m // partitions
    tm = min(tile_rows, part_rows)
    if (
        tm < 1
        or part_rows % tm
        or up_width != width
        or down.shape != (experts, hidden, width)
        or groups.shape != (experts,)
        or pre.shape != (m, hidden)
        or dy.shape != x.shape
        or any(value.dtype != x.dtype for value in (pre, dy, up, down))
        or x.dtype not in (jnp.float32, jnp.bfloat16)
    ):
        raise ValueError("Expected complete row tiles and matching local expert shapes/dtypes")
    if not interpret and (tm % 128 or width % 128 or hidden % 128):
        raise ValueError("TPU expert tile dimensions must be multiples of 128")
    offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(groups)))
    cuts = jnp.arange(1, partitions, dtype=jnp.int32) * part_rows
    cut_experts = jnp.searchsorted(offsets[:-1], cuts, side="right") - 1
    expanded_starts, expert_ids = lax.sort_key_val(
        jnp.concatenate((offsets[:-1], cuts)),
        jnp.concatenate((jnp.arange(experts, dtype=jnp.int32), cut_experts)),
        is_stable=True,
    )
    expanded_offsets = jnp.concatenate((expanded_starts, offsets[-1:]))
    expanded_experts = experts + partitions - 1
    cut_slots = jnp.searchsorted(offsets[:-1], cuts, side="right") + jnp.arange(partitions - 1)
    part_starts = jnp.concatenate((jnp.zeros((1,), jnp.int32), cut_slots))
    part_ends = jnp.concatenate((cut_slots, jnp.asarray([expanded_experts], jnp.int32)))
    original_slots = jnp.arange(experts) + jnp.sum(cuts[:, None] < offsets[None, :-1], axis=0)

    def metadata_for(part):
        local_offsets = jnp.clip(expanded_offsets - part * part_rows, 0, part_rows)
        return make_group_metadata(
            group_sizes=jnp.diff(local_offsets),
            m=part_rows,
            tm=tm,
            start_group=part_starts[part],
            num_nonzero_groups=part_ends[part] - part_starts[part],
            visit_empty_groups=True,
        )

    metadata, counts = jax.vmap(metadata_for)(jnp.arange(partitions, dtype=jnp.int32))

    def safe_slot(part, slot, counts):
        return jnp.minimum(slot, counts[part] - 1)

    def kernel(
        meta,
        counts,
        expert_ids,
        x_ref,
        pre_ref,
        dy_ref,
        up_ref,
        down_ref,
        dx_out,
        up_out,
        down_out,
        assembled_dx,
        accumulated_up,
        accumulated_down,
    ):
        del expert_ids
        part, slot = pl.program_id(0), pl.program_id(1)
        index = safe_slot(part, slot, counts)
        offsets, ids, tile_ids = meta
        expert, tile = ids[part, index], tile_ids[part, index]
        previous = jnp.maximum(index - 1, 0)
        following = jnp.minimum(index + 1, counts[part] - 1)

        @pl.when(slot < counts[part])
        def compute():
            @pl.when((slot == 0) | (tile != tile_ids[part, previous]))
            def initialize_dx():
                assembled_dx[...] = jnp.zeros((tm, width), x.dtype)

            @pl.when((slot == 0) | (expert != ids[part, previous]))
            def initialize_weights():
                accumulated_up[...] = jnp.zeros((width, hidden), jnp.float32)
                accumulated_down[...] = jnp.zeros((hidden, width), jnp.float32)

            @pl.when(offsets[part, expert + 1] > offsets[part, expert])
            def products():
                saved, dy = pre_ref[...], dy_ref[...]
                d_hidden = _dot(dy, down_ref[...].T).astype(x.dtype)
                derivative = (jnp.asarray(2, x.dtype) * saved).astype(x.dtype)
                d_pre = jnp.where(saved > 0, (d_hidden * derivative).astype(x.dtype), 0)
                dx = _dot(d_pre, up_ref[...].T).astype(x.dtype)

                def row_mask(columns):
                    rows = lax.broadcasted_iota(jnp.int32, (tm, columns), 0) + tile * tm
                    return (rows >= offsets[part, expert]) & (rows < offsets[part, expert + 1])

                mask_width, mask_hidden = row_mask(width), row_mask(hidden)
                assembled_dx[...] = jnp.where(mask_width, dx, assembled_dx[...])
                masked_x = jnp.where(mask_width, x_ref[...], 0)
                masked_d_pre = jnp.where(mask_hidden, d_pre, 0)
                post = jnp.square(jnp.maximum(saved, 0)).astype(x.dtype)
                masked_post = jnp.where(mask_hidden, post, 0)
                masked_dy = jnp.where(mask_width, dy, 0)
                accumulated_up[...] += _dot(masked_x.T, masked_d_pre)
                accumulated_down[...] += _dot(masked_post.T, masked_dy)

            @pl.when((slot == counts[part] - 1) | (expert != ids[part, following]))
            def finish_weights():
                up_out[...] = accumulated_up[...].astype(up.dtype)
                down_out[...] = accumulated_down[...].astype(down.dtype)

            @pl.when((slot == counts[part] - 1) | (tile != tile_ids[part, following]))
            def finish_dx():
                dx_out[...] = assembled_dx[...]

    def row_index(part, slot, meta, counts, expert_ids):
        return part * (part_rows // tm) + meta[2][part, safe_slot(part, slot, counts)], 0

    def weight_index(part, slot, meta, counts, expert_ids):
        return expert_ids[meta[1][part, safe_slot(part, slot, counts)]], 0, 0

    def partial_weight_index(part, slot, meta, counts, expert_ids):
        return meta[1][part, safe_slot(part, slot, counts)], 0, 0

    inputs = (x, pre, dy, up, down)
    outputs = (
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct((expanded_experts, width, hidden), up.dtype),
        jax.ShapeDtypeStruct((expanded_experts, hidden, width), down.dtype),
    )
    tile_specs = tuple(
        jax.ShapeDtypeStruct(shape, x.dtype)
        for shape in ((tm, width), (tm, hidden), (tm, width), (width, hidden), (hidden, width))
    )
    body_cost = pl.estimate_cost(_tile_backward_math, *tile_specs)
    # Include the maximum number of group-boundary tiles; empty groups skip GEMMs.
    max_tiles = m // tm + expanded_experts - 1
    cost = with_io_bytes_accessed(
        pl.CostEstimate(
            flops=body_cost.flops * max_tiles,
            transcendentals=body_cost.transcendentals * max_tiles,
            bytes_accessed=0,
        ),
        kernel_inputs_specs=(metadata, counts, expert_ids, inputs),
        kernel_outputs_specs=outputs,
    )
    dx, partial_up, partial_down = pl.pallas_call(
        kernel,
        out_shape=outputs,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(partitions, jnp.max(counts)),
            in_specs=(
                pl.BlockSpec((tm, width), row_index),
                pl.BlockSpec((tm, hidden), row_index),
                pl.BlockSpec((tm, width), row_index),
                pl.BlockSpec((None, width, hidden), weight_index),
                pl.BlockSpec((None, hidden, width), weight_index),
            ),
            out_specs=(
                pl.BlockSpec((tm, width), row_index),
                pl.BlockSpec((None, width, hidden), partial_weight_index),
                pl.BlockSpec((None, hidden, width), partial_weight_index),
            ),
            scratch_shapes=(
                pltpu.VMEM((tm, width), x.dtype),
                pltpu.VMEM((width, hidden), jnp.float32),
                pltpu.VMEM((hidden, width), jnp.float32),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        cost_estimate=cost,
        name="belay_fused_all_expert_gradients",
        interpret=interpret,
    )(metadata, counts, expert_ids, *inputs)

    def merge_boundaries(partials):
        merged = partials[original_slots]
        for boundary in range(partitions - 1):
            expert = cut_experts[boundary]
            value = merged[expert].astype(jnp.float32) + partials[cut_slots[boundary]].astype(jnp.float32)
            merged = merged.at[expert].set(value.astype(partials.dtype))
        return merged

    return dx, merge_boundaries(partial_up), merge_boundaries(partial_down)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def fused_expert_relu2(
    x: Float[Array, "rows width"],
    up: Float[Array, "experts width hidden"],
    down: Float[Array, "experts hidden width"],
    groups: jax.Array,
    interpret: bool = False,
    partitions: int = 2,
    tile_rows: int = 512,
) -> jax.Array:
    """Compute a local ReLU² expert MLP with fused forward and all gradients."""
    return fused_row_products(x, up, down, groups, interpret=interpret, partitions=partitions, tile_rows=tile_rows)[1]


def _forward(x, up, down, groups, interpret, partitions, tile_rows):
    pre, output = fused_row_products(
        x, up, down, groups, interpret=interpret, partitions=partitions, tile_rows=tile_rows
    )
    return output, (x, pre, up, down, groups)


def _backward(interpret, partitions, tile_rows, residuals, gradient):
    x, pre, up, down, groups = residuals
    dx, d_up, d_down = fused_expert_gradients(
        x, pre, gradient, up, down, groups, interpret=interpret, partitions=partitions, tile_rows=tile_rows
    )
    return dx, d_up, d_down, None


fused_expert_relu2.defvjp(_forward, _backward)


def fused_expert_mlp(rows, w_up, w_down, groups, active_rows, *, interpret=False, tile_rows=512):
    """Adapt the fused local kernel after the caller masks rejected routes.

    ``local_expert_mlp`` supplies zero input rows for the rejected suffix and
    masks that suffix again before combine, including its incoming cotangent.
    The physical groups include those rows so every kernel output is initialized.
    """
    del active_rows
    return fused_expert_relu2(rows, w_up, w_down, groups, interpret=interpret, tile_rows=tile_rows)
