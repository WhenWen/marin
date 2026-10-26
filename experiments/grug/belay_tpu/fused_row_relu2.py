# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep both ReLU-squared expert products in each local TPU row tile."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata

from experiments.grug.belay_tpu.expert_gradient_tpu import expert_weight_gradient


@partial(jax.jit, static_argnames=("backward", "partitions", "tile_rows", "interpret"))
def fused_row_products(x, up, down, groups, pre=None, *, backward=False, partitions=2, tile_rows=512, interpret=False):
    """Return saved preactivation/output, or preactivation/input gradients.

    All arrays are device-local. ``groups`` partitions all rows in expert order,
    including any zero-filled rejected suffix. The caller masks rejected rows.
    No routing, capacity decisions or collectives are performed here.
    """
    if not interpret and jax.default_backend() != "tpu":
        raise ValueError("Fused expert products require TPU or explicit interpreter mode")
    if partitions < 1:
        raise ValueError("Expert products need at least one row partition")
    m, width = x.shape
    experts, up_width, hidden = up.shape
    tm = min(tile_rows, m // partitions)
    if (
        tm < 1
        or experts % partitions
        or m % (tm * partitions)
        or up_width != width
        or down.shape != (experts, hidden, width)
        or groups.shape != (experts,)
        or up.dtype != x.dtype
        or down.dtype != x.dtype
        or x.dtype not in (jnp.float32, jnp.bfloat16)
        or (backward and (pre is None or pre.shape != (m, hidden) or pre.dtype != x.dtype))
    ):
        raise ValueError("Expected complete row partitions and matching local expert shapes/dtypes")
    if not interpret and (tm % 128 or width % 128 or hidden % 128):
        raise ValueError("TPU expert tile dimensions must be multiples of 128")
    metadata, active = make_group_metadata(
        group_sizes=groups,
        m=m,
        tm=tm,
        start_group=jnp.asarray(0, jnp.int32),
        num_nonzero_groups=experts,
        visit_empty_groups=False,
    )
    tiles = metadata[2]
    valid = jnp.arange(tiles.size) < active
    boundaries = jnp.arange(partitions + 1) * (m // tm // partitions)
    starts = jnp.sum(valid[None, :] & (tiles[None, :] < boundaries[:, None]), axis=1, dtype=jnp.int32)
    counts = starts[1:] - starts[:-1]

    def slot_index(part, slot, starts, counts):
        return starts[part] + jnp.minimum(slot, counts[part] - 1)

    def kernel(meta, starts, counts, left, up_ref, down_ref, *refs):
        if backward:
            pre_ref, first_out, second_out, assembled_first, assembled_second = refs
        else:
            first_out, second_out, assembled_first, assembled_second = refs
        part, slot = pl.program_id(0), pl.program_id(1)
        index = slot_index(part, slot, starts, counts)
        offsets, ids, tile_ids = meta
        tile, expert = tile_ids[index], ids[index]
        previous = tile_ids[jnp.maximum(index - 1, starts[part])]
        following = tile_ids[jnp.minimum(index + 1, starts[part + 1] - 1)]

        def dot(a, b):
            return lax.dot_general(
                a, b, dimension_numbers=(((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            ).astype(x.dtype)

        @pl.when(slot < counts[part])
        def compute():
            @pl.when((slot == 0) | (tile != previous))
            def initialize():
                assembled_first[...] = jnp.zeros((tm, hidden), x.dtype)
                assembled_second[...] = jnp.zeros((tm, width), x.dtype)

            if backward:
                d_hidden = dot(left[...], down_ref[...].T)
                saved = pre_ref[...]
                derivative = (jnp.asarray(2, x.dtype) * saved).astype(x.dtype)
                first = jnp.where(saved > 0, (d_hidden * derivative).astype(x.dtype), 0)
                second = dot(first, up_ref[...].T)
            else:
                first = dot(left[...], up_ref[...])
                hidden_value = jnp.square(jnp.maximum(first, 0)).astype(x.dtype)
                second = dot(hidden_value, down_ref[...])

            row_first = lax.broadcasted_iota(jnp.int32, (tm, hidden), 0) + tile * tm
            mask_first = (row_first >= offsets[expert]) & (row_first < offsets[expert + 1])
            row_second = lax.broadcasted_iota(jnp.int32, (tm, width), 0) + tile * tm
            mask_second = (row_second >= offsets[expert]) & (row_second < offsets[expert + 1])
            assembled_first[...] = jnp.where(mask_first, first, assembled_first[...])
            assembled_second[...] = jnp.where(mask_second, second, assembled_second[...])

            @pl.when((slot == counts[part] - 1) | (tile != following))
            def finish():
                first_out[...] = assembled_first[...]
                second_out[...] = assembled_second[...]

    def row_index(part, slot, meta, starts, counts):
        return meta[2][slot_index(part, slot, starts, counts)], 0

    def weight_index(part, slot, meta, starts, counts):
        return meta[1][slot_index(part, slot, starts, counts)], 0, 0

    inputs = [x, up, down]
    specs = [
        pl.BlockSpec((tm, width), row_index),
        pl.BlockSpec((None, width, hidden), weight_index),
        pl.BlockSpec((None, hidden, width), weight_index),
    ]
    if backward:
        inputs.append(pre)
        specs.append(pl.BlockSpec((tm, hidden), row_index))
    return pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((m, hidden), x.dtype), jax.ShapeDtypeStruct((m, width), x.dtype)),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(partitions, jnp.max(counts)),
            in_specs=specs,
            out_specs=(pl.BlockSpec((tm, hidden), row_index), pl.BlockSpec((tm, width), row_index)),
            scratch_shapes=[pltpu.VMEM((tm, hidden), x.dtype), pltpu.VMEM((tm, width), x.dtype)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        name="belay_fused_row_relu2_backward" if backward else "belay_fused_row_relu2_forward",
        interpret=interpret,
    )(metadata, starts, counts, *inputs)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def fused_row_relu2(x, up, down, groups, interpret=False, partitions=2, tile_rows=512):
    """Compute local relu(x @ up)**2 @ down, saving preactivation for backward."""
    return fused_row_products(x, up, down, groups, interpret=interpret, partitions=partitions, tile_rows=tile_rows)[1]


def _forward(x, up, down, groups, interpret, partitions, tile_rows):
    pre, output = fused_row_products(
        x, up, down, groups, interpret=interpret, partitions=partitions, tile_rows=tile_rows
    )
    return output, (x, up, down, groups, pre)


def _backward(interpret, partitions, tile_rows, residuals, gradient):
    x, up, down, groups, pre = residuals
    d_pre, d_x = fused_row_products(
        gradient,
        up,
        down,
        groups,
        pre,
        backward=True,
        interpret=interpret,
        partitions=partitions,
        tile_rows=tile_rows,
    )
    d_up = expert_weight_gradient(
        x, d_pre, groups, partitions=partitions, tile_rows=tile_rows, output_dtype=up.dtype, interpret=interpret
    )
    d_down = expert_weight_gradient(
        pre,
        gradient,
        groups,
        partitions=partitions,
        tile_rows=tile_rows,
        output_dtype=down.dtype,
        interpret=interpret,
        lhs_relu2=True,
    )
    return d_x, d_up, d_down, None


fused_row_relu2.defvjp(_forward, _backward)
