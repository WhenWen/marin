# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Port Belay's aligned Inkling slices while retaining the original attention math.

The slice indexing comes from blocked_inkling_attention in Marin commit
1c64d46c541fe5c61492f580ac5a080915771f49. Only bias expansion changes here;
QK contractions, softmax, document masks and the backward remain native KMA.
"""

import jax
import jax.numpy as jnp
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band


def sliced_dense_rel_bias(band, q_len, k_len):
    """Expand aligned query blocks with slices, avoiding elementwise gathers."""
    if q_len % REL_BIAS_BLOCK or q_len > k_len or band.shape[2] != q_len:
        raise ValueError("KMA bias slicing requires aligned query blocks and Q <= K")
    rel_extent = rel_extent_of_band(band)

    def block(index):
        start = index * REL_BIAS_BLOCK
        tile = jax.lax.dynamic_slice_in_dim(band, start, REL_BIAS_BLOCK, axis=2)
        padded = jnp.pad(tile, ((0, 0), (0, 0), (0, 0), (k_len, k_len)))
        return jax.lax.dynamic_slice_in_dim(padded, k_len + rel_extent - start, k_len, axis=-1)

    blocks = jax.lax.map(block, jnp.arange(q_len // REL_BIAS_BLOCK))
    return blocks.transpose(1, 2, 0, 3, 4).reshape(*band.shape[:2], q_len, k_len)
