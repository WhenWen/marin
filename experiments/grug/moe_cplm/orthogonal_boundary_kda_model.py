# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input-dependent orthogonal mixing of native KDA states at chunk boundaries."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track import model
from experiments.grug.moe import boundary_stack_kda
from experiments.grug.moe_cplm.boundary_stack_kda_model import BoundaryStackKimiDeltaAttention


def cayley_matrix(coordinates):
    """Map six upper-triangle entries to an orthogonal 4-by-4 matrix."""
    a, b, c, d, e, f = (coordinates[..., i] for i in range(6))
    zero = jnp.zeros_like(a)
    rows = ((zero, a, b, c), (-a, zero, d, e), (-b, -d, zero, f), (-c, -e, -f, zero))
    skew = jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)
    identity = jnp.eye(4, dtype=coordinates.dtype)
    return jnp.linalg.solve(identity - skew, identity + skew)


class OrthogonalBoundaryKimiDeltaAttention(BoundaryStackKimiDeltaAttention):
    w_boundary_matrix: jax.Array | None = None
    boundary_matrix_bias: jax.Array | None = None

    @staticmethod
    def init(cfg, *, key, precision="mixed"):
        if cfg.num_heads != 4:
            raise ValueError("Orthogonal boundary mixing uses exactly four native KDA heads")
        base = BoundaryStackKimiDeltaAttention.init(cfg, key=key, precision=precision)
        fields = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
        fields.update(w_stack_actions=None, stack_action_bias=None)
        return OrthogonalBoundaryKimiDeltaAttention(
            **fields,
            w_boundary_matrix=reshard(jnp.zeros((cfg.hidden_dim, 6), jnp.float32), P(None, None)),
            boundary_matrix_bias=reshard(jnp.zeros(6, jnp.float32), P(None)),
        )

    def boundary_actions(self, x):
        boundary = x[:, boundary_stack_kda.CHUNK - 1 :: boundary_stack_kda.CHUNK, :]
        # Keep the small controller and Cayley solve in FP32, including its AD.
        coordinates = (
            jnp.einsum(
                "bcd,dk->bck",
                boundary.astype(jnp.float32),
                self.w_boundary_matrix,
                precision=jax.lax.Precision.HIGHEST,
            )
            + self.boundary_matrix_bias
        )
        matrix = cayley_matrix(coordinates)
        identity = jnp.eye(4, dtype=jnp.float32)
        gram = jnp.einsum("...ji,...jk->...ik", matrix, matrix, precision=jax.lax.Precision.HIGHEST)
        stats = {
            "orthogonal_error": jnp.max(jnp.abs(gram - identity)),
            "orthogonal_identity_distance": jnp.mean(jnp.sum((matrix - identity) ** 2, axis=(-2, -1))),
            "orthogonal_input_variance": jnp.mean(jnp.var(matrix, axis=(0, 1))),
            "orthogonal_coordinate_rms": jnp.sqrt(jnp.mean(coordinates**2)),
        }
        stats = {model._KDA_STAT_PREFIX + name: jax.lax.stop_gradient(value) for name, value in stats.items()}
        return matrix.reshape(*matrix.shape[:-2], 16), stats
