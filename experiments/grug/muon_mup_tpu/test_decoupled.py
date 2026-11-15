# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.grug.muon_mup_tpu.decoupled import muon_md_step, muown_step


def _torch_orthogonalize(g, coefficients):
    # Unbatched PyTorch reference; production uses bucketed, sharded JAX kernels.
    x = g.to(torch.bfloat16)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.float().square().sum().to(torch.bfloat16).sqrt() + torch.tensor(1e-7, dtype=torch.bfloat16))
    for a, b, c in coefficients:
        a, b, c = (torch.tensor(v, dtype=torch.bfloat16) for v in (a, b, c))
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.T if transpose else x).float()


@pytest.mark.parametrize("shape", [(7, 4), (4, 7), (2, 7, 4), (2, 3, 4, 7)])
@pytest.mark.parametrize("method", ["muown", "md"])
def test_decoupled_multistep_matches_autograd_reparameterization(shape, method):
    rng = np.random.default_rng(35)
    original = rng.normal(0, 0.2, shape).astype(np.float32)
    params = jnp.asarray(original)
    tx = (
        muown_step(learning_rate=0.003, weight_decay=0.1)
        if method == "muown"
        else muon_md_step(learning_rate=0.01, gain_learning_rate=0.001)
    )
    state = tx.init(params)
    references = []
    coefficients = (
        [(3.4445, -4.7750, 2.0315)] * 5
        if method == "muown"
        else [
            (4.0848, -6.8946, 2.9270),
            (3.9505, -6.3029, 2.6377),
            (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046),
            (2.8366, -3.0525, 1.2012),
        ]
    )
    for w in original.reshape((-1, *shape[-2:])):
        w = torch.tensor(w.T.copy())
        v = w.clone().requires_grad_()
        if method == "muown":
            gains = [w.norm(dim=1, keepdim=True).requires_grad_()]
        else:
            gains = [
                torch.full((w.shape[0], 1), math.log(math.expm1(1))).requires_grad_(),
                torch.full((1, w.shape[1]), math.log(math.expm1(1))).requires_grad_(),
            ]
        adam = torch.optim.Adam(
            gains, lr=0.003 if method == "muown" else 0.001, betas=(0.9, 0.95 if method == "muown" else 0.99), eps=1e-8
        )
        references.append((v, gains, adam, torch.zeros_like(w), w.norm()))
    for _ in range(5):
        grads = rng.normal(size=shape).astype(np.float32)
        updates, state = tx.update(jnp.asarray(grads), state, params)
        params = optax.apply_updates(params, updates)
        expected = []
        for (v, gains, adam, momentum, radius), g in zip(references, grads.reshape((-1, *shape[-2:])), strict=True):
            adam.zero_grad()
            v.grad = None
            if method == "muown":
                fused = gains[0] * v / v.norm(dim=1, keepdim=True)
            else:
                fused = torch.nn.functional.softplus(gains[0]) * v * torch.nn.functional.softplus(gains[1])
            old = fused.detach().clone()
            # Autograd supplies the chain rule, independent of production's hand-derived gradients.
            (fused * torch.tensor(g.T.copy())).sum().backward()
            with torch.no_grad():
                momentum.mul_(0.95).add_(v.grad)
                orthogonal = _torch_orthogonalize(v.grad + 0.95 * momentum, coefficients)
                scale = 0.2 * math.sqrt(max(v.shape)) if method == "muown" else radius / math.sqrt(min(v.shape))
                v.add_(orthogonal * (-0.003 if method == "muown" else -0.01) * scale)
                if method == "md":
                    v.mul_(radius / v.norm())
            adam.step()
            with torch.no_grad():
                if method == "muown":
                    r = v.norm(dim=1, keepdim=True)
                    w = gains[0] * v / r - 0.003 * 0.1 * old
                    gains[0].copy_(w.norm(dim=1, keepdim=True))
                    v.copy_(w / gains[0] * r)
                else:
                    w = torch.nn.functional.softplus(gains[0]) * v * torch.nn.functional.softplus(gains[1])
                expected.append(w.T.numpy().copy())
        # bf16 NS rounds differently between the two backends. This bound is
        # below 10% of a single nominal update; axis/gain omissions exceed it.
        np.testing.assert_allclose(params, np.stack(expected).reshape(shape), rtol=2e-3, atol=2e-4)
    if method == "md":
        bare = params / (jax.nn.softplus(state.row_raw) * jax.nn.softplus(state.col_raw))
        np.testing.assert_allclose(
            np.linalg.norm(bare, axis=(-2, -1)), np.linalg.norm(original, axis=(-2, -1)), rtol=2e-6, atol=1e-7
        )


@pytest.mark.parametrize("axis", ["data", "expert"])
def test_decoupled_updates_and_gain_state_match_replicated_on_four_devices(axis):
    run_on_cpu_devices(
        """
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, AxisType, PartitionSpec as P
from experiments.grug.muon_mup_tpu.decoupled import muown_step, muon_md_step

axis = AXIS
shape = (2, 4, 8, 16)
p = jnp.asarray(np.random.default_rng(7).normal(0, .2, size=shape), dtype=jnp.float32)
g = jnp.asarray(np.random.default_rng(8).normal(size=shape), dtype=jnp.float32)
transforms = [muown_step(learning_rate=.003, weight_decay=.1),
              muon_md_step(learning_rate=.01, gain_learning_rate=.001)]
references = [tx.update(g, tx.init(p), p) for tx in transforms]
mesh_shape = (1, 4, 1, 1) if axis == 'data' else (1, 1, 4, 1)
mesh = Mesh(np.array(jax.devices()).reshape(mesh_shape),
    ('replica_dcn', 'data', 'expert', 'model'), axis_types=(AxisType.Explicit,) * 4)
with jax.set_mesh(mesh):
    spec = P(None, 'expert', None, 'data')
    p = jax.sharding.reshard(p, spec)
    g = jax.sharding.reshard(g, spec)
    for tx, reference in zip(transforms, references, strict=True):
        actual = jax.jit(tx.update)(g, tx.init(p), p)
        assert actual[0].sharding.spec == p.sharding.spec
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True):
            np.testing.assert_allclose(a, b, rtol=3e-6, atol=3e-7)
""".replace(
            "AXIS", repr(axis)
        ),
        device_count=4,
    )
