# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the complete base16 d512 forward/backward on TPU, without a dataset.

Random-token numerical validation is not a Belay training result. The model,
global batch and sequence length match the frozen run; no optimizer updates run.
"""

import json
import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.fast_track.belay_tpu import SOURCE_CONFIG, base16_model_config
from experiments.grug.fast_track.model import Transformer


def main() -> None:
    DistributedConfig().initialize()
    devices = jax.devices()
    if any(d.platform != "tpu" or "v4" not in d.device_kind for d in devices):
        raise RuntimeError("Full-model validation requires TPU v4")
    config = base16_model_config()
    source = json.loads(SOURCE_CONFIG.read_text())
    batch = source["trainer"]["trainer"]["train_batch_size"]
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    mesh = compact_grug_mesh(expert_axis_size=len(devices), replica_axis_size=1)
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "random-token forward/backward; not a training result",
        "batch_size": batch,
        "sequence_length": config.max_seq_len,
        "chips": len(devices),
        "seed": 0,
    }
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(lambda key: Transformer.init(config, key=key))(jax.random.key(0))
        spec = P(("replica_dcn", "data", "expert"), None)
        tokens = jax.sharding.reshard(
            jax.random.randint(jax.random.key(1), (batch, config.max_seq_len), 0, config.vocab_size), spec
        )
        weights = jax.sharding.reshard(jnp.ones(tokens.shape, jnp.float32), spec)
        segments = jax.sharding.reshard(jnp.broadcast_to(jnp.arange(config.max_seq_len) // 1024, tokens.shape), spec)
        mask = AttentionMask.causal().with_segment_ids(segments)

        def loss(model):
            return policy.cast_to_compute(model).next_token_loss(tokens, weights, mask=mask)

        start = time.monotonic()
        loss_value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(model)
        loss_value.block_until_ready()
        finite = jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads) if eqx.is_array(x)]).all()
        report.update(
            loss=float(loss_value), finite_gradients=bool(finite), compile_and_execute_seconds=time.monotonic() - start
        )
        (output / f"full-model-rank-{jax.process_index()}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if not report["finite_gradients"] or not bool(jnp.isfinite(loss_value)):
            raise RuntimeError("Nonfinite base16 loss or gradients")


if __name__ == "__main__":
    main()
