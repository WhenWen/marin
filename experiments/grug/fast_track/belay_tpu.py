# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen Belay base16 d512 on TPU v4, using the native Fast Track trainer."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import draccus
import jax
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import OptimizerConfig
from levanter.tokenizers import load_tokenizer
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.fast_track.model import GrugModelConfig
from experiments.grug.fast_track.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, _run_grug_local

SOURCE_CONFIG = Path(__file__).with_name("belay_base15_d512.json")
SOURCE_COMMIT = "358283f7da684d156d8bca095669cf8c023ef2f3"
SOURCE_RUN = "https://wandb.ai/marin-community/marin_moe/runs/lc1-b205-ctrl"
BASE16_CHANGE = "https://github.com/marin-community/marin/issues/9451#issuecomment-6105926801"
REGION_PREFIX = "gs://marin-us-central2/"
TPU_XLA_ROUNDING_FLAG = "--xla_allow_excess_precision=false"


def require_tpu_rounding() -> None:
    """Require declared bf16 rounding for Muon's sharded Newton-Schulz math."""
    if TPU_XLA_ROUNDING_FLAG not in os.environ.get("XLA_FLAGS", "").split():
        raise RuntimeError(f"Launch TPU training with XLA_FLAGS={TPU_XLA_ROUNDING_FLAG}")


def base16_model_config() -> GrugModelConfig:
    """Recorded base15 plus the base16 gate removals and TPU rematerialization."""
    model = json.loads(SOURCE_CONFIG.read_text())["model"]
    model.update(kda_write_gate=False, kda_erase_gate=False, kda_onchip_erase_gate=False)
    # Recompute intermediates to reduce HBM use; preserve mathematical settings.
    model.update(
        attn_res_layer_backward="RECOMPUTE",
        kda_save_chunk_states=False,
        attn_res_remat_attention=True,
        kda_state_grads="XLA",
        kda_xla_scan_impl="parallel_stacked_split",
    )
    return draccus.decode(GrugModelConfig, model)


def build_config(
    *,
    run_id: str,
    cache_map: dict[str, str],
    output: Path,
    checkpoint_path: str,
    chips: int,
    microbatches: int,
    seed: int,
    tokenizer_path: str | None = None,
) -> GrugRunConfig:
    """Decode the recorded recipe and apply documented TPU execution changes.

    The cache map must cover every original component by name. Cache equivalence
    is the caller's responsibility: paths alone cannot prove token identity.
    """
    source = json.loads(SOURCE_CONFIG.read_text())
    data = source["data"]
    if tokenizer_path is not None:
        data["tokenizer"] = tokenizer_path
    if set(cache_map) != set(data["components"]):
        raise ValueError("cache-map must name exactly the frozen training and validation components")
    if not checkpoint_path.startswith(REGION_PREFIX):
        raise ValueError("TPU v4 checkpoints must be in us-central2")
    for name, component in data["components"].items():
        path = cache_map[name]
        if not path.startswith(REGION_PREFIX):
            raise ValueError(f"{name}: cache must be mirrored in us-central2 before launch")
        component.update(type="cached", cache_dir=path)
        component["format"]["type"] = "text"
        if component["source"] is not None:
            component["source"]["type"] = "url"
            component["source"]["cache_dir"] = path
            component["source"]["format"]["type"] = "text"
    source["optimizer"]["type"] = "grug_fast_track_muonh_v1"
    outer = source["trainer"]
    # Route the entire logical batch through one expert group. Subdividing into
    # independent data-parallel groups would change the capacity/drop boundary.
    expert_shards = chips
    batch = outer["trainer"]["train_batch_size"]
    if microbatches < 1 or batch % (chips * microbatches):
        raise ValueError("global batch must divide into equal device microbatches")
    outer.update(
        expert_axis_size=expert_shards,
        replica_axis_size=1,
        grad_accum_microbatches=microbatches,
        loader_buffer_batches=8,
        completion_marker_path=f"{checkpoint_path}/final_eval_done.json",
        save_checkpoints=True,
    )
    trainer = outer["trainer"]
    trainer.update(
        id=run_id,
        seed=seed,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        load_checkpoint=False,
        load_checkpoint_path=None,
        log_dir=str(output / "logs"),
        tracker=[{"type": "json_logger"}],
    )
    trainer["checkpointer"].update(base_path=checkpoint_path, temporary_base_path=f"{checkpoint_path}/temporary")
    trainer["jax_config"]["jax_enable_pgle"] = False
    # Keep the logical eval coverage and pure EMA semantics, with fewer examples
    # live on each chip. Training steps and optimizer schedules remain 2817.
    evaluation = source["eval"]
    evaluation["eval_batch_size"] = chips
    return GrugRunConfig(
        model=base16_model_config(),
        optimizer=draccus.decode(OptimizerConfig, source["optimizer"]),
        data=draccus.decode(LmDataConfig, data),
        trainer=draccus.decode(GrugTrainerConfig, outer),
        eval=draccus.decode(GrugEvalConfig, evaluation),
        resources=ResourceConfig.with_tpu(f"v4-{chips * 2}"),
        tensorstore_cache_bytes=source["tensorstore_cache_bytes"],
        stop_after_steps=source["stop_after_steps"],
        processes_per_task=1,
        max_retries_failure=0,
        max_task_failures=0,
    )


def stage_data_baseline(url: str, directory: Path) -> tuple[dict[str, str], str, dict]:
    """Verify a new-data manifest and stage its small tokenizer files per host."""
    if not url.startswith(REGION_PREFIX):
        raise ValueError("Data baseline must be in us-central2")
    raw = StoragePath(url).read_bytes()
    baseline = json.loads(raw)
    if baseline["status"] != "validated" or baseline["tokenizer"]["vocab_size"] != 16384:
        raise ValueError("Data baseline must be validated with a 16384-token vocabulary")
    if baseline["train"]["tokens"] < 1_600_000_000:
        raise ValueError("Data baseline is too small for the frozen run")
    directory.mkdir(parents=True, exist_ok=True)
    for name, identity in baseline["tokenizer"]["files"].items():
        if Path(name).name != name or not identity["url"].startswith(REGION_PREFIX):
            raise ValueError("Invalid tokenizer artifact path")
        data = StoragePath(identity["url"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != identity["sha256"]:
            raise ValueError(f"Tokenizer SHA256 mismatch: {name}")
        (directory / name).write_bytes(data)
    tokenizer = load_tokenizer(str(directory))
    if (tokenizer.vocab_size, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (16384, 0, 1, 7):
        raise ValueError("Unexpected tokenizer vocabulary or document boundary IDs")
    return baseline["cache_map"], str(directory), {
        "manifest": url, "sha256": hashlib.sha256(raw).hexdigest(),
        "kind": baseline["kind"], "tokenizer": baseline["tokenizer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    data_args = parser.add_mutually_exclusive_group(required=True)
    data_args.add_argument("--cache-map", type=Path)
    data_args.add_argument("--data-baseline", help="Validated GCS manifest for a separately labeled new data baseline.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--chips", type=int, choices=(4, 16), default=16)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("IRIS_OUTPUT_DIR", "outputs")))
    parser.add_argument("--run", action="store_true", help="Execute; otherwise write the resolved configuration.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer_path, data_provenance = None, None
    if args.data_baseline:
        cache_map, tokenizer_path, data_provenance = stage_data_baseline(
            args.data_baseline, args.output / "tokenizer"
        )
    else:
        cache_map = json.loads(args.cache_map.read_text())
    config = build_config(
        run_id=args.run_id,
        cache_map=cache_map,
        output=args.output,
        checkpoint_path=args.checkpoint_path,
        chips=args.chips,
        microbatches=args.microbatches,
        seed=args.seed,
        tokenizer_path=tokenizer_path,
    )
    with (args.output / "resolved-config.yaml").open("w") as stream:
        draccus.dump(config, stream)
    provenance = {
        "source_commit": SOURCE_COMMIT,
        "source_run": SOURCE_RUN,
        "source_config_sha256": hashlib.sha256(SOURCE_CONFIG.read_bytes()).hexdigest(),
        "base16_change": BASE16_CHANGE,
        "seed": args.seed,
        "microbatches": args.microbatches,
        "chips": args.chips,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "equivalence": "unverified: EP topology differs; microbatches > 1 also change routing capacity",
        "data_baseline": data_provenance,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if not args.run:
        return
    require_tpu_rounding()
    # Initialize distributed topology before inspecting devices. Do not apply the
    # H100 launcher's CUDA command-buffer or ragged-transport flags on TPU.
    config.trainer.trainer.distributed.initialize()
    devices = jax.devices()
    if len(devices) != args.chips or any(d.platform != "tpu" or "v4" not in d.device_kind for d in devices):
        raise RuntimeError(f"Expected {args.chips} TPU v4 chips; got {devices}")
    rank_output = args.output / f"rank-{jax.process_index()}"
    rank_output.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(rank_output / "metrics.jsonl")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("levanter.json_logger").addHandler(handler)
    _run_grug_local(config)


if __name__ == "__main__":
    main()
