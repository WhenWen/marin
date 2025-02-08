# https://github.com/stanford-crfm/marin/issues/621
# Sweep to determine optimal training configs for small models
import dataclasses
import itertools
import logging
import math
from collections.abc import Sequence

import numpy as np
import ray
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_train
from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.llama import llama_150m, llama_1_4b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main, versioned

# TODO: might be nice to do use wandb sweeps, but not today.

logger = logging.getLogger("ray")

# Sweep to determine optimal training config
LR_CHOICES = [1e-4, 1e-3, 3e-3, 1e-2]  # 3e-3 is best
BEST_LR = 3e-3
MUP_LR_CHOICES = [1e-4, 1e-3, 3e-3, 1e-2]
MUP_BEST_LR = 1e-3
SCALING_FACTOR = llama_1_4b.hidden_dim / llama_150m.hidden_dim
WD = 0.1
TPU_TYPES_150m = "v4-128"
TPU_TYPES_1_4b = "v4-128"
TOKEN_TARGETS = 4_000_000_000
BATCH_SIZE = 1024
SEQ_LEN = 4096


def all_combos(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=False))


def step_target(token_target, batch_size, seq_len):
    actual_step_count = math.ceil(token_target / (batch_size * seq_len))
    nice_round_step_count = math.ceil(actual_step_count / 1000) * 1000
    return nice_round_step_count


train_configs_150m = []

for combo in all_combos(lr=LR_CHOICES):
    num_train_steps = step_target(TOKEN_TARGETS, BATCH_SIZE, SEQ_LEN)

    train_configs_150m.append(
        SimpleTrainConfig(
            tpu_type=versioned(TPU_TYPES_150m),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_train_steps,
            learning_rate=combo["lr"],
            weight_decay=WD,
        )
    )

train_configs_150m_mup = []
for combo in all_combos(lr=MUP_LR_CHOICES):
    num_train_steps = step_target(TOKEN_TARGETS, BATCH_SIZE, SEQ_LEN)

    train_configs_150m_mup.append(
        SimpleTrainConfig(
            tpu_type=versioned(TPU_TYPES_150m),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_train_steps,
            learning_rate=combo["lr"],
            weight_decay=WD,
        )
    )


def format_train_config(prefix: str, config: SimpleTrainConfig):
    return (
        f"{prefix}-lr={config.learning_rate}"
    )


def make_sweep_steps(
    prefix: str,
    model_config: LlamaConfig,
    train_configs: list[SimpleTrainConfig],
    tokenized_data: ExecutorStep,
    tags: Sequence[str] = (),
):
    steps = []
    for train_config in train_configs:
        model_config = dataclasses.replace(
            model_config,
            seq_len=SEQ_LEN,
        )

        name = format_train_config(prefix, train_config)

        step = default_train(
            name=name,
            train_config=train_config,
            model_config=model_config,
            tokenized=tokenized_data,
            tags=tags,
        )

        # step = dataclasses.replace(step, fn=_failure_ok_train)

        steps.append(step)
    return steps


sp_steps_150m = []
sp_steps_150m = make_sweep_steps(
    prefix="sweep621-150m",
    model_config=llama_150m,
    train_configs=train_configs_150m,
    tokenized_data=fineweb_edu_tokenized,
    tags=("llama", "150m", "621_mup", "fineweb_edu"),
)

mup_llama_150m = dataclasses.replace(llama_150m, use_mup=True)
mup_steps_150m = make_sweep_steps(
    prefix="sweep621-150m-mup5",
    model_config=mup_llama_150m,
    train_configs=train_configs_150m_mup,
    tokenized_data=fineweb_edu_tokenized,
    tags=("llama", "150m", "621_mup", "fineweb_edu"),
)

num_train_steps = step_target(TOKEN_TARGETS * 10, BATCH_SIZE, SEQ_LEN)

heuristic_train_config = SimpleTrainConfig(
    tpu_type=versioned(TPU_TYPES_1_4b),
    train_batch_size=BATCH_SIZE,
    num_train_steps=num_train_steps,
    learning_rate=BEST_LR / SCALING_FACTOR,
    weight_decay=WD,
)
heuristic_step = default_train(
    name=format_train_config("sweep621-1.4b-heuristic", heuristic_train_config),
    train_config=heuristic_train_config,
    model_config=llama_1_4b,
    tokenized=fineweb_edu_tokenized,
    tags=("llama", "1.4b", "621_mup", "fineweb_edu"),
)

sp_train_config = SimpleTrainConfig(
    tpu_type=versioned(TPU_TYPES_1_4b),
    train_batch_size=BATCH_SIZE,
    num_train_steps=num_train_steps,
    learning_rate=BEST_LR,
    weight_decay=WD,
)
sp_step = default_train(
    name=format_train_config("sweep621-1.4b-sp", sp_train_config),
    train_config=sp_train_config,
    model_config=llama_1_4b,
    tokenized=fineweb_edu_tokenized,
    tags=("llama", "1.4b", "621_mup", "fineweb_edu"),
)

mup_llama_1_4b = dataclasses.replace(llama_1_4b, use_mup=True)
mup_train_config = SimpleTrainConfig(
    tpu_type=versioned(TPU_TYPES_1_4b),
    train_batch_size=BATCH_SIZE,
    num_train_steps=num_train_steps,
    learning_rate=MUP_BEST_LR,
    weight_decay=WD,
)
mup_step = default_train(
    name=format_train_config("sweep621-1.4b-mup", mup_train_config),
    train_config=mup_train_config,
    model_config=mup_llama_1_4b,
    tokenized=fineweb_edu_tokenized,
    tags=("llama", "1.4b", "621_mup", "fineweb_edu"),
)


if __name__ == "__main__":
    executor_main(sp_steps_150m + mup_steps_150m + [heuristic_step, sp_step, mup_step])