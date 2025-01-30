"""
Test continued training from checkpoint to support different mixtures.
Issue: https://github.com/stanford-crfm/marin/issues/702
"""

import os
from itertools import chain
from typing import List, Optional
import random

from levanter.optim import AdamConfig

from experiments.defaults import _prepare_data_config
from experiments.llama import llama_150m, llama_300m, llama_600m

from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config

from experiments.curriculum.curriculum_stages import train_executor_step, tokenize_train_validation, tokenize_train_validation_sft
from experiments.instruction_datasets import get_instruction_dataset

marin_prefix = os.environ["MARIN_PREFIX"]

print("Launching experiment from:", marin_prefix)

if 'us-central2' in marin_prefix:
    STACK_PYTHON = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/python/data-{id:05d}-of-00144.parquet"
    STACK_CPP = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/cpp/data-{id:05d}-of-00110.parquet"
    DOLMA_C4 = marin_prefix + "/raw/dolma/v1.7/c4-{id:04d}.json.gz" # different across regions
    DOLMA_TULU_FLAN = marin_prefix + "/raw/dolma/v1.7/tulu_flan-{id:04d}.json.gz" # different across regions
    SPJ6B = marin_prefix + "/raw/SlimPajama-6B-be35b7/b5f90f4/huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/b5f90f4/data/train-{id:05d}-of-00048*.parquet"
    tpu_type = "v4-128"
elif 'eu-west4' in marin_prefix:
    STACK_PYTHON = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/python/data-{id:05d}-of-00144.parquet"
    STACK_CPP = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/cpp/data-{id:05d}-of-00110.parquet"
    DOLMA_C4 = marin_prefix + "/raw/dolma-c4-split/c4-{id:04d}.json.gz" # different across regions
    DOLMA_TULU_FLAN = marin_prefix + "/raw/dolma-tulu_flan-split/tulu_flan-{id:04d}.json.gz" # different across regions
    SPJ6B = marin_prefix + "/raw/SlimPajama-6B-be35b7/b5f90f4/huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/b5f90f4/data/train-{id:05d}-of-00048*.parquet"
    tpu_type = "v6e-32"
else:
    raise ValueError("Unknown prefix")

# randomly split stack python parquet files into two seperate groups
stack_file_ids = list(range(144))
random.seed(42)
random.shuffle(stack_file_ids)
stack_file_ids_stage1 = stack_file_ids[0:72]
stack_file_ids_stage2 = stack_file_ids[72:143]
stack_file_ids_validation = stack_file_ids[143:144]

# randomly split dolma c4 json.gz files into two seperate groups
dolma_file_ids = list(range(171))
random.shuffle(dolma_file_ids)
dolma_file_ids_stage1 = dolma_file_ids[0:85]
dolma_file_ids_stage2 = dolma_file_ids[85:170]
dolma_file_ids_validation = dolma_file_ids[170:171]

# randomly split stack cpp parquet files into two seperate groups
stack_cpp_file_ids = list(range(110))
random.shuffle(stack_cpp_file_ids)
stack_cpp_file_ids_stage1 = stack_cpp_file_ids[0:55]
stack_cpp_file_ids_stage2 = stack_cpp_file_ids[55:109]
stack_cpp_file_ids_validation = stack_cpp_file_ids[109:110]

tulu_file_ids = list(range(6))
random.shuffle(tulu_file_ids)
tulu_file_ids_stage1 = tulu_file_ids[0:1]
tulu_file_ids_stage2 = tulu_file_ids[1:5]
tulu_file_ids_validation = tulu_file_ids[5:6]

spj6b_file_ids = list(range(48))
random.shuffle(spj6b_file_ids)
spj6b_file_ids_stage1 = spj6b_file_ids[0:23]
spj6b_file_ids_stage2 = spj6b_file_ids[23:47]
spj6b_file_ids_validation = spj6b_file_ids[47:48]

flan_file_ids = list(range(66))
random.shuffle(flan_file_ids)
flan_file_ids_stage1 = flan_file_ids[0:32]
flan_file_ids_stage2 = flan_file_ids[32:65]
flan_file_ids_validation = flan_file_ids[65:66]

stack_dedup_stage1_tokenized = tokenize_train_validation(
    train_files=[STACK_PYTHON.format(id=id) for id in stack_file_ids_stage1],
    validation_files=[STACK_PYTHON.format(id=id) for id in stack_file_ids_validation],
    name="stack_dedup_stage1",
    text_key="content"
)

dolma_c4_stage1_tokenized = tokenize_train_validation(
    train_files=[DOLMA_C4.format(id=id) for id in dolma_file_ids_stage1],
    validation_files=[DOLMA_C4.format(id=id) for id in dolma_file_ids_validation],
    name="dolma_c4_stage1",
    text_key="text"
)

stack_cpp_stage1_tokenized = tokenize_train_validation(
    train_files=[STACK_CPP.format(id=id) for id in stack_cpp_file_ids_stage1],
    validation_files=[STACK_CPP.format(id=id) for id in stack_cpp_file_ids_validation],
    name="stack_cpp_stage1",
    text_key="content"
)

spj6b_stage1_tokenized = tokenize_train_validation(
    train_files=[SPJ6B.format(id=id) for id in spj6b_file_ids_stage1],
    validation_files=[SPJ6B.format(id=id) for id in spj6b_file_ids_validation],
    name="spj6b_stage1",
    text_key="text"
)

flan_stage1_tokenized = tokenize_train_validation(
    train_files=[DOLMA_TULU_FLAN.format(id=id) for id in flan_file_ids_stage1],
    validation_files=[DOLMA_TULU_FLAN.format(id=id) for id in flan_file_ids_validation],
    name="flan_stage1",
    text_key="text"
)

stack_dedup_stage2_tokenized = tokenize_train_validation(
    train_files=[STACK_PYTHON.format(id=id) for id in stack_file_ids_stage2],
    validation_files=[STACK_PYTHON.format(id=id) for id in stack_file_ids_validation],
    name="stack_dedup_stage2",
    text_key="content"
)

dolma_c4_stage2_tokenized = tokenize_train_validation(
    train_files=[DOLMA_C4.format(id=id) for id in dolma_file_ids_stage2],
    validation_files=[DOLMA_C4.format(id=id) for id in dolma_file_ids_validation],
    name="dolma_c4_stage2",
    text_key="text"
)

stack_cpp_stage2_tokenized = tokenize_train_validation(
    train_files=[STACK_CPP.format(id=id) for id in stack_cpp_file_ids_stage2],
    validation_files=[STACK_CPP.format(id=id) for id in stack_cpp_file_ids_validation],
    name="stack_cpp_stage2",
    text_key="content"
)

spj6b_stage2_tokenized = tokenize_train_validation(
    train_files=[SPJ6B.format(id=id) for id in spj6b_file_ids_stage2],
    validation_files=[SPJ6B.format(id=id) for id in spj6b_file_ids_validation],
    name="spj6b_stage2",
    text_key="text"
)

flan_stage2_tokenized = tokenize_train_validation(
    train_files=[DOLMA_TULU_FLAN.format(id=id) for id in flan_file_ids_stage2],
    validation_files=[DOLMA_TULU_FLAN.format(id=id) for id in flan_file_ids_validation],
    name="flan_stage2",
    text_key="text"
)

stage_data = {
    "stack_dedup": {
        "stage1": stack_dedup_stage1_tokenized,
        "stage2": stack_dedup_stage2_tokenized,
    },
    "c4": {
        "stage1": dolma_c4_stage1_tokenized,
        "stage2": dolma_c4_stage2_tokenized,
    },
    "stack_cpp": {
        "stage1": stack_cpp_stage1_tokenized,
        "stage2": stack_cpp_stage2_tokenized,
    },
    "spj6b": {
        "stage1": spj6b_stage1_tokenized,
        "stage2": spj6b_stage2_tokenized,
    },
    "flan": {
        "stage1": flan_stage1_tokenized,
        "stage2": flan_stage2_tokenized,
    }
}

def full_training_stage_allstage2(
    data1_name : str,
    data2_name : str,
    total_data1_portion : float,
    duration_fracs_stage2 : List[float],
    learning_rate : float = 3e-3,
    cooldown_frac : Optional[float] = None,
    schedule_type : str = "cosine",
    model_size : str = "150m",
    num_train_steps : int = 3000,
    version_tag : str = "",
    additional_tags : List[str] = [],
):
    """
    Generalized version of varsched that works with any two datasets.
    
    Args:
        data1_name: Name of first dataset (e.g. "stack_dedup", "stack_cpp")
        data2_name: Name of second dataset (e.g. "c4")
        total_data1_portion: Total portion of data1 across both stages
        duration_fracs_stage2: Fraction of total steps to spend in stage 2
    """

    duration_fracs_stage2 = sorted(duration_fracs_stage2, reverse=True)

    def steps_stage1(duration_frac_stage2):
        return int(num_train_steps * (1 - duration_frac_stage2))

    # Construct executor steps for training
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
        "600m": llama_600m,
    }[model_size]
    train_batch_size=1024
    num_train_steps=num_train_steps 

    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    name_prefix = f"{data1_name}-{data2_name}-allstage2"
    
    if model_size == "300m" or model_size == "600m":
        name_prefix += f"-{model_size}"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )
        name_prefix += f"-{schedule_type}-{cooldown_frac}"
    elif schedule_type == "cosine":
        optimizer_config = None
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")

    data_config_stage1 = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage1"], data2_name: stage_data[data2_name]["stage1"]},
        weights={data1_name: 0.0, data2_name: 1.0},
    )

    pretraining_data_stage1, evaluation_data_stage1 = _prepare_data_config(data_config_stage1, use_default_validation=True, use_default_evaluation=True)

    train_step_stage1 = train_executor_step(
        name=f"{name_prefix}-stage1{version_tag}",
        pretraining_data=pretraining_data_stage1,
        evaluation_data=evaluation_data_stage1,
        model=model,
        model_checkpoint=None,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export_list=[steps_stage1(duration_frac_stage2) for duration_frac_stage2 in duration_fracs_stage2],
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags + ["stage1"],
    )

    train_steps_stage2 = []

    for duration_frac_stage2 in duration_fracs_stage2:
        data1_weight_stage2 = round(total_data1_portion / duration_frac_stage2, 7)

        assert 0 <= data1_weight_stage2 <= 1, f"data1_weight_stage2: {data1_weight_stage2}"

        data_config_stage2 = lm_mixture_data_config(
            components={data1_name: stage_data[data1_name]["stage2"], data2_name: stage_data[data2_name]["stage2"]},
            weights={data1_name: data1_weight_stage2, data2_name: 1 - data1_weight_stage2},
        )

        pretraining_data_stage2, evaluation_data_stage2 = _prepare_data_config(data_config_stage2, use_default_validation=True, use_default_evaluation=True)

        train_step_stage2 = train_executor_step(
            name=f"{name_prefix}-{data1_weight_stage2}-stage2{version_tag}",
            pretraining_data=pretraining_data_stage2,
            evaluation_data=evaluation_data_stage2,
            model=model,
            model_checkpoint=output_path_of(train_step_stage1).cd(f"checkpoints/step-{steps_stage1(duration_frac_stage2)}"),
            train_batch_size=train_batch_size,
            num_train_steps=num_train_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            steps_per_eval=steps_per_eval,
            steps_per_export_list=[num_train_steps],
            tpu_type=tpu_type,
            optimizer_config=optimizer_config,
            additional_tags=additional_tags + ["stage2"],
        )

        train_steps_stage2.append(train_step_stage2)

    return [train_step_stage1, *train_steps_stage2]

def full_training_stage_varsched(data1_name, data2_name, total_data1_portion, duration_frac_stage2, data1_frac_alloc_stage2, learning_rate=3e-3, cooldown_frac=None, schedule_type="cosine", model_size="150m", num_train_steps=3000, version_tag="", additional_tags=[]):
    """
    Generalized version of varsched that works with any two datasets.
    
    Args:
        data1_name: Name of first dataset (e.g. "stack_dedup", "stack_cpp")
        data2_name: Name of second dataset (e.g. "c4")
        total_data1_portion: Total portion of data1 across both stages
        duration_frac_stage2: Fraction of total steps to spend in stage 2
        data1_frac_alloc_stage2: Fraction of data1's total portion to allocate to stage 2
    """
    duration_frac_stage1 = 1 - duration_frac_stage2
    data1_frac_alloc_stage1 = 1 - data1_frac_alloc_stage2

    data1_weight_stage1 = round(total_data1_portion * data1_frac_alloc_stage1 / duration_frac_stage1, 7)
    data1_weight_stage2 = round(total_data1_portion * data1_frac_alloc_stage2 / duration_frac_stage2, 7)

    print('-' * 100)
    print(f"total_data1_portion: {total_data1_portion}")
    print(f"duration_frac_stage1: {duration_frac_stage1}, data1_frac_alloc_stage1: {data1_frac_alloc_stage1}, data1_weight_stage1: {data1_weight_stage1}")
    print(f"duration_frac_stage2: {duration_frac_stage2}, data1_frac_alloc_stage2: {data1_frac_alloc_stage2}, data1_weight_stage2: {data1_weight_stage2}")

    assert 0 <= data1_weight_stage1 <= 1, f"data1_weight_stage1: {data1_weight_stage1}"
    assert 0 <= data1_weight_stage2 <= 1, f"data1_weight_stage2: {data1_weight_stage2}"

    data_config_stage1 = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage1"], data2_name: stage_data[data2_name]["stage1"]},
        weights={data1_name: data1_weight_stage1, data2_name: 1 - data1_weight_stage1},
    )

    pretraining_data_stage1, evaluation_data_stage1 = _prepare_data_config(data_config_stage1, use_default_validation=True, use_default_evaluation=True)

    data_config_stage2 = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage2"], data2_name: stage_data[data2_name]["stage2"]},
        weights={data1_name: data1_weight_stage2, data2_name: 1 - data1_weight_stage2},
    )

    pretraining_data_stage2, evaluation_data_stage2 = _prepare_data_config(data_config_stage2, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
        "600m": llama_600m,
    }[model_size]
    train_batch_size=1024
    num_train_steps=num_train_steps 

    steps_stage1 = int(num_train_steps * duration_frac_stage1)

    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    name_prefix = f"{data1_name}-{data2_name}-vs-{data1_frac_alloc_stage2}"
    if model_size == "300m" or model_size == "600m":
        name_prefix += f"-{model_size}"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )
        name_prefix += f"-{schedule_type}-{cooldown_frac}"
    elif schedule_type == "cosine":
        optimizer_config = None
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")

    train_step_stage1 = train_executor_step(
        name=f"{name_prefix}-{data1_weight_stage1}-{data1_weight_stage2}-stage1{version_tag}",
        pretraining_data=pretraining_data_stage1,
        evaluation_data=evaluation_data_stage1,
        model=model,
        model_checkpoint=None,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export_list=[steps_stage1],
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags + ["stage1"],
    )

    train_step_stage2 = train_executor_step(
        name=f"{name_prefix}-{data1_weight_stage1}-{data1_weight_stage2}-stage2{version_tag}",
        pretraining_data=pretraining_data_stage2,
        evaluation_data=evaluation_data_stage2,
        model=model,
        model_checkpoint=output_path_of(train_step_stage1).cd(f"checkpoints/step-{steps_stage1}"),
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export_list=[num_train_steps],
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags + ["stage2"],
    )

    return [train_step_stage1, train_step_stage2]

def full_training_stage_baseline_sweep(data1_name, data2_name, learning_rate, schedule_type, cooldown_frac=None, num_train_steps=3000, model_size="150m", additional_tags=[], data1_portion=0.005, train_batch_size=1024, version_tag=""):
    data_config = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage1"], data2_name: stage_data[data2_name]["stage2"]},
        weights={data1_name: data1_portion, data2_name: 1 - data1_portion},
    )

    pretraining_data, evaluation_data = _prepare_data_config(data_config, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
    }[model_size]
    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    steps_per_export=num_train_steps // 2
    name_prefix = f"{data1_name}-{data2_name}-{num_train_steps // 1000}B-{model_size}-baseline{version_tag}"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )

        schedule_type = f"linear-{cooldown_frac}"
    elif schedule_type == "cosine":
        optimizer_config = None
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")
    
    train_step = train_executor_step(
        name=name_prefix,
        pretraining_data=pretraining_data,
        evaluation_data=evaluation_data,
        model=model,
        model_checkpoint=None,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export_list=[steps_per_export],
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags,
    )

    return [train_step]

############################################################

if __name__ == "__main__":
    # stage_pairs = [
    #     full_training_stage_allstage2(
    #         data1_name="stack_dedup",
    #         data2_name="stack_cpp",
    #         total_data1_portion=0.005,
    #         duration_fracs_stage2=[0.4, 0.2],
    #         schedule_type=schedule_type,
    #         cooldown_frac=cooldown_frac,
    #         num_train_steps=500,
    #         additional_tags=["debug"]
    #     )
    #     for schedule_type, cooldown_frac in [("cosine", None), ("linear", 0.0)]
    # ]

    stage_pairs = [
        full_training_stage_varsched(
            data1_name="flan",
            data2_name="c4",
            total_data1_portion=0.5,
            duration_frac_stage2=0.4,
            data1_frac_alloc_stage2=0.25,
            schedule_type="linear",
            cooldown_frac=0.05,
            model_size="150m",
            num_train_steps=300,
            additional_tags=["debug-eu-flan-c4"],
        )
    ]

    # stage_pairs = [
    #     full_training_stage_varsched(
    #         data1_name="flan",
    #         data2_name="c4",
    #         total_data1_portion=0.005,
    #         duration_frac_stage2=duration_frac_stage2,
    #         data1_frac_alloc_stage2=data1_frac_alloc_stage2,
    #         schedule_type=schedule_type,
    #         cooldown_frac=cooldown_frac,
    #         model_size="150m",
    #         num_train_steps=3000,
    #         additional_tags=["flan-c4-0.005-varsched-cooldown-0.05-sweep"],
    #     )
    #     for duration_frac_stage2 in [0.4, 0.2, 0.1, 0.05, 0.025, 0.00625]
    #     for schedule_type, cooldown_frac in [("linear", 0.05)]
    #     for data1_frac_alloc_stage2 in [0.25, 0.5, 0.75, 1.0]
    # ]

    # stage_pairs = [
    #     full_training_stage_allstage2(
    #         data1_name="flan",
    #         data2_name="c4",
    #         total_data1_portion=0.005,
    #         duration_fracs_stage2=[0.4, 0.2, 0.1, 0.05, 0.025, 0.00625],
    #         schedule_type=schedule_type,
    #         cooldown_frac=cooldown_frac,
    #         num_train_steps=3000,
    #         additional_tags=["flan-c4-0.005-allstage2-sweep"],
    #         version_tag="-v1"
    #     )
    #     for schedule_type, cooldown_frac in [("linear", 0.05)]
    # ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description=f"Test training with varying mixtures",
    )
