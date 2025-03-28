import dataclasses
from experiments.simple_train_config import SimpleTrainConfig
from experiments.llama import llama_8b
from experiments.dolma.exp442_dolma import dolma_llama3_tokenized
from experiments.evals.task_configs import CORE_TASKS, CORE_TASKS_PLUS_MMLU
from experiments.defaults import default_train
from experiments.exp780_scaling_laws_with_some_flan_data import dclm_flan_mixture_config
from marin.execution.executor import executor_main


llama_8b_train_config = SimpleTrainConfig(
    tpu_type="v4-128",
    node_count=1,
    train_batch_size=1024,
    num_train_steps=1_000_000,  # using wsd-s so this doesn't really matter
    # these hypers from Table 12 in https://arxiv.org/html/2406.11794v1#A6
    learning_rate=1e-3,  # we get divergence with 2e-3
    weight_decay=0.05,
    # WSD-S
    cycle_length=10000,
    steps_per_eval=10000,
    steps_per_task_eval=10000,
    steps_per_export=20000,
    warmup=1000,  # initial warmup
    # TODO: do we need rewarmup
    decay=0.1,  # 10% of 5000 = 500 steps
    lr_schedule="inv",
)

llama_8b_tootsie_dolma = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-dolma-0.001",
        tokenized=dolma_llama3_tokenized,
        model_config=llama_8b,
        train_config=llama_8b_train_config,
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
        tags=["llama", "8b", "wsd-s", "exp780", "scaling_laws"],
    ),
)

# with flan data
llama_8b_tootsie_dclm_default_plus_flan = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-dclm-default-plus-flan-0.001",
        tokenized=dclm_flan_mixture_config,
        model_config=llama_8b,
        train_config=llama_8b_train_config,
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
        tags=["llama", "8b", "wsd-s", "exp780", "scaling_laws"],
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[
            #llama_8b_tootsie_dolma,
            llama_8b_tootsie_dclm_default_plus_flan,
        ],
        description="Train 8B models on Dolma and DCLM+FLAN using WSD-S.",
    )

