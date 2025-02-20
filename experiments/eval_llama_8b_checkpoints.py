from experiments.evals.evals import default_eval
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from marin.execution.executor import ExecutorMainConfig, executor_main
from experiments.evals.resource_configs import SINGLE_TPU_V4_8, SINGLE_TPU_V6E_8, SINGLE_TPU_V4_256

executor_main_config = ExecutorMainConfig()

eval_check1 = default_eval(
    step="gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/checkpoints/step-740000/",
    evals=list(CORE_TASKS_PLUS_MMLU),
    resource_config=SINGLE_TPU_V4_8,
)

eval_check2 = default_eval(
    step="gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/checkpoints/step-730000/",
    # marin-us-central2/checkpoints/llama-8b-tootsie-phase2/hf/step-730000
    #step="gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/hf/step-730000",
    evals=list(CORE_TASKS_PLUS_MMLU),
    resource_config=SINGLE_TPU_V4_8,
)

steps = [
    eval_check1,
    eval_check2,
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
