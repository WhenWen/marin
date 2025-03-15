from experiments.evals.evals import default_eval
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from marin.execution.executor import ExecutorMainConfig, executor_main

checkpoints_to_eval = [
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-20000",
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-40000",
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-60000",
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-80000",
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-100000",
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-120000",
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-140000",
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-160000",
    "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-180000",
]


def eval_checkpoint(checkpoint_path: str):

    step = default_eval(
        step=checkpoint_path,
        evals=list(CORE_TASKS_PLUS_MMLU),
    )

    return step


if __name__ == "__main__":
    executor_main(
        steps=[
            eval_checkpoint(checkpoint_path) for checkpoint_path in checkpoints_to_eval
        ]
    )
