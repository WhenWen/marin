from experiments.dclm.exp433_dclm_run import dclm_baseline_only_model
from experiments.evals.evals import default_eval, evaluate_lm_evaluation_harness
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorMainConfig, executor_main
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
"""
For evals that need to be run on GPUs (e.g. LM Evaluation Harness).
"""

executor_main_config = ExecutorMainConfig()

# example of how to eval a specific checkpoint
quickstart_eval_step = evaluate_lm_evaluation_harness(
    model_name="pf5pe4ut/step-600",
    model_path="gs://marin-us-central2/checkpoints/quickstart_single_script_docker_test_09_18/"
    "pf5pe4ut/hf/pf5pe4ut/step-600",
    evals=EvalTaskConfig("mmlu", num_fewshot=0),
)

# example of how to use default_eval to run CORE_TASKS on a step
exp433_dclm_1b_1x_eval_nov12 = default_eval(dclm_baseline_only_model)

# eval_check1 = default_eval(
#     step="gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/checkpoints/step-740000/",
#     evals=list(CORE_TASKS_PLUS_MMLU),
#     resource_config=SINGLE_TPU_V4_8,
# )

# eval_check2 = default_eval(
#     step="gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/checkpoints/step-730000/",
#     # marin-us-central2/checkpoints/llama-8b-tootsie-phase2/hf/step-730000
#     #step="gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/hf/step-730000",
#     evals=list(CORE_TASKS_PLUS_MMLU),
#     resource_config=SINGLE_TPU_V4_8,
# )

# run the evals for llama-8b-tootsie-phase2
eval_check1 = evaluate_lm_evaluation_harness(
    model_name="checkpoints/step-730000/",
    model_path="gs://marin-us-central2/checkpoints/llama-8b-tootsie-phase2/checkpoints/step-730000/",
    evals=list(CORE_TASKS_PLUS_MMLU),
)


steps = [
    quickstart_eval_step,
    exp433_dclm_1b_1x_eval_nov12,
]

if __name__ == "__main__":
    executor_main(executor_main_config, steps=steps)
