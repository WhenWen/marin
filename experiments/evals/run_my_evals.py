from experiments.evals.evals import default_key_evals, evaluate_lm_evaluation_harness, extract_model_name_and_path
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from marin.execution.executor import executor_main
from marin.evaluation.evaluation_config import EvalTaskConfig

# Insert your model path here
# model_path = "gs://marin-us-central2/checkpoints/llama-8b-control-00f31b/hf/step-210388"

GENERATION_TASKS = (
    # EvalTaskConfig(name="ifeval", num_fewshot=0),
    # EvalTaskConfig(name="gsm8k_cot", num_fewshot=8),
    # EvalTaskConfig(name="drop", num_fewshot=0),
    # EvalTaskConfig(name="humaneval", num_fewshot=10),
    # EvalTaskConfig(name="bbh_cot_fewshot", num_fewshot=3, task_alias="bbh"),
    EvalTaskConfig(name="hendrycks_math", num_fewshot=5, task_alias="hendrycks_math_5shot"),
)

VLLM_ENGINE_KWARGS = {
    "max_model_len": 4096,
}

# model_gcs_path = "simplescaling/s1-32B"
# model_gcs_path = "meta-llama/Llama-3.1-8B"
model_gcs_path = "meta-llama/Llama-3.1-8B-Instruct"
# model_gcs_path = "gs://marin-us-central2/checkpoints/suhas/open-web-math--r5-dclm-llama3.1-8b-5B-ra1.0"

name, model_step_path = extract_model_name_and_path(model_gcs_path)

gen_eval_step = evaluate_lm_evaluation_harness(
    model_name=name,
    model_path=model_step_path,
    evals=GENERATION_TASKS,
    engine_kwargs=VLLM_ENGINE_KWARGS,
    resource_config=SINGLE_TPU_V6E_8,
)

if __name__ == "__main__":
    executor_main(steps=[gen_eval_step])
