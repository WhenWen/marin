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
    # EvalTaskConfig(name="minerva_math", num_fewshot=4, task_alias="minerva_math_4shot"),
    # EvalTaskConfig(name="bec2016eu", num_fewshot=0, task_alias="bec2016eu"),
    # EvalTaskConfig(name="vaxx_stance", num_fewshot=0, task_alias="vaxx_stance"),
    # EvalTaskConfig(name="wnli_eu", num_fewshot=0, task_alias="wnli_eu"),
    # EvalTaskConfig(name="xcopa_eu", num_fewshot=0, task_alias="xcopa_eu"),
    EvalTaskConfig(name="scrolls_quality", num_fewshot=0, task_alias="scrolls_quality"),
    # EvalTaskConfig(name="qnlieu", num_fewshot=0, task_alias="qnlieu"),
    # EvalTaskConfig(name="basque-glue", num_fewshot=0, task_alias="basque-glue"),
)

VLLM_ENGINE_KWARGS = {
    "max_model_len": 4096,
    "max_gen_toks": 1280,
}

# model_gcs_path = "simplescaling/s1-32B"
# model_gcs_path = "meta-llama/Llama-3.1-8B"
# model_gcs_path = "meta-llama/Llama-3.1-8B-Instruct"
# model_gcs_path = "gs://marin-us-central2/checkpoints/suhas/open-web-math--r5-dclm-llama3.1-8b-5B-ra1.0"
# model_step_path = "gs://marin-us-east5/gcsfuse_mount/models/meta-llama--Llama-3-1-8B"
# model_step_path = ""


model_gcs_paths = [
    # "gs://marin-us-central2/checkpoints/suhas/open-web-math--r5-dclm-llama3.1-8b-5B-ra1.0/hf/step-9999",
    # "gs://marin-us-central2/checkpoints/suhas/open-web-math--dclm-llama3.1-8b-10B-ra1.0/hf/step-19999",
    # "gs://marin-us-central2/checkpoints/suhas/open-web-math--r5-dclm-llama3.1-8b-5.263B-ra0.95/hf/step-10525",
    # "gs://marin-us-central2/checkpoints/suhas/open-web-math--dclm-llama3.1-8b-10.526B-ra0.95/hf/step-21051",
    # "gs://marin-us-central2/checkpoints/suhas/open-web-math--spj-llama3.1-8b-10.526B-ra0.95/hf/step-21051",
    # "gs://marin-us-central2/checkpoints/suhas/open-web-math--spj-llama3.1-8b-10B-ra1.0/hf/step-19999",
    # "gs://marin-us-central2/checkpoints/suhas/open-web-math--spj-llama3.1-8b-13.333B-ra0.75/hf/step-26665"
    "gs://marin-us-east5/gcsfuse_mount/models/meta-llama--Llama-3-1-8B"
    # "gs://marin-us-central2/checkpoints/suhas/flan--spj-llama3.1-8b-0.1B-ra1.0/hf/step-199/",
    # "gs://marin-us-central2/checkpoints/suhas/flan--spj-llama3.1-8b-0.133B-ra0.75/hf/step-265/",
    # "gs://marin-us-central2/checkpoints/suhas/flan--spj-llama3.1-8b-0.013B-ra0.75/hf/step-25",
    # "gs://marin-us-central2/checkpoints/suhas/flan--spj-llama3.1-8b-0.01B-ra1.0/hf/step-19",
    # "gs://marin-us-central2/checkpoints/suhas/flan--r4-spj-llama3.1-8b-0.053B-ra0.75/hf/step-105",
    # "gs://marin-us-central2/checkpoints/suhas/flan--r4-spj-llama3.1-8b-0.04B-ra1.0/hf/step-79",
    
]
# name, model_step_path = extract_model_name_and_path(model_gcs_path)

gen_eval_steps = [] 
for model_gcs_path in model_gcs_paths:
    if "step-" in model_gcs_path:
        name, model_step_path = extract_model_name_and_path(model_gcs_path)
    else:
        name = "llama-8b"
        model_step_path = model_gcs_path
    gen_eval_steps.append(evaluate_lm_evaluation_harness(
        model_name=f"{name}-3-20-v1",
        model_path=model_step_path,
        evals=GENERATION_TASKS,
        engine_kwargs=VLLM_ENGINE_KWARGS,
        resource_config=SINGLE_TPU_V6E_8,
    ))

if __name__ == "__main__":
    executor_main(steps=gen_eval_steps)
