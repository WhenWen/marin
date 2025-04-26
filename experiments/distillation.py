"""An example to demonstrate how to generate synthetic data given a seed dataset.

In this example, we use the MATH-500 dataset from HuggingFace and generate synthetic data using
a Llama-3.1-8B-Instruct model. To try a different model or dataset,
you can change the `model_name` or `huggingface_dataset_id` variables, respectively.
"""

from dataclasses import dataclass

import ray

from experiments.evals.resource_configs import (
    TPU_V4_16_STRICT_PACK,
)
from experiments.models import get_model_local_path, llama_3_1_8b_instruct
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.generation.inference import TextGenerationInferenceConfig, run_inference
from marin.utils import get_directory_friendly_name
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf

# tpu_type = "TPU-v6e-8"
tpu_type = "TPU-v4-16"


@dataclass
class DummyInferenceConfig:
    prompt: str


@ray.remote(resources={"TPU": 1, f"{tpu_type}-head": 1})
def dummy_inference(cfg: DummyInferenceConfig):
    import os

    os.environ["VLLM_TPU_DEBUG"] = "1"  # Add at top of function

    from marin.generation.llm_generation import vLLMProvider

    # # Configure model and generation parameters
    model_path = get_model_local_path(llama_3_1_8b_instruct)
    engine_kwargs = {
        "max_model_len": 1024,
        "enforce_eager": True,
        "tensor_parallel_size": 1,
    }
    generation_kwargs = {
        "temperature": 0.8,
        "max_tokens": 512,
    }
    # compilation_config = {
    #     "level": 2,
    #     "backend": "pallas",
    #     "compile_sizes": [32],      # just compile for one small size
    #     "cudagraph_capture_sizes": [1],
    #     "max_capture_size": 32,
    #     "splitting_ops": []         # no tensor splitting since we're on single TPU
    # }

    # # Initialize vLLM provider
    llm = vLLMProvider(model_path, engine_kwargs, generation_kwargs)

    # print("Initializing LLM...")

    # llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",#model_path,
    #           max_model_len=1024,
    #           enforce_eager=True)
    # sampling_params = SamplingParams(temperature=0.8, max_tokens=256)
    print("LLM initialized!")

    # Dummy prompt to debug
    prompt = cfg.prompt
    template = (
        "You will be given a problem. Please reason step by step, "
        "and put your final answer within \\boxed{{}}:\n{example}"
    )
    formatted_prompt = template.format(example=prompt)
    print(f"Formatted prompt: {formatted_prompt}")

    # Generate text
    print("About to call generate()...")  # Add this line
    result = llm.generate([formatted_prompt])
    print(f"Generated result: {result}")

    return result


huggingface_dataset_id = "HuggingFaceH4/MATH-500"
tensor_parallel_size = 1

dataset_name = get_directory_friendly_name(huggingface_dataset_id)
math500 = ExecutorStep(
    name=f"raw/{dataset_name}-retry-2",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=huggingface_dataset_id,
        revision=versioned("ff5b202"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

generations = ExecutorStep(
    name="documents/synthetic_data_llama_8b_debug",
    fn=run_inference,
    config=TextGenerationInferenceConfig(
        input_path=output_path_of(math500),
        output_path=this_output_path(),
        model_name=get_model_local_path(llama_3_1_8b_instruct),
        engine_kwargs={
            "max_model_len": 1024,
            "enforce_eager": True,
            "tensor_parallel_size": tensor_parallel_size,
        },
        generation_kwargs={
            "temperature": 0.8,
            "max_tokens": 512,
        },
        template="You will be given a problem. Please reason step by step, \
            and put your final answer within \boxed{{}}:\n{example}",
        tensor_parallel_size=tensor_parallel_size,
        prompt_column="problem",
        filetype="jsonl",
        # resource_config=TPU_V6E_8_STRICT_PACK,
        # resource_config=SINGLE_TPU_V4_8,
        # resource_config=SINGLE_TPU_V4_16,
        resource_config=TPU_V4_16_STRICT_PACK,
    ),
)

generations_alt = ExecutorStep(
    name="documents/synthetic_data_llama_8b_alt_debug",
    fn=dummy_inference,
    config=DummyInferenceConfig(prompt="What is 1+1?"),
)

steps = [math500, generations]

if __name__ == "__main__":
    executor_main(steps)
