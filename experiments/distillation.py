"""An example to demonstrate how to generate synthetic data given a seed dataset.

In this example, we use the MATH-500 dataset from HuggingFace and generate synthetic data using
a Llama-3.1-8B-Instruct model. To try a different model or dataset,
you can change the `model_name` or `huggingface_dataset_id` variables, respectively.
"""

from experiments.models import get_model_local_path, llama_3_1_8b_instruct
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.generation.inference import TextGenerationInferenceConfig, run_inference
from marin.utils import get_directory_friendly_name
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf

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
    name="documents/testinfer-math500-suhas/llama_8b_debugv2",
    fn=run_inference,
    config=TextGenerationInferenceConfig(
        input_path=output_path_of(math500),
        output_path=this_output_path(),
        model_name=get_model_local_path(llama_3_1_8b_instruct),
        engine_kwargs={
            "max_model_len": 8192,
            "enforce_eager": True,
            "tensor_parallel_size": tensor_parallel_size,
        },
        generation_kwargs={
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        template="For the following document, give me a paraphrase of the same using very terse and abstruse language that only an erudite scholar will understand. Replace simple words and phrases with rare and complex ones. Directly output the paraphrased document without any other text at the start or end.\n\n{example}",
        tensor_parallel_size=tensor_parallel_size,
        prompt_column="problem",
        filetype="jsonl",
        tpu_type="TPU-v4-16",
    ),
)

steps = [math500, generations]

if __name__ == "__main__":
    executor_main(steps)
