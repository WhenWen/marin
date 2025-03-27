"""Synthetic data generation for the GSM8K dataset in the style of MIND.

Inspiration from the Olmo-2 paper where they utilize the MIND rewrite technique to generate
synthetic math datasets from existing datasets.
"""

from transformers import AutoTokenizer

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.medu.medu_mmlu import mmlu_science_pipeline
from experiments.midtraining_datasets import finemath, finemath_3_plus_tokenized
from experiments.models import get_model_local_path, llama_3_1_8b_instruct, llama_3_3_70b_instruct
from marin.execution.executor import ExecutorStep, InputName, executor_main, output_path_of, this_output_path, versioned
from marin.generation.inference import TextGenerationInferenceConfig, run_inference
from marin.utils import get_directory_friendly_name
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf

huggingface_dataset_id = "openai/gsm8k"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tensor_parallel_size = 1

dataset_name = get_directory_friendly_name(huggingface_dataset_id)
gsm8k = ExecutorStep(
    name=f"raw/{dataset_name}",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=huggingface_dataset_id,
        revision=versioned("e53f048"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
).cd("main/train-00000-of-00001.parquet")

finemath_4_plus_single_parquet = finemath.cd("finemath-4plus/train-00000-of-00064.parquet")

tokenizer = AutoTokenizer.from_pretrained(model_name)

STUDENT_TEACHER_TEMPLATE = """
"{example}\n\nConvert the context above as a multi-turn discussions between a teacher and a student.
The student has questions about the context and the teacher solves each of them step-by-step.
Make sure that their discussions strictly adhere to the context above and remains faithful to
information in the context. Please DO NOT add any new information/reference other than the context."""

PROBLEM_SOLVING_TEMPLATE = """
"{example}\n\nConvert the context above as a multi-turn problem-solving conversation where participants
analyze challenges or scenarios presented in the content and brainstorm solutions within the context
of the provided material, avoiding speculation or unrelated discussions. Make sure that their conversation
strictly adhere to the context above and remains faithful to information in the context.
Please DO NOT add any new information/reference other than the context."""


def qa_rewrite_document(
    input_path: ExecutorStep | InputName,
    model_name_or_path: ExecutorStep,
    document_name: str,
    template: str,
    filetype: str,
    prompt_column: str,
    tensor_parallel_size: int,
):

    if isinstance(input_path, ExecutorStep):
        input_path = output_path_of(input_path)

    return ExecutorStep(
        name=f"documents/{document_name}",
        fn=run_inference,
        config=TextGenerationInferenceConfig(
            input_path=input_path,
            output_path=this_output_path(),
            model_name=get_model_local_path(model_name_or_path),
            engine_kwargs={
                "max_model_len": 8192,
                "enforce_eager": False,
                "tensor_parallel_size": tensor_parallel_size,
            },
            generation_kwargs={
                "temperature": 0.8,
                "max_tokens": 1024,
                "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            },
            template=template,
            tensor_parallel_size=tensor_parallel_size,
            prompt_column=prompt_column,
            filetype=filetype,
            output_filetype_override="jsonl.gz",
            one_to_one_input_output_mapping=False,
            generated_text_column_name="text",
            batch_size=64,
        ),
    )


gsm8k_student_teacher_llama8b = qa_rewrite_document(
    input_path=gsm8k,
    model_name_or_path=llama_3_1_8b_instruct,
    document_name="gsm8k-llama8b-mind/student_teacher",
    template=STUDENT_TEACHER_TEMPLATE,
    filetype="parquet",
    prompt_column="question",
    tensor_parallel_size=tensor_parallel_size,
)

gsm8k_problem_solving_llama8b = qa_rewrite_document(
    input_path=gsm8k,
    model_name_or_path=llama_3_1_8b_instruct,
    document_name="gsm8k-llama8b-mind/problem_solving",
    template=PROBLEM_SOLVING_TEMPLATE,
    filetype="parquet",
    prompt_column="question",
    tensor_parallel_size=tensor_parallel_size,
)

# 87M * 4 / 0.30
gsm8k_rewrite_model = default_quality_ablation(
    candidate_tokenized=default_tokenize(
        name="gsm8k-llama8b-mind-qa",
        dataset=output_path_of(gsm8k_student_teacher_llama8b),
        tokenizer=llama3_tokenizer,
    ),
    config=QualityAblationConfig(
        tpu_type="v6e-128",
        mcq_weight=0.0,
        candidate_weight=0.30,
        num_anneal_tokens=int(7_000_000 * 4 / 0.30),
        train_batch_size=256,  # I want more steps because less tokens
        model_name_prefix="8b-quality-eval-bsz-256",
    ),
)


tensor_parallel_size_70b = 8
# Create the rewrite steps using the function
mind_rewrite_student_teacher_70b = qa_rewrite_document(
    input_path=gsm8k,
    model_name_or_path=llama_3_3_70b_instruct,
    document_name="gsm8k-llama70b-mind/student_teacher",
    template=STUDENT_TEACHER_TEMPLATE,
    filetype="parquet",
    prompt_column="question",
    tensor_parallel_size=tensor_parallel_size_70b,
)

mind_rewrite_problem_solving_70b = qa_rewrite_document(
    input_path=gsm8k,
    model_name_or_path=llama_3_3_70b_instruct,
    document_name="gsm8k-llama70b-mind/problem_solving",
    template=PROBLEM_SOLVING_TEMPLATE,
    filetype="parquet",
    prompt_column="question",
    tensor_parallel_size=tensor_parallel_size_70b,
)

finemath_4_plus_student_teacher_llama8b = qa_rewrite_document(
    input_path=finemath_4_plus_single_parquet,
    model_name_or_path=llama_3_1_8b_instruct,
    document_name="finemath-4plus-llama8b-mind/student_teacher",
    template=STUDENT_TEACHER_TEMPLATE,
    filetype="parquet",
    prompt_column="text",
    tensor_parallel_size=1,
)

REPHRASE_THE_WEB_QA_TEMPLATE = """
{example}\n\nConvert the context above into a conversational format between a user and an assistant
with multiple tags of "Question:" followed by "Answer:"
"""
mmlu_science_shard_one = mmlu_science_pipeline.filtered_documents.cd("local-shard_0_of_10")

mmlu_science_qa = qa_rewrite_document(
    input_path=mmlu_science_shard_one,
    model_name_or_path=llama_3_1_8b_instruct,
    document_name="medu-mmlu-science-llama8b-mind-qa",
    template=REPHRASE_THE_WEB_QA_TEMPLATE,
    filetype="jsonl.zst",
    prompt_column="text",
    tensor_parallel_size=1,
)

mmlu_science_rewrite_model = default_quality_ablation(
    candidate_tokenized=default_tokenize(
        name="medu-mmlu-science-llama8b-mind-qa",
        dataset=output_path_of(mmlu_science_qa),
        tokenizer=llama3_tokenizer,
    ),
    config=QualityAblationConfig(
        tpu_type="v6e-128",
        mcq_weight=0.0,
        candidate_weight=0.30,
        num_anneal_tokens=int(1_783_169_234 * 4 / 0.30),
        train_batch_size=1024,  # I want more steps because less tokens
        model_name_prefix="8b-quality-qa-ep-4-30",
    ),
)

mmlu_science_entire_shard = mmlu_science_pipeline.filtered_documents
mmlu_science_qa_whole_shard = qa_rewrite_document(
    input_path=mmlu_science_entire_shard,
    model_name_or_path=llama_3_1_8b_instruct,
    document_name="medu-mmlu-science-llama8b-qa-whole",
    template=REPHRASE_THE_WEB_QA_TEMPLATE,
    filetype="jsonl.zst",
    prompt_column="text",
    tensor_parallel_size=1,
)

mmlu_science_qa_whole_shard_tokenized = default_tokenize(
    name="medu-candidate-mmlu-science-llama-8b-qa",
    dataset=output_path_of(mmlu_science_qa_whole_shard),
    tokenizer=llama3_tokenizer,
)

mmlu_science_og_tokenized = default_tokenize(
    name="medu-candidate-mmlu-science",
    dataset=output_path_of(mmlu_science_entire_shard),
    tokenizer=llama3_tokenizer,
)

mmlu_science_qa_model = default_quality_ablation(
    candidate_tokenized=mmlu_science_qa_whole_shard_tokenized,
    config=QualityAblationConfig(
        mcq_component=mmlu_science_qa_whole_shard_tokenized,
        tpu_type="v6e-128",
        mcq_weight=0.30,
        candidate_weight=0.0,
        num_anneal_tokens=50_000_000_000,
        model_name_prefix="8b-dclm-70-qa-30-50b",
    ),
)

mmlu_science_qa_model_og_15_15 = default_quality_ablation(
    candidate_tokenized=mmlu_science_og_tokenized,
    config=QualityAblationConfig(
        mcq_component=mmlu_science_qa_whole_shard_tokenized,
        tpu_type="v6e-128",
        mcq_weight=0.15,
        candidate_weight=0.15,
        num_anneal_tokens=50_000_000_000,
        model_name_prefix="8b-dclm-70-og-15-qa-15-50b",
    ),
)

finemath3_plus_40_anneal = default_quality_ablation(
    candidate_tokenized=finemath_3_plus_tokenized,
    config=QualityAblationConfig(
        tpu_type="v6e-128",
        mcq_weight=0.0,
        candidate_weight=0.40,
        num_anneal_tokens=50_000_000_000,
        model_name_prefix="8b-quality-eval-noflan-40",
    ),
)

finemath3_plus_30_anneal = default_quality_ablation(
    candidate_tokenized=finemath_3_plus_tokenized,
    config=QualityAblationConfig(
        tpu_type="v6e-128",
        mcq_weight=0.0,
        candidate_weight=0.30,
        num_anneal_tokens=50_000_000_000,
        model_name_prefix="8b-quality-eval-noflan-30",
    ),
)

steps = [
    mmlu_science_qa_model_og_15_15,
    mmlu_science_qa_model,
    # mmlu_science_rewrite_msodel,
    # mmlu_science_qa_whole_shard,
    # finemath3_plus_30_anneal,
    # finemath3_plus_40_anneal,
    # gsm8k_rewrite_model,
    # finemath_4_plus_student_teacher_llama8b,
    # mmlu_science_qa,
    # finemath_4_plus,
    # mind_rewrite_student_teacher_llama8b,
    # gsm8k_problem_solving_llama8b,
    # mind_rewrite_student_teacher_70b,
    # mind_rewrite_problem_solving_70b,
]

if __name__ == "__main__":
    executor_main(steps)
