"""
CS 224N script for tokenization of data, and training of reference models, and 1B model experiments.
"""


from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from experiments.llama import llama3_tokenizer
from marin.processing.tokenize import (
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    lm_data_config,
    tokenize,
)
from experiments.simple_train_config import SimpleTrainConfig


from experiments.evals.task_configs import (
    CORE_TASKS_PLUS_MMLU,
)

from experiments.defaults import default_train
from experiments.llama import LlamaConfig

from marin.scaling_laws.create_ladder_suite import scaling_law_suite, WS_EMA_DEFAULT_TRAIN_CONFIG
import dataclasses

### Tokenize shard 2 of DCLM-Baseline-1.0

BASE_PATH = "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_01_of_10/local-shard_2_of_10"
FILE_PATTERN = "**/*.jsonl.zst"

tokenized_dclm_shard_2 = ExecutorStep(
    name="tokenized/dclm-shard-2-of-10",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[f"{BASE_PATH}/{FILE_PATTERN}"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
    pip_dependency_groups=["tokenize_train"],
)

### Train two reference models on shard 2 of DCLM-Baseline-1.0

reference_model_sizes_hidden_dim = [512, 1024] # corresponds to 200 and ~500M params

# replace the steps for eval and steps per task eval in the train config, and replace the #train steps to correspond to 50B tokens
training_config = dataclasses.replace(WS_EMA_DEFAULT_TRAIN_CONFIG, steps_per_eval=2500, steps_per_task_eval=2500, num_train_steps=12000) # corresponds to 50B tokens

# replace the train steps in the train config and use the default scaling law suite to define the runs
reference_models = scaling_law_suite(
    sweep_name="quality_ref_sl",
    tokenized=tokenized_dclm_shard_2,
    widths=reference_model_sizes_hidden_dim,
    training_config=training_config,
    tags=["scaling_laws"],
)

### Define the 1B experiments- one for random sampling and one for filtered data

NUM_TRAIN_TOKENS = int(25e9)  # 25 billion tokens
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (256 * 2048)  # 256 is the batch size, 2048 is the sequence length

llama_1_4b_dclm = LlamaConfig(
    seq_len=2048,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=24,
    use_flash_attention=True,
)

training_config_1b_model = SimpleTrainConfig(
    tpu_type="v4-128",
    train_batch_size=256,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=3e-3,
    weight_decay=0.033,
    min_lr_ratio=0.1,
    warmup=5000,
    z_loss_weight=1e-4,
    steps_per_eval=1000,
    steps_per_task_eval=1000,
)


EXPERIMENT_TAG_RANDOM_SAMPLING = ["quality_sl_1b_random_sampling", "dclm"]

### Tokenize shard 3 of DCLM-Baseline-1.0

BASE_PATH = "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_01_of_10/local-shard_3_of_10"
FILE_PATTERN = "**/*.jsonl.zst"


tokenized_dclm_shard_3 = ExecutorStep(
    name="tokenized/dclm-shard-3-of-10",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[f"{BASE_PATH}/{FILE_PATTERN}"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
    pip_dependency_groups=["tokenize_train"],
)

### Train the 1B model on shard 3 of DCLM-Baseline-1.0 with random sampling

dclm_baseline_random_sampling_model = default_train(
    name="dclm_baseline_random_sampling",
    tokenized=tokenized_dclm_shard_3,
    model_config=llama_1_4b_dclm,
    train_config=training_config_1b_model,
    eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    tags=EXPERIMENT_TAG_RANDOM_SAMPLING,
)

### Tokenize shard 3 of DCLM-Baseline-1.0 with PPL filter quality factor

# gs://marin-us-central2/tokenized/tokenized/quality_filtering/scaling-filter-quality-factor/dclm-global-shard-01-of-10-local-shard_3_of_10-06d752
tokenized_dclm_shard_3_filtered = ExecutorStep(
    name="tokenized/dclm-shard-3-of-10-filtered",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[f"{BASE_PATH}/{FILE_PATTERN}"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
    pip_dependency_groups=["tokenize_train"],
    override_output_path="gs://marin-us-central2/tokenized/tokenized/quality_filtering/scaling-filter-quality-factor/dclm-global-shard-01-of-10-local-shard_3_of_10-06d752",
)

### Train the 1B model on shard 3 of DCLM-Baseline-1.0 with PPL filter quality factor

EXPERIMENT_TAG_FILTERED_DATA = ["quality_sl_1b_filtered_data", "dclm"]

dclm_baseline_filtered_model = default_train(
    name="dclm_baseline_filtered",
    tokenized=tokenized_dclm_shard_3_filtered,
    model_config=llama_1_4b_dclm,
    train_config=training_config_1b_model,
    eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    tags=EXPERIMENT_TAG_FILTERED_DATA,
)

### Tokenize shard 3 of DCLM-Baseline-1.0 with fineweb-edu quality factor

tokenized_dclm_shard_3_filtered_fineweb_edu = ExecutorStep(
    name="tokenized/dclm-shard-3-of-10-filtered-fineweb-edu",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[f"{BASE_PATH}/{FILE_PATTERN}"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
    pip_dependency_groups=["tokenize_train"],
    override_output_path="gs://marin-us-central2/tokenized/tokenized/quality_filtering/fineweb-edu/dclm-global-shard-01-of-10-local-shard_3_of_10-85c24f"
)

### Train the 1B model on shard 3 of DCLM-Baseline-1.0 with fineweb-edu quality factor
dclm_baseline_filtered_fineweb_edu = default_train(
    name="dclm_baseline_filtered_fineweb_edu",
    tokenized=tokenized_dclm_shard_3_filtered_fineweb_edu,
    model_config=llama_1_4b_dclm,
    train_config=training_config_1b_model,
    eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    tags=EXPERIMENT_TAG_FILTERED_DATA,
)

### Tokenize shard 3 of DCLM-Baseline-1.0 with dclm-fasttext quality factor

tokenized_dclm_shard_3_dclm_fasttext = ExecutorStep(
    name="tokenized/dclm-shard-3-of-10-dclm-fasttext",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[f"{BASE_PATH}/{FILE_PATTERN}"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
    pip_dependency_groups=["tokenize_train"],
    override_output_path="gs://marin-us-central2/tokenized/tokenized/quality_filtering/dclm-fasttext/dclm-global-shard-01-of-10-local-shard_3_of_10-2dc426/",
)

### Train the 1B model on shard 3 of DCLM-Baseline-1.0 with dclm-fasttext quality factor

dclm_baseline_filtered_dclm_fasttext = default_train(
    name="dclm_baseline_filtered_dclm_fasttext",
    tokenized=tokenized_dclm_shard_3_dclm_fasttext,
    model_config=llama_1_4b_dclm,
    train_config=training_config_1b_model,
    eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    tags=EXPERIMENT_TAG_FILTERED_DATA,
)

### Tokenize fineweb data-- note that this did not make it into our final report

FINEWEB_BASE_PATH = "gs://marin-us-central2/documents/scaling-filter-fineweb-subset"
FINEWEB_FILE_PATTERN = "**/*.jsonl.gz"

fineweb_tokenized = ExecutorStep(
    name="tokenized/fineweb_quality_ref_sl",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[f"{FINEWEB_BASE_PATH}/{FINEWEB_FILE_PATTERN}"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
    pip_dependency_groups=["tokenize_train"],
)

### Train the 1B model on fineweb data-- note that this did not make it into our final report

fineweb_dclm_baseline_1b = default_train(
    name="fineweb_dclm_baseline_1b",
    tokenized=fineweb_tokenized,
    model_config=llama_1_4b_dclm,
    train_config=training_config_1b_model,
    eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
)

# run the reference models
executor_main(
    steps=[
        tokenized_dclm_shard_2,
        *reference_models,
        tokenized_dclm_shard_3,
        dclm_baseline_random_sampling_model,
        tokenized_dclm_shard_3_filtered,
        dclm_baseline_filtered_model,
        tokenized_dclm_shard_3_filtered_fineweb_edu,
        dclm_baseline_filtered_fineweb_edu,
        tokenized_dclm_shard_3_dclm_fasttext,
        dclm_baseline_filtered_dclm_fasttext,
        fineweb_tokenized,
        fineweb_dclm_baseline_1b,
    ],
    description="Run the reference models",
)


















