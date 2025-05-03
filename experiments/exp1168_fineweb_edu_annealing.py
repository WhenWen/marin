"""
This experiment evaluates the quality of Wikipedia data for model cooldown using `default_quality_ablation`
which fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% FineMath Crawl
- 15% Dolma/FLAN dataset

Reference Issue: https://github.com/stanford-crfm/marin/issues/845
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main

# Tokenize the Wikipedia dataset
fineweb_edu_tokenized = default_tokenize(
    "fineweb-edu-crawl",
    "gs://marin-us-central2/scratch/nfliu/text/fineweb_edu_unique_100M_passing_minhash_against_fineweb_edu/",
    tokenizer=llama3_tokenizer,
)

# Conduct the cooldown experiment over v4-128 TPU else the v5litepod-128
# TPU is used which is not available in us-central2
cooldown_config = QualityAblationConfig()

# Conduct the cooldown experiment
fineweb_edu_cooldown_ablation = default_quality_ablation(
    fineweb_edu_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            fineweb_edu_tokenized,
            fineweb_edu_cooldown_ablation,
        ]
    )
