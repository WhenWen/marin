"""
This script visualizes the log probabilities of the Tootsie 8b model at various stages of training.
@dlwh was interested in the weird loss behavior of the model after we switched to a longer WSD-S cooldown.
This script visualizes the log probabilities of the model at various stages of training to see if we can
spot any differences.
The differences were structural formatting differences in the eval data:
* Reddit data started with `&gt;` (sic) instead of `>`, which the model didn't like.
* Similarly, the twitter data uniformally ended with a ` ` (space) character, which the model didn't like.
The cooldown seems to function as a kind of sharpening/annealing
"""

from experiments.defaults import default_validation_sets
from experiments.exp600_tootsie import llama3_tokenizer, llama_8b
from experiments.llama import llama_8b_old_rotary
from marin.evaluation.visualize import VizLmConfig, mixture_for_visualization, visualize_lm_log_probs
from marin.execution.executor import ExecutorStep, executor_main, versioned
from marin.scaling_laws.create_ladder_suite import scaling_law_suite, WS_EMA_DEFAULT_TRAIN_CONFIG
from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3

#COMPARISON_MODEL = "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/hf/step-660000"

CHECKPOINTS = [
    "gs://marin-us-central2/checkpoints/scaling-law-suite-default-v2-512-4d173f/hf/step-49999",
    "gs://marin-us-central2/checkpoints/scaling-law-suite-default-v2-768-12373d/hf/step-49999",
    "gs://marin-us-central2/checkpoints/scaling-law-suite-default-v2-1024-77c98b/hf/step-49999",
    "gs://marin-us-central2/checkpoints/scaling-law-suite-default-v2-1536-18344d/hf/step-49999",
    "gs://marin-us-central2/checkpoints/scaling-law-suite-default-v2-2048-7845a1/hf/step-49999",
]


def path_to_step_name(path):
    # we want llama-8b-tootsie-phase2-730000
    components = path.split("/")
    step = components[-2].split("-")[-1]
    name = components[-4].split("/")[-1]
    return f"analysis/viz/{name}-{step}"


eval_sets = default_validation_sets(tokenizer=versioned(llama3_tokenizer))
eval_set_mixture = mixture_for_visualization(eval_sets)

suite_configs = scaling_law_suite(
    sweep_name="scaling-law-suite-default-v2",
    tokenized=dclm_mixture_config_llama3,
    tags=["scaling_laws"],
)

all_steps = []

# name = path_to_step_name(COMPARISON_MODEL)
# all_steps.append(
#     ExecutorStep(
#             name=name,
#             fn=visualize_lm_log_probs,
#             config=VizLmConfig(
#                 checkpoint_path=COMPARISON_MODEL,
#                 model=llama_8b,
#                 datasets=eval_set_mixture,
#                 num_docs_per_dataset=32,
#                 comparison_model_path=None,
#             ),
#         )
# )

for i, checkpoint in enumerate(CHECKPOINTS):
    name = path_to_step_name(checkpoint)
    all_steps.append(
        ExecutorStep(
            name=name,
            fn=visualize_lm_log_probs,
            config=VizLmConfig(
                checkpoint_path=checkpoint,
                model=suite_configs[i],
                datasets=eval_set_mixture,
                num_docs_per_dataset=32,
                comparison_model_path=None#COMPARISON_MODEL if checkpoint != COMPARISON_MODEL else None,
            ),
        )
    )

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Visualize log probabilities of scaling law suite at different scales, and compare it to 8B",
    )
