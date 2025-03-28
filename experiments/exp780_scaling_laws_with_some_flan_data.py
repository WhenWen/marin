"""
Runs scaling law experiments on DCLM mixture (3.8T baseline + 0.25T StarCoder + 0.055T ProofPile)
with 10% FLAN data (~0.4T tokens). Original components from the DCLM mixture are reduced to 90% to obtain
these proportions and maintain a total token count of ~4.1T.

See: https://github.com/stanford-crfm/marin/issues/780
"""

from defaults import default_scaling_law_pred

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3, dclm_components_llama3
from experiments.evals.task_configs import CORE_TASKS
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite

from marin.processing.tokenize import lm_mixture_data_config
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps


# Define mixture weights with 10% FLAN data (total ~4.105T tokens)
# Original components reduced to 90% to accommodate FLAN data
DCLM_MIXTURE_WITH_FLAN = {
    # Reduce existing components by ~90%
    "dclm_baseline": 3.42,  # 3.8 * 0.9
    "starcoderdata": 0.225,  # 0.25 * 0.9
    "proofpile_2": 0.0495,  # 0.055 * 0.9
    # Add FLAN as 10% of total
    "dolma/flan": 0.4105,  # ~10% of total tokens
}

# Create mixture config combining original DCLM components with FLAN
dclm_flan_mixture_config = lm_mixture_data_config(
    components={
        **dclm_components_llama3,  # Include original DCLM components
        "dolma/flan": tokenize_dolma_steps()["dolma/flan"],  # Add FLAN component
    },
    weights=DCLM_MIXTURE_WITH_FLAN
)

default_suite_with_some_flan_data = scaling_law_suite(
    sweep_name="scaling-law-suite-default-plus-flan",
    tokenized=dclm_flan_mixture_config, 
    tags=["scaling_laws"],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            *default_suite_with_some_flan_data,
        ],
        description="suite + predictions for scaling laws on DCLM-Baseline+StarCoder+ProofPile+Flan mix",
    )
