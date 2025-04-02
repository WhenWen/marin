"""
This script runs a suite of scaling laws on the Dolma mix.

Link to issue: https://github.com/stanford-crfm/marin/issues/780
"""

from defaults import default_scaling_law_pred

from experiments.dolma.exp442_dolma import dolma_llama3_tokenized
from experiments.evals.task_configs import CORE_TASKS
from marin.execution.executor import executor_main
from marin.scaling_laws.create_ladder_suite import scaling_law_suite

print("Starting Dolma suite")

dolma_suite = scaling_law_suite(
    sweep_name="scaling-law-suite-dolma-v2",
    tokenized=dolma_llama3_tokenized,
    tags=["scaling_laws"],
)

RUNS = [
    "scaling-law-suite-dolma-v2-512-b899a2",
    "scaling-law-suite-dolma-v2-768-7d27cb",
    "scaling-law-suite-dolma-v2-1024-b13520",
    "scaling-law-suite-dolma-v2-1536-ace8cb",
    "scaling-law-suite-dolma-v2-2048-ea678e",
]

dolma_suite_scaling_laws_pred = default_scaling_law_pred(
    ladder_runs=RUNS,  # dolma_suite,
    pred_run="llama-8b-tootsie-dolma-0.001-565941",  # this will give us readouts at various scales
    task_losses=(
        "eval/paloma/c4_en/bpb",
        "eval/bpb",
        "eval/loss",
        "eval/paloma/c4_en/loss",
    ),
    task_accuracies=CORE_TASKS,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            # *dolma_suite,
            dolma_suite_scaling_laws_pred,
        ],
        description="suite for scaling laws on Dolma mix",
    )
