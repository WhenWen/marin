from marin.scaling_laws.create_ladder_suite import create_smaller_ladder_suite
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from marin.execution.executor import executor_main
from experiments.dolma.exp442_dolma import dolma_llama3_tokenized


model_table = [
    {"name": "4M", "hidden_dim": 64, "num_layers": 8, "num_heads": 8, "batch_size": 32, "learning_rate": 1.4e-2},
    {"name": "6M", "hidden_dim": 96, "num_layers": 8, "num_heads": 8, "batch_size": 32, "learning_rate": 1.2e-2},
    {"name": "8M", "hidden_dim": 128, "num_layers": 8, "num_heads": 8, "batch_size": 32, "learning_rate": 1.1e-2},
    {"name": "10M", "hidden_dim": 144, "num_layers": 8, "num_heads": 8, "batch_size": 32, "learning_rate": 1.0e-2},
    {"name": "14M", "hidden_dim": 192, "num_layers": 8, "num_heads": 8, "batch_size": 32, "learning_rate": 9.2e-3},
    {"name": "16M", "hidden_dim": 208, "num_layers": 8, "num_heads": 8, "batch_size": 32, "learning_rate": 8.9e-3},
    {"name": "20M", "hidden_dim": 192, "num_layers": 16, "num_heads": 8, "batch_size": 64, "learning_rate": 8.4e-3},
    {"name": "60M", "hidden_dim": 384, "num_layers": 16, "num_heads": 12, "num_kv_heads": 12, "batch_size": 96, "learning_rate": 5.8e-3},
    {"name": "90M", "hidden_dim": 528, "num_layers": 16, "num_heads": 12, "num_kv_heads": 12, "batch_size": 160, "learning_rate": 4.9e-3},
]

suite = create_smaller_ladder_suite(
    sweep_name="small-suite-dolma",
    tokenized=dolma_llama3_tokenized,
    tags=["small_models", "scaling_laws"],
    model_table=model_table,
    eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
)

if __name__ == "__main__":
    executor_main(
        steps=suite,
        description="Suite for scaling laws on small models (4M-90M) with Dolma mix",
    )
# print(suite)
