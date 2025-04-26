from itertools import chain

from marin.execution.executor import executor_main

from experiments.curriculum.cpt.cpt_launch import full_cpt_varying_mixture

# 2000 steps is 1B tokens

base_steps = {
    "finemath": 200,
    "open-web-math": 20000,
    "pubmed": 100,
    "flan": 20,
    "latxa": 400,
    "spj": 2000,
}

if __name__ == "__main__":
    stage_pairs = [
        full_cpt_varying_mixture(
            data1_name=data1_name,
            data2_name=data2_name,
            total_data1_portion=total_data1_portion,
            duration_frac_stage2=1.0,  # Single stage training
            data1_frac_alloc_stage2=1.0,
            schedule_type="cosine",
            model_name="meta-llama/Meta-Llama-3.1-8B",
            num_train_steps=int(base_steps[data1_name] * num_data1_repetitions / total_data1_portion),
            learning_rate=lr,
            num_eval=4 if base_steps[data1_name] == 20 else 20,
            num_lm_eval_harness=4,
            num_data1_repetitions=num_data1_repetitions,
            batch_size=128,
            additional_tags=[f"{data1_name}-replay-sweep-4-1"],
            min_lr_ratio=0.0,
            version_tag=f"-lr-{lr}",
            warmup_steps=0.05,
            # data_seed=43,
            # weight_decay=weight_decay,
        )
        for total_data1_portion in [1.0, 0.75, 0.5, 0.25, 0.1]
        for data1_name in ["latxa"]
        for data2_name in ["spj"]
        for num_data1_repetitions in [2]
        for lr in [3e-5]
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Train on finemath with varying rarity",
    )