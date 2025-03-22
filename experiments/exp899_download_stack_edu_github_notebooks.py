from experiments.pretraining_datasets import stack_edu, issues_kaggle_notebooks
from marin.execution.executor import executor_main

if __name__ == "__main__":
    executor_main(
        steps=[
            stack_edu,
            issues_kaggle_notebooks,
        ],
    )
