"""
Experiment to test different max_length values for the quality model classifier.

This experiment uses the labeled documents from mmlu_science_pipeline and trains
multiple quality models with different max_length values, then evaluates their performance.
"""

from dataclasses import dataclass

from experiments.evals.resource_configs import TPU_V6E_8_STRICT_PACK, ResourceConfig
from experiments.exp923_medu_mmlu import mmlu_science_pipeline
from marin.classifiers.hf.launch_ray_training import LaunchConfig, launch_training_with_ray
from marin.classifiers.hf.train_classifier import HFTrainingConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path


@dataclass
class MaxLengthAblationConfig:
    """Configuration for the max_length ablation experiment."""

    experiment_name: str = "mmlu-science-max-length-ablation"
    # List of max_length values to test
    max_lengths: list[int] = None
    # Resource config for training the models
    resource_config: ResourceConfig = TPU_V6E_8_STRICT_PACK

    def __post_init__(self):
        if self.max_lengths is None:
            # Default max_length values to test
            self.max_lengths = [256, 512, 1024, 2048, 4096]


def train_quality_model_with_max_length(
    labeled_documents: ExecutorStep,
    experiment_name: str,
    max_length: int,
    resource_config: ResourceConfig,
) -> ExecutorStep:
    """Train a quality filter model with a specific max_length.

    Inputs:
        labeled_documents: An ExecutorStep that represents the labeled documents.
        experiment_name: The name of the experiment.
        max_length: The max_length to use for the model.
        resource_config: The resource config to use for training the quality filter model.

    Outputs:
        An ExecutorStep that represents the quality filter model.
    """
    length_to_batch_size = {
        256: 384,
        512: 128,
        1024: 32,
        2048: 16,
        4096: 4,
    }

    # Train the model with the specified max_length
    classifier_remote = ExecutorStep(
        name=f"classifiers/medu-bert/{experiment_name}-length-{max_length}",
        fn=launch_training_with_ray,
        config=LaunchConfig(
            training_config=HFTrainingConfig(
                train_dataset="gs://marin-us-east1/documents/medu-datasets/mmlu-science-3423e5/converted",
                output_dir=this_output_path(),
                num_labels=1,
                target_column="label",
                max_length=max_length,  # Use the specified max_length
                train_size=0.9,
                eval_steps=100,
                save_steps=100,
                logging_steps=10,
                run_name=f"datashop-classifier-{experiment_name}-max-length-{max_length}",
                tpu_num_cores=resource_config.num_tpu,
                per_device_train_batch_size=length_to_batch_size[max_length],
            ),
            resource_config=resource_config,
        ),
    )

    return classifier_remote


def create_max_length_ablation_experiment(config: MaxLengthAblationConfig = None) -> list[ExecutorStep]:
    """Create steps for the max_length ablation experiment.

    This function gets the labeled documents from the mmlu_science_pipeline and creates
    multiple quality models with different max_length values.
    """
    if config is None:
        config = MaxLengthAblationConfig()

    # Get the labeled documents from the mmlu_science_pipeline
    labeled_documents = mmlu_science_pipeline.labeled_documents

    # Train multiple quality models with different max_length values
    steps = []
    for max_length in config.max_lengths:
        classifier = train_quality_model_with_max_length(
            labeled_documents=labeled_documents,
            experiment_name=config.experiment_name,
            max_length=max_length,
            resource_config=config.resource_config,
        )
        steps.append(classifier)

    return steps


if __name__ == "__main__":
    # Create the experiment with default config
    experiment_steps = create_max_length_ablation_experiment()

    # Run the experiment
    executor_main(experiment_steps)
