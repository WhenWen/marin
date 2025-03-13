"""
Experiment TBD: Train (Fasttext) quality classifiers on reasoning data as positive examples.

See TBD for more details.
"""

import os
from dataclasses import dataclass, field

from marin.classifiers.utils import DatasetConfig
from marin.core.runtime import TaskConfig
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.classification.config.inference_config import RuntimeConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.fasttext.train_fasttext import (
    TrainFasttextClassifierConfig,
)
from marin.processing.classification.fasttext.train_fasttext import (
    train as train_fasttext,
)
from marin.processing.classification.inference import InferenceConfig, run_inference
from operations.transform.conversation.conversation_to_dolma import (
    ConversationToDolmaConfig,
)
from operations.transform.conversation.conversation_to_dolma import (
    process_dataset as convert_to_dolma,
)


@dataclass
class ExperimentConfig:
    """Configuration for comparing BERT vs Fasttext classifiers.

    This config defines parameters for an experiment that:
    1. Takes input documents from specified data sources
    2. Takes positive and negative documents from specified data sources.
    3. Trains both a BERT and a Fasttext classifier on the positive/negative documents.
    4. Filters and tokenizes the resulting data.

    Args:
        experiment_name: Identifier for this experiment
        input_data_source_to_path: Mapping of data source names to their GCS paths
        keep_fraction: Fraction of highest-quality documents to keep after filtering
    """

    experiment_name: str
    classifier_training_datasets: list[DatasetConfig]
    input_data_source_to_path: dict[str, str] = field(
        default_factory=lambda: {
            "fineweb_2024_18": (
                "gs://marin-us-central2/documents/fineweb-small-resiliparse-preserve-formatting-v2-e72837/md/CC-MAIN-2024-18/"
            ),
            "dclm_fineweb_edu": "gs://marin-us-central2/documents/quality_filtering/fineweb-edu",
            "stackexchange": "gs://marin-us-central2/documents/stackexchange",
            "dclm_hq": "gs://marin-us-central2/documents/quality_filtering/original-dclm-quality-classifier",
            "medu_economics": (
                "gs://marin-us-central2/documents/quality_filtering/dclm-global-shard-01-of-10-medu-economics-3plus-bbb96b"
            ),
        }
    )
    keep_fraction: float = 0.05  # Keep 5% of the documents


def get_model_path(model_path: str | ExecutorStep):
    if isinstance(model_path, ExecutorStep):
        return output_path_of(model_path, "model.bin")
    return versioned(model_path)


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create the steps for a single experiment.

    Variation of exp614_quality_filtering.py.
    """

    steps = []

    fasttext_classifier_train = ExecutorStep(
        name=f"classifiers/{config.experiment_name}/fasttext",
        fn=train_fasttext,
        config=TrainFasttextClassifierConfig(
            datasets=config.classifier_training_datasets,
            output_path=this_output_path(),
            fasttext_args={"lr": versioned(0.1), "thread": 4, "wordNgrams": 2},
            val_frac=versioned(0.0),
            seed=versioned(0),
        ),
        pip_dependency_groups=["fasttext"],
    )

    for input_data_source, input_data_path in config.input_data_source_to_path.items():
        # Get the basename of the input directory
        input_basename = os.path.basename(os.path.normpath(input_data_path))

        fasttext_inference = ExecutorStep(
            name=f"attributes/quality_filtering/{config.experiment_name}/fasttext/{input_data_source}",
            fn=run_inference,
            config=InferenceConfig(
                input_path=input_data_path,
                output_path=this_output_path(input_basename),
                model_name=get_model_path(fasttext_classifier_train),
                model_type="fasttext",
                attribute_name=versioned(f"{config.experiment_name}-fasttext_classifier"),
                runtime=RuntimeConfig(
                    memory_limit_gb=12,
                ),
                task=TaskConfig(max_in_flight=500),
            ),
            pip_dependency_groups=["fasttext", "datasets", "filelock"],
        )

        fasttext_consolidate_step = ExecutorStep(
            name=f"documents/quality_filtering/{config.experiment_name}/fasttext/{input_data_source}",
            fn=consolidate,
            config=ConsolidateConfig(
                input_path=input_data_path,
                output_path=this_output_path(input_basename),
                filters=[
                    FilterConfig(
                        type=versioned("classify"),
                        attribute_path=output_path_of(fasttext_inference, input_basename),
                        name=versioned(f"{config.experiment_name}-fasttext_classifier"),
                        label="__label__hq",
                        threshold=versioned(None),
                        keep_fraction=versioned(config.keep_fraction),
                    ),
                ],
                ray_memory_limit_gb=12,
            ),
            pip_dependency_groups=["ddsketch"],
        )

        # fasttext_tokenize_step = default_tokenize(
        #     name=f"quality_filtering/{config.experiment_name}/fasttext/{input_data_source}",
        #     dataset=output_path_of(fasttext_consolidate_step),
        #     tokenizer=llama3_tokenizer,
        # )

        steps.append(fasttext_consolidate_step)
        # steps.append(fasttext_tokenize_step)

    return steps


def main():
    experiment_name = "reasoning_fasttext_classifier"
    reasoning_in_dolma_format = ExecutorStep(
        name=f"documents/{experiment_name}/reasoning_in_dolma_format",
        fn=convert_to_dolma,
        config=ConversationToDolmaConfig(
            input_path=versioned("gs://marin-us-central2/documents/facebook--natural_reasoning-main-6067ba/"),
            output_path=this_output_path(),
        ),
    )
    classifier_training_datasets = [
        DatasetConfig(
            input_doc_path=output_path_of(reasoning_in_dolma_format),
            label="hq",
            sampling_rate=1.0,
            max_sample_size=versioned(100000),
        ),
        DatasetConfig(
            input_doc_path="gs://marin-us-central2/documents/dclm_negative_examples-bd7218",
            label="lq",
            sampling_rate=1.0,
            max_sample_size=versioned(100000),
        ),
    ]
    experiment_config = ExperimentConfig(
        experiment_name=experiment_name,
        classifier_training_datasets=classifier_training_datasets,
    )

    steps = create_steps(experiment_config)
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
