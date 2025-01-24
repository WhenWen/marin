"""
Test different html->text transformation methods (on Wikipedia Dump, train 1.4B models).
https://github.com/stanford-crfm/marin/issues/647
"""

import logging

from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.evals.evals import default_eval
from experiments.llama import llama_1_4b, llama_1_4b_train_config

from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")

EXPERIMENT_TAG = ["wiki_subbed_dolma"]

weights = DOLMA_OLMO_MIXTURE_WEIGHTS.pop("dolma/wiki")


# No References, No Links
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki-subbed-resiliparse-with-preserving-formatting-no-references-no-links-sanity"] = weights

wiki_resiliparse_no_refs_no_links_path = "gs://marin-us-central2/documents/wikipedia-resiliparse-with-preserving-formatting-no-references-cb4306/20241201/*.jsonl.gz"
wiki_resiliparse_no_refs_no_links_files = fsspec_glob(wiki_resiliparse_no_refs_no_links_path)

wiki_resiliparse_subbed_dolma_no_refs_no_link_tokenized = tokenize_dolma_steps(
    substitute={"wiki": wiki_resiliparse_no_refs_no_links_files},
    prefix="resiliparse-with-preserving-formatting-no-references-no-links-sanity",
)

wiki_resiliparse_no_refs_no_links_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=wiki_resiliparse_subbed_dolma_no_refs_no_link_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS
)

wiki_resiliparse_no_refs_no_links_subbed_dolma_1_4b_model = default_train(
    name="dolma-wiki-resiliparse-no-refs-no-links-sanity",
    tokenized=wiki_resiliparse_no_refs_no_links_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

wiki_resiliparse_no_refs_no_links_subbed_dolma_1_4b_evals = default_eval(step=wiki_resiliparse_no_refs_no_links_subbed_dolma_1_4b_model)


# No References, With Links
DOLMA_OLMO_MIXTURE_WEIGHTS.pop("dolma/wiki-subbed-resiliparse-with-preserving-formatting-no-references-no-links-sanity")
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki-subbed-resiliparse-with-preserving-formatting-no-references-with-links-sanity"] = weights

wiki_resiliparse_no_refs_with_links_path = "gs://marin-us-central2/documents/wikipedia-resiliparse-with-preserving-formatting-no-references-with-links-fc8d08/20241201/*.jsonl.gz"
wiki_resiliparse_no_refs_with_links_files = fsspec_glob(wiki_resiliparse_no_refs_with_links_path)

wiki_resiliparse_subbed_dolma_no_refs_with_links_tokenized = tokenize_dolma_steps(
    substitute={"wiki": wiki_resiliparse_no_refs_with_links_files},
    prefix="resiliparse-with-preserving-formatting-no-references-with-links-sanity",
)

wiki_resiliparse_no_refs_with_links_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=wiki_resiliparse_subbed_dolma_no_refs_with_links_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS
)

wiki_resiliparse_no_refs_with_links_subbed_dolma_1_4b_model = default_train(
    name="dolma-wiki-resiliparse-no-refs-with-links-sanity",
    tokenized=wiki_resiliparse_no_refs_with_links_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

wiki_resiliparse_no_refs_with_links_subbed_dolma_1_4b_evals = default_eval(step=wiki_resiliparse_no_refs_with_links_subbed_dolma_1_4b_model)


# With References, No Links
weights = DOLMA_OLMO_MIXTURE_WEIGHTS.pop("dolma/wiki-subbed-resiliparse-with-preserving-formatting-no-references-with-links-sanity")
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki-subbed-resiliparse-with-preserving-formatting-with-references-no-links-sanity"] = weights

wiki_resiliparse_with_refs_no_links_path = "gs://marin-us-central2/documents/wikipedia-resiliparse-with-preserving-formatting-with-references-a1e27d/20241201/*.jsonl.gz"
wiki_resiliparse_with_refs_no_links_files = fsspec_glob(wiki_resiliparse_with_refs_no_links_path)

wiki_resiliparse_subbed_dolma_with_refs_no_links_tokenized = tokenize_dolma_steps(
    substitute={"wiki": wiki_resiliparse_with_refs_no_links_files},
    prefix="resiliparse-with-preserving-formatting-with-references-no-links-sanity",
)

wiki_resiliparse_with_refs_no_links_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=wiki_resiliparse_subbed_dolma_with_refs_no_links_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS
)

wiki_resiliparse_with_refs_no_links_subbed_dolma_1_4b_model = default_train(
    name="dolma-wiki-resiliparse-with-refs-no-links-sanity",
    tokenized=wiki_resiliparse_with_refs_no_links_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

wiki_resiliparse_with_refs_no_links_subbed_dolma_1_4b_evals = default_eval(step=wiki_resiliparse_with_refs_no_links_subbed_dolma_1_4b_model)


# With References, With Links
weights = DOLMA_OLMO_MIXTURE_WEIGHTS.pop("dolma/wiki-subbed-resiliparse-with-preserving-formatting-with-references-no-links-sanity")
DOLMA_OLMO_MIXTURE_WEIGHTS["dolma/wiki-subbed-resiliparse-with-preserving-formatting-with-references-with-links-sanity"] = weights

wiki_resiliparse_with_refs_with_links_path = "gs://marin-us-central2/documents/wikipedia-resiliparse-with-preserving-formatting-with-references-with-links-dd7e6b/20241201/*.jsonl.gz"
wiki_resiliparse_with_refs_with_links_files = fsspec_glob(wiki_resiliparse_with_refs_with_links_path)

wiki_resiliparse_subbed_dolma_with_refs_with_links_tokenized = tokenize_dolma_steps(
    substitute={"wiki": wiki_resiliparse_with_refs_with_links_files},
    prefix="resiliparse-with-preserving-formatting-with-references-with-links-sanity",
)

wiki_resiliparse_with_refs_with_links_subbed_dolma_llama3_tokenized = lm_mixture_data_config(
    components=wiki_resiliparse_subbed_dolma_with_refs_with_links_tokenized,
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS
)

wiki_resiliparse_with_refs_with_links_subbed_dolma_1_4b_model = default_train(
    name="dolma-wiki-resiliparse-with-refs-with-links-sanity",
    tokenized=wiki_resiliparse_with_refs_with_links_subbed_dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

wiki_resiliparse_with_refs_with_links_subbed_dolma_1_4b_evals = default_eval(step=wiki_resiliparse_with_refs_with_links_subbed_dolma_1_4b_model)


if __name__ == "__main__":
    tokenization_steps = [
        *wiki_resiliparse_subbed_dolma_no_refs_no_link_tokenized.values(),
        *wiki_resiliparse_subbed_dolma_no_refs_with_links_tokenized.values(),
        *wiki_resiliparse_subbed_dolma_with_refs_no_links_tokenized.values(),
        *wiki_resiliparse_subbed_dolma_with_refs_with_links_tokenized.values(),
    ]

    executor_main(
        steps=[
            *tokenization_steps,
            wiki_resiliparse_no_refs_no_links_subbed_dolma_1_4b_model,
            wiki_resiliparse_no_refs_no_links_subbed_dolma_1_4b_evals,
            wiki_resiliparse_no_refs_with_links_subbed_dolma_1_4b_model,
            wiki_resiliparse_no_refs_with_links_subbed_dolma_1_4b_evals,
            wiki_resiliparse_with_refs_no_links_subbed_dolma_1_4b_model,
            wiki_resiliparse_with_refs_no_links_subbed_dolma_1_4b_evals,
            wiki_resiliparse_with_refs_with_links_subbed_dolma_1_4b_model,
            wiki_resiliparse_with_refs_with_links_subbed_dolma_1_4b_evals
        ]
    )
