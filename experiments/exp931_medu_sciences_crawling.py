"""
This script crawls the open-web-math dataset, which executed the `default_crawl` step for the dataset.

Link to issue: https://github.com/stanford-crfm/marin/issues/868
"""
import re
import os
import json
import pandas as pd

from experiments.crawl.default import default_crawl
from marin.crawl.common.schemas import HtmlExtractionConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.crawl.jsonl_to_parquet import ParquetConversionConfig, convert_medu_dump_to_parquet
from marin.utils import fsspec_glob


def warc_path_extractor(df: pd.DataFrame) -> str:
    def get_warc_path(x: str) -> str:
        match = re.search(r'isPartOf:\s*(CC-MAIN-\d{4}-\d{2})', x["warcinfo"])
        is_part_of = match.group(1)

        url = x["url"]

        return {
            "target_uri": url,
            "split_id": is_part_of,
        }

    return df.apply(lambda row: get_warc_path(row), axis=1)


medu_jsonl_to_parquet_step = ExecutorStep(
    name="documents/medu-dclm-pretraining-subset-mmlu-science",
    fn=convert_medu_dump_to_parquet,
    config=ParquetConversionConfig(
        input_path="gs://marin-us-east1/documents/quality_filtering/medu/medu-dclm-pretraining-subset-mmlu-science-217322/",
        output_path=this_output_path(),
    ),
)


medu_crawling_steps = []
medu_dumps = fsspec_glob("gs://marin-us-central2/documents/medu-dclm-pretraining-subset-mmlu-science-f8d04c/local-*")


for dump in medu_dumps:
    steps = default_crawl(
        config=HtmlExtractionConfig(
            input_path=dump,
            output_path=this_output_path(),
            source_name=f"medu/medu-dclm-pretraining-subset-mmlu-science/{os.path.basename(dump)}",
            columns=[
                "bff_contained_ngram_count_before_dedupe",
                "language_id_whole_page_fasttext",
                "metadata",
                "previous_word_count",
                "text",
                "url",
                "warcinfo",
                "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob",
                "id",
            ],
            warc_path_extractor=warc_path_extractor,
            url_column="url",
            file_path_column="file_path",
        ),
        yield_fn=None,
        input_pattern="local-*/*_links.jsonl.gz",
    )
    medu_crawling_steps.extend(steps)


if __name__ == "__main__":
    executor_main(
        steps=[
            medu_jsonl_to_parquet_step,
            *medu_crawling_steps,
        ],
    )
