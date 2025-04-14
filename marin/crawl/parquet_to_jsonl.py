"""
Utility to convert Parquet files to JSONL format.
"""
import os
import ray
import json
import fsspec
import draccus
import logging
import pandas as pd

from dataclasses import dataclass

from marin.utils import fsspec_glob
from marin.core.runtime import cached_or_construct_output

logger = logging.getLogger("ray")


@dataclass
class JsonlConversionConfig:
    input_path: str
    output_path: str


@ray.remote(memory=0.5 * 1024 * 1024 * 1024)
@cached_or_construct_output(success_suffix="SUCCESS")
def convert_parquet_to_jsonl(input_path: str, output_path: str) -> bool:
    """
    Convert a Parquet file to JSONL format.
    
    Args:
        input_path: Path to the input Parquet file
        output_path: Path to save the JSONL file (will be compressed with zstandard)
        
    Returns:
        True if successful
    """
    logger.info(f"Converting {input_path} to {output_path}")
    
    # Read the Parquet file
    df = pd.read_parquet(input_path)
    
    # Convert records to JSON strings
    records = df.to_dict(orient='records')
    
    # Write to JSONL with zstandard compression
    import io
    from zstandard import ZstdCompressor
    
    with fsspec.open(output_path, 'wb') as f:
        cctx = ZstdCompressor(level=3)  # Adjust compression level as needed
        with cctx.stream_writer(f) as writer:
            text_buffer = io.StringIO()
            for record in records:
                json_line = json.dumps(record) + '\n'
                text_buffer.write(json_line)
            
            writer.write(text_buffer.getvalue().encode('utf-8'))
    
    logger.info(f"Successfully converted {input_path} to {output_path}")
    return True


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def convert_shards_to_jsonl(input_path: str, output_path: str) -> bool:
    """
    Convert all Parquet files in the input path to JSONL format.
    """
    MAX_CONCURRENT_SHARDS = 50

    shards = fsspec_glob(os.path.join(input_path, "*.parquet"))
    result_refs = []

    for shard in shards:
        if len(result_refs) > MAX_CONCURRENT_SHARDS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue
        
        shard_output_path = os.path.join(output_path, os.path.basename(shard).replace(".parquet", ".jsonl.zst"))
        result_refs.append(convert_parquet_to_jsonl.remote(shard, shard_output_path))

    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")

    logger.info(f"Successfully converted {input_path} to {output_path}")
    return True


@draccus.wrap()
def convert_medu_dump_to_jsonl(cfg: JsonlConversionConfig) -> bool:
    """
    Convert a Medu dump in Parquet format back to JSONL format.
    """
    logger.info(f"Converting {cfg.input_path} to {cfg.output_path}")

    cfg.output_path = "gs://marin-us-east1/documents/quality_filtering/medu/medu-dclm-pretraining-subset-mmlu-science-217322"

    shards = fsspec_glob(os.path.join(cfg.input_path, "local-*"))

    refs = []

    for shard in shards:
        refs.append(convert_shards_to_jsonl.remote(shard, f"{cfg.output_path}/{os.path.basename(shard)}"))

    try:
        ray.get(refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")

    logger.info(f"Successfully converted {cfg.input_path} to {cfg.output_path}")
    return True 