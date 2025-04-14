"""
Utility to convert JSONL files to Parquet format.
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
class ParquetConversionConfig:
    input_path: str
    output_path: str


@ray.remote(memory=0.5 * 1024 * 1024 * 1024)
@cached_or_construct_output(success_suffix="SUCCESS")
def convert_jsonl_to_parquet(input_path: str, output_path: str) -> bool:
    """
    Convert a JSONL file to Parquet format.
    
    Args:
        input_path: Path to the input JSONL file (can be compressed)
        output_path: Path to save the Parquet file
        
    Returns:
        True if successful
    """
    logger.info(f"Converting {input_path} to {output_path}")
    
    # Create records list
    records = []
    
    # Read the JSONL file
    import io
    from zstandard import ZstdDecompressor
    
    with fsspec.open(input_path, 'rb') as f:
        dctx = ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line in {input_path}, skipping")
                    continue
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Save to Parquet
    df.to_parquet(output_path)
    
    logger.info(f"Successfully converted {input_path} to {output_path}")


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def convert_shards_to_parquet(input_path: str, output_path: str) -> bool:
    """
    Convert all JSONL files in the input path to Parquet format.
    """
    MAX_CONCURRENT_SHARDS = 50

    shards = fsspec_glob(os.path.join(input_path, "*.jsonl.zst"))
    result_refs = []

    for shard in shards:
        if len(result_refs) > MAX_CONCURRENT_SHARDS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue
        
        shard_output_path = os.path.join(output_path, os.path.basename(shard).replace(".jsonl.zst", ".parquet"))
        result_refs.append(convert_jsonl_to_parquet.remote(shard, shard_output_path))

    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")

    logger.info(f"Successfully converted {input_path} to {output_path}")


@draccus.wrap()
def convert_medu_dump_to_parquet(cfg: ParquetConversionConfig) -> bool:
    """
    Convert a Medu dump to Parquet format.
    """
    logger.info(f"Converting {cfg.input_path} to {cfg.output_path}")

    shards = fsspec_glob(os.path.join(cfg.input_path, "local-*"))

    refs = []

    for shard in shards:
        refs.append(convert_shards_to_parquet.remote(shard, f"{cfg.output_path}/{os.path.basename(shard)}"))

    try:
        ray.get(refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")

    logger.info(f"Successfully converted {cfg.input_path} to {cfg.output_path}")