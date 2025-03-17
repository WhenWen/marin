"""
Usage:

python marin/run/ray_run.py -- python -m marin.validation.find_top_urls \
    --input_path <file-path>
"""

import argparse
import json
import os
from collections import Counter
from urllib.parse import urlparse

import fsspec
import ray

from marin.utils import fsspec_glob

MAX_TASKS_IN_FLIGHT = 1000


def extract_domain_from_url(url):
    """Extract the main domain from a URL."""
    try:
        parsed_url = urlparse(url)
        # Get the network location (typically hostname)
        domain = parsed_url.netloc

        # Remove port number if present
        if ":" in domain:
            domain = domain.split(":")[0]

        # Remove 'www.' prefix if present
        if domain.startswith("www."):
            domain = domain[4:]

        return domain.lower()
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
        return "unknown"


def count_domains_in_file(filename: str) -> Counter:
    """Count occurrences of domains in a file."""
    domain_counter = Counter()

    with fsspec.open(filename, "rt", compression="infer") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Navigate to the URL in the nested structure
                if "metadata" in data and "WARC-Target-URI" in data["metadata"]:
                    url = data["metadata"]["WARC-Target-URI"]
                    domain = extract_domain_from_url(url)
                    if domain:
                        domain_counter[domain] += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {filename}: {e}")
            except Exception as e:
                print(f"Unexpected error processing file {filename}: {e}")

    return domain_counter


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def process_file(input_filename: str):
    """Process a file and return domain counts."""
    domain_counts = count_domains_in_file(input_filename)
    return domain_counts


def print_domain_histogram(domain_counter, top_n=100):
    """Print a histogram of domain counts."""
    total_domains = sum(domain_counter.values())
    unique_domains = len(domain_counter)

    print(f"\n{'=' * 60}")
    print(f"Domain Count Histogram (Top {top_n})")
    print(f"Total URLs processed: {total_domains}")
    print(f"Unique domains found: {unique_domains}")
    print(f"{'=' * 60}")
    print(f"{'Domain':<40} | {'Count':>10} | {'Percentage':>10}")
    print(f"{'-' * 40}-+-{'-' * 10}-+-{'-' * 10}")

    for domain, count in domain_counter.most_common(top_n):
        percentage = (count / total_domains) * 100
        print(f"{domain[:40]:<40} | {count:>10,} | {percentage:>9.2f}%")


def find_top_domains(input_path: str, filetype: str, top_n: int = 100) -> Counter:
    """Process all files and count domain occurrences."""
    responses = []
    domain_counter = Counter()

    input_paths = fsspec_glob(os.path.join(input_path, f"**/*.{filetype}"))
    print(f"Found {len(input_paths)} files to process")

    for input_file in input_paths:
        while len(responses) >= MAX_TASKS_IN_FLIGHT:
            ready_refs, responses = ray.wait(responses, num_returns=1)
            for result_ref in ready_refs:
                file_counter = ray.get(result_ref)
                domain_counter.update(file_counter)

        result_ref = process_file.remote(input_file)
        responses.append(result_ref)

    # Wait for all tasks to complete
    for result_ref in responses:
        file_counter = ray.get(result_ref)
        domain_counter.update(file_counter)

    # Print the histogram
    print_domain_histogram(domain_counter, top_n)

    return domain_counter


def main():
    parser = argparse.ArgumentParser(description="Find top domains in DCLM files.")
    parser.add_argument("--input_path", type=str, required=True, help="Input directory containing DCLM files")
    parser.add_argument("--filetype", type=str, default="jsonl.zst", help="Filetype of the input files")
    parser.add_argument("--top_n", type=int, default=100, help="Number of top domains to display")

    args = parser.parse_args()
    domain_counter = find_top_domains(args.input_path, args.filetype, args.top_n)
    print(f"Total unique domains found: {len(domain_counter)}")


if __name__ == "__main__":
    main()
