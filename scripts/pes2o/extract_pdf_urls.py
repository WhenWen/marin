"""
file: extract_pdf_urls.py
---
This file contains scripts to extract PDF URLs from the pes2o dataset.
It does so by querying the Semantic Scholar API for paper metadata.
For more information about pes2o, see https://github.com/allenai/peS2o
"""

import sys
import os
import json
import gzip
import logging
import random
import requests
import time
from tqdm import tqdm
from dotenv import load_dotenv
from thefuzz import fuzz
from ratelimit import limits, sleep_and_retry

load_dotenv()

# request at https://www.semanticscholar.org/product/api
api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(asctime)s: %(message)s",
    stream=sys.stdout,
)

DEBUG = 1


@sleep_and_retry
@limits(calls=1, period=1.5)  # 1 call per second + eps=0.5 to be safe
def check_limit():
    # empty call to check rate limit b/c ratelimit doesn't work globally (sigh)
    if DEBUG:
        logging.info("Rate limit check")


def _backoff_request(request, max_retries=10, backoff=2):
    retries = 0
    while retries < max_retries:
        try:
            response = request()
            return response
        except Exception as e:
            retries += 1
            delay = (2**retries) * backoff + random.uniform(0, 1)  # exp backoff with jitter
            if retries < max_retries:
                logging.warning(
                    f"Request failed (attempt {retries}/{max_retries}): {e}. Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            else:
                logging.error(f"Max retries exceeded. Giving up.")
                raise e


def extract_pdf_urls(pes2o_file, out_dir):
    """
    Takes a file from the pes2o dataset and splits entries into: - open access
    papers with pdf urls given in Semantic Scholar - open access papers without
    pdf urls (to be processed later) - non-open access papers

    It writes the results to various files. All metadata for papers with URLs
    is stripped aside from the Semantic Scholar CorpusId. However, one can
    recover the metadata by querying their API.
    - open access w/ url -> pdf_urls.csv: (only corpusId, url)
    - open access w/o url -> open_no_urls.json
    - non-open access -> nopen.json
    """
    # list of open accesspdf urls
    url_filepath = os.path.join(out_dir, "pdf_urls.csv")
    url_file = open(url_filepath, "w")
    url_file.write("corpusId,url\n")  # header

    # stores json records
    no_url_json_list = []  # list of open access papers without urls
    nopen_json_list = []  # list of non-open access papers

    # first pull CorpusIds from pes2o data
    pes2o_ids = []
    pes2o_titles = []
    with gzip.open(open(pes2o_file, "rb"), "rt", encoding="utf-8") as f:
        for line in tqdm(f):
            if line:
                example = json.loads(line)
                pes2o_ids.append(example["id"])
                # for fuzzy matching
                text = example["text"].split("\n")
                # find the first line that is not empty
                it = 0
                while it < len(text):
                    if text[it].strip() != "":
                        pes2o_titles.append(text[it])
                        break
                    it += 1
                if it == len(text):
                    pes2o_titles.append("")
    logging.info(f"Collected {len(pes2o_ids)} ids")

    # query Semantic Scholar API for paper metadata

    for i in tqdm(range(0, len(pes2o_ids), 500)):  # api can only handle batches of 500/10MB
        batch = [f"CorpusId:{id}" for id in pes2o_ids[i : i + 500]]
        response = _batch_query(batch)

        for idx, paper in enumerate(response):
            try:
                process_response(paper, url_file, no_url_json_list, nopen_json_list)
            except:
                closest_match = _find_closest_match(pes2o_titles[i + idx])
                if closest_match is not None:
                    process_response(closest_match, url_file, no_url_json_list, nopen_json_list)
                else:
                    logging.error(
                        f"Error processing paper with pes2o id: {batch[idx]}. Server response: {paper}"
                    )

        if DEBUG:
            logging.info("Debug mode. Exiting.")
            break

    # write jsons to files
    url_file.close()
    no_url_filepath = os.path.join(out_dir, "open_no_urls.json")
    with open(no_url_filepath, "w") as f:
        json.dump(no_url_json_list, f)
    nopen_filepath = os.path.join(out_dir, "nopen.json")
    with open(nopen_filepath, "w") as f:
        json.dump(nopen_json_list, f)


def _find_closest_match(first_line):
    """
    Some Semantic Scholar papers seem to have had their IDs changed.
    E.g., 239601370 -> 244347457, 123884760->263121666
    This lookup fuzzes the titles (found in pes2o text) to find the closest match.
    """
    check_limit()
    response = _backoff_request(
        lambda: requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"x-api-key": api_key},
            params={
                "query": first_line,
                "fields": "corpusId,title,authors,externalIds,isOpenAccess,openAccessPdf",
                "limit": 10,
            },
            hooks={"response": lambda r, *args, **kwargs: r.raise_for_status()},
        ).json(),
        max_retries=5,
        backoff=1,
    )
    if (response is not None) and (response["total"] > 0) and ("data" in response):
        for paper in response["data"]:
            if DEBUG:
                logging.info(f"Fuzzy match for: {paper['title']}")
            if fuzz.ratio(first_line, paper["title"]) > 90:
                return paper
    if DEBUG:
        logging.info(f"No fuzzy match found. Discarding record entry.")
    return None


def _batch_query(batch):
    """
    Looks up a batch of Semantic Scholar papers by their corpusId.
    """
    check_limit()
    if DEBUG:
        logging.info(f"Submitting batch request of size {len(batch)}...")
    response = _backoff_request(
        lambda: requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            headers={"x-api-key": api_key},
            params={"fields": "corpusId,title,authors,externalIds,isOpenAccess,openAccessPdf"},
            json={"ids": batch},  # manually validated this matches
            hooks={"response": lambda r, *args, **kwargs: r.raise_for_status()},
        ).json(),
        max_retries=5,
        backoff=1,
    )
    return response


def process_response(response, url_file, no_url_json_list, nopen_json_list):
    if response["isOpenAccess"]:
        if response["openAccessPdf"]:
            pdf_url = response["openAccessPdf"]["url"]
            url_file.write(f"{response['corpusId']},{pdf_url}\n")
        else:
            no_url_json_list.append(response)
    else:
        nopen_json_list.append(response)
