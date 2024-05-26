"""
file: download_pdfs.py
---
This file contains scripts to download the original PDFs from the pes2o dataset.
For more information about pes2o, see https://github.com/allenai/peS2o
"""

import sys
import os
import time
import gzip
import json
import logging
import requests
from dotenv import load_dotenv
from tqdm import tqdm
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

# PES2O_FILEPATH = "data/sample.json.gz"  # single file for debugging

DEBUG = 0


@sleep_and_retry
@limits(calls=1, period=2)  # 1 call per second + eps=1 to be safe
def check_limit():
    # empty call to check rate limit b/c ratelimit doesn't work globally (sigh)
    return


def _find_closest_match(abstract):
    """
    Some Semantic Scholar papers seem to have had their IDs changed.
    E.g., 239601370 -> 244347457, 123884760->263121666
    This lookup fuzzes the abstracts (the only other ID info in pes2o) to find the closest match.
    """
    first_line = abstract.split("\n")[0]
    check_limit()
    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"x-api-key": api_key},
        params={
            "query": first_line,
            "fields": "corpusId,title,abstract,externalIds,isOpenAccess,openAccessPdf",
            "limit": 10,
        },
        hooks={"response": lambda r, *args, **kwargs: r.raise_for_status()},
    ).json()
    if response and response["total"] > 0:
        for paper in response["data"]:
            if DEBUG:
                logging.info(f"Fuzzy match for: {paper['title']}")
            if fuzz.ratio(abstract, paper["abstract"]) > 90:
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
    response = requests.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        headers={"x-api-key": api_key},
        params={"fields": "corpusId,title,abstract,externalIds,isOpenAccess,openAccessPdf"},
        json={"ids": batch},  # manually validated this matches
        hooks={"response": lambda r, *args, **kwargs: r.raise_for_status()},
    ).json()
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


def extract_pdf_urls(pes2o_file, out_dir):
    # list of open accesspdf urls
    url_filepath = os.path.join(out_dir, "pdf_urls.csv")
    url_file = open(url_filepath, "w")
    url_file.write("corpusId,url\n")  # header

    # stores json records
    no_url_json_list = []  # list of open access papers without urls
    nopen_json_list = []  # list of non-open access papers

    # first pull CorpusIds from pes2o data
    pes2o_ids = []
    pes2o_abstracts = []
    with gzip.open(open(pes2o_file, "rb"), "rt", encoding="utf-8") as f:
        for line in f:
            if line:
                example = json.loads(line)
                pes2o_ids.append(example["id"])
                # TODO: need to revisit when move from s2ag
                pes2o_abstracts.append(example["text"])  # for fuzzy matching
    logging.info(f"Collected {len(pes2o_ids)} ids")

    # query Semantic Scholar API for paper metadata

    for i in tqdm(range(0, len(pes2o_ids), 500)):  # api can only handle batches of 500/10MB
        batch = [f"CorpusId:{id}" for id in pes2o_ids[i : i + 500]]
        response = _batch_query(batch)

        for idx, paper in enumerate(response):
            try:
                process_response(paper, url_file, no_url_json_list, nopen_json_list)
            except:
                pes2o_abstract = pes2o_abstracts[i + idx]
                closest_match = _find_closest_match(pes2o_abstract)
                if closest_match is not None:
                    process_response(closest_match, url_file, no_url_json_list, nopen_json_list)
                else:
                    logging.error(f"Error processing paper with pes2o id: {batch[idx]}. Server response: {paper}")

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


def download_open_access_pdf(id, url):
    # should be as simple as a wget...
    check_limit()
    response = requests.get(url)
    with open(f"data/pdfs/{id}.pdf", "wb") as f:
        f.write(response.content)


def run_analytics(out_dir):
    url_file = os.path.join(out_dir, "pdf_urls.csv")
    open_no_url_file = os.path.join(out_dir, "open_no_urls.json")
    nopen_file = os.path.join(out_dir, "nopen.json")

    open_with_url_count = len(open(url_file).readlines()) - 1  # subtract header
    open_no_url_count = len(json.load(open(open_no_url_file)))
    nopen_count = len(json.load(open(nopen_file)))

    total_len = open_with_url_count + open_no_url_count + nopen_count
    if total_len == 0:
        logging.error("No records found. Exiting.")
        return

    logging.info(f"Total records: {total_len}")
    logging.info(f"Open access with URL: {open_with_url_count}. Percentage: {open_with_url_count*100/total_len:.2f}%")
    logging.info(f"Open access without URL: {open_no_url_count}. Percentage: {open_no_url_count*100/total_len:.2f}%")
    logging.info(f"Non-open access: {nopen_count}. Percentage: {nopen_count*100/total_len:.2f}%")


if __name__ == "__main__":
    # files are stored in dir data/raw/s2orc
    s2orc_files = os.listdir("data/raw/s2orc")
    for file in tqdm(s2orc_files):
        if file.endswith(".json.gz"):
            logging.info(f"Processing file: {file}. Writing to data/processed/{file.split('.')[0]}")
            file_path = os.path.join("data/raw/s2orc", file)
            out_dir = os.path.join("data/processed", file.split(".")[0])
            os.makedirs(out_dir, exist_ok=True)
            extract_pdf_urls(file_path, out_dir)
            logging.info("Finished processing file.")
            run_analytics(out_dir)
