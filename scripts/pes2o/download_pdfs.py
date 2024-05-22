"""
file: download_pdfs.py
---
This file contains scripts to download the original PDFs from the pes2o dataset.
For more information about pes2o, see https://github.com/allenai/peS2o
"""

import sys
import os
import gzip
import json
import logging
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# request at https://www.semanticscholar.org/product/api
api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(message)s",
    stream=sys.stdout,
)

PES2O_FILEPATH = "data/zero.json.gz"  # single file for debugging
# TODO: figure out where to put these scratch/intermediate files
# TODO: use titles to validate we're pulling right data and then remove
PAPER_TITLE_FILEPATH = "data/paper_titles.txt"
PDF_URL_FILEPATH = "data/pdf_urls.txt"  # list of open accesspdf urls
NO_URL_FILEPATH = "data/no_urls.txt"  # list of jsons that don't have urls

DEBUG = 1


def extract_pdf_urls():
    # clear files if they exist
    if os.path.exists(PDF_URL_FILEPATH) or os.path.exists(NO_URL_FILEPATH) or os.path.exists(PAPER_TITLE_FILEPATH):
        if input("Files already exist. Overwrite? (y/n): ") != "y":
            return
    for url in [PDF_URL_FILEPATH, NO_URL_FILEPATH, PAPER_TITLE_FILEPATH]:
        if os.path.exists(url):
            os.remove(url)

    # first pull CorpusIds from pes2o data
    pes2o_ids = []
    with gzip.open(open(PES2O_FILEPATH, "rb"), "rt", encoding="utf-8") as f:
        for line in tqdm(f):
            if line:
                example = json.loads(line)
                pes2o_ids.append(example["id"])
    logging.info(f"Collected {len(pes2o_ids)} ids")

    titles_f = open(PAPER_TITLE_FILEPATH, "w")
    url_f = open(PDF_URL_FILEPATH, "w")
    no_urls_f = open(NO_URL_FILEPATH, "w")

    # query Semantic Scholar API for paper metadata
    open_access_count = 0
    for i in tqdm(range(0, len(pes2o_ids), 450)):  # api can only handle batches of 500
        batch = pes2o_ids[i : i + 450]
        print(f"Downloading batch {i} to {i+450}")
        # API schema: https://api.semanticscholar.org/api-docs/graph
        response = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            headers={"x-api-key": api_key},
            params={"fields": "paperId,corpusId,title,openAccessPdf"},
            json={"ids": [f"CorpusId:{id}" for id in batch]},
        ).json()

        for idx, paper in enumerate(response):
            try:
                title = paper["title"]
                # for sanity checking we're pulling the right papers from pes2o
                titles_f.write(title + "\n")

                open_access = paper["openAccessPdf"]
                if not open_access:
                    no_urls_f.write(json.dumps(paper) + "\n")
                    continue
                pdf_url = paper["openAccessPdf"]["url"]
                url_f.write(pdf_url + "\n")
                open_access_count += 1
            except:
                logging.error(f"Error processing paper with pes2o idx: {batch[idx]}. Server response: {paper}")

        if DEBUG:
            logging.info("Debug mode. Exiting.")
            break

    logging.info(
        f"Got {open_access_count} pdf urls out of {len(pes2o_ids)}. Percentage: {open_access_count*100/len(pes2o_ids)}%"
    )


if __name__ == "__main__":
    extract_pdf_urls()
