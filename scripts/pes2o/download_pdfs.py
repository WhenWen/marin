"""
file: download_pdfs.py
---
This file contains scripts to download the original PDFs from the pes2o dataset.
For more information about pes2o, see https://github.com/allenai/peS2o
"""

import sys
import os
import time
import json
import logging
import random
import requests
import timeit
import threading
import concurrent.futures
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(asctime)s: %(message)s",
    stream=sys.stdout,
)


class RateLimitedScraper:
    """
    Issues requests with a rate limiting mechanism to prevent getting blocked
    """

    def __init__(self, domains, max_workers=20):
        domains = set(domains)  # dedup

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.domain_mutexes = {domain: threading.Lock() for domain in domains}
        self.last_domain_access = {domain: timeit.default_timer() for domain in domains}
        self.domain_delays = {}

        crawl_delay_file = "crawl_delay.jsonl"

        if os.path.exists(crawl_delay_file):
            logging.info("Loading crawl delays from file")
            self.load_crawl_delay_from_file(crawl_delay_file)
        else:
            logging.info("No crawl delay file found. Fetching from robots.txt")
            self.get_crawl_delay(domains)
            self.serialize_crawl_delay_to_file(crawl_delay_file)

    def get_crawl_delay(self, domain_set):
        # save crawl delays from robots.txt
        for domain in tqdm(domain_set):
            self.domain_delays[domain] = 60  # default is a minute

            # since domains (should be) unique, scrape without limiting
            robots = f"https://{domain}/robots.txt"
            try:
                response = self.thread_pool.submit(requests.get, robots, timeout=3).result()
                self.last_domain_access[domain] = timeit.default_timer()
                for line in response.text.split("\n"):
                    if "Crawl-delay" in line:
                        self.domain_delays[domain] = int(line.split(":")[1].strip())
                        break
            except Exception as e:
                logging.error(f"Failed to scrape {robots}: {e}")

    def load_crawl_delay_from_file(self, file):
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                self.domain_delays[data["domain"]] = data["delay"]

    def serialize_crawl_delay_to_file(self, file):
        with open(file, "w") as f:
            for domain, delay in self.domain_delays.items():
                f.write(json.dumps({"domain": domain, "delay": delay}) + "\n")

    def manual_crawl_delay_updates(self):
        # manual updates for certain domains
        pass

    def scrape_with_limiting(self, url):
        """
        Given a list of domains, spawn threads to scrape each domain. Uses
        internal rate limiting system to prevent overloading servers.
        """
        domain = url.split("/")[2]
        self.domain_mutexes[domain].acquire()
        # check if we need to wait
        time_since_last_access = timeit.default_timer() - self.last_domain_access[domain]
        if time_since_last_access < self.domain_delays[domain]:
            time.sleep(self.domain_delays[domain] - time_since_last_access)
        try:
            response = self.thread_pool.submit(requests.get, url).result()
        except Exception as e:
            logging.error(f"Failed to scrape {url}: {e}")
            response = None
        self.last_domain_access[domain] = timeit.default_timer()
        self.domain_mutexes[domain].release()
        return response


def gather_open_access_urls(url_csv_dir):
    # gathers all .csv files in the directory or any subdirectories and
    # extracts their contents
    urls = []
    for root, _, files in os.walk(url_csv_dir):
        for file in files:
            if file.endswith(".csv"):
                with open(os.path.join(root, file), "r") as f:
                    # skip csv header
                    next(f)
                    for line in f:
                        # url can have commas
                        corpus_id, url = line.strip().split(",", 1)
                        urls.append((corpus_id, url))
    return urls


def download_open_access_pdf(url, outfile):
    # should be as simple as a wget...
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/8.8 (Macintosh; Intel Mac OS X 8888_8888) AppleWebKit/888.8.88 (KHTML, like Gecko) Version/88.8.8 Safari/888.8.88",
            # referer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
        },
    )
    if response.status_code != 200:
        logging.error(f"Failed to download {url}. Response: {response.status_code}")
    else:
        with open(outfile, "wb") as f:
            f.write(response.content)

    return url


if __name__ == "__main__":
    urls = gather_open_access_urls("data/processed/")
    rate_limiter = RateLimitedScraper([url.split("/")[2] for _, url in urls])
