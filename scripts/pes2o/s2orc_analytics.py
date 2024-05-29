"""
file: pes2o_analytics.py
-------------------------
Scripts for analyzing scraped data from Semantic Scholar API.
"""

import os
import json
from collections import defaultdict


def eval_open_access(dir_paths, print_subdir_info=False):
    """
    Computes the number and proportion of open access papers.

    args:
        dir_paths (str or list of str):
            path to directory containing scraped data
        print_subdir_info (bool):
            whether to additionally print the stats for each subdirectory
    """
    if type(dir_paths) == str:
        dir_paths = [dir_paths]

    tot_num_url = 0
    tot_num_no_url = 0
    tot_num_nopen = 0
    for path in dir_paths:
        nopen_file_path = os.path.join(path, "nopen.json")
        no_url_file_path = os.path.join(path, "open_no_urls.json")
        urls_file_path = os.path.join(path, "pdf_urls.csv")

        num_url = len(open(urls_file_path).readlines()) - 1  # subtract header
        num_no_url = len(json.load(open(no_url_file_path)))
        num_nopen = len(json.load(open(nopen_file_path)))

        if print_subdir_info:
            print(f"Open access with URL: {num_url}")
            print(f"Open access, no URL:  {num_no_url}")
            print(f"No open access:       {num_nopen}\n")

        tot_num_url += num_url
        tot_num_no_url += num_no_url
        tot_num_nopen += num_nopen

    total = tot_num_nopen + tot_num_no_url + tot_num_url
    print(f"Open access with URL: {tot_num_url} ({tot_num_url / total:.2%}).")
    print(f"Open access, no URL:  {tot_num_no_url} ({tot_num_no_url / total:.2%}).")
    print(f"No open access:       {tot_num_nopen} ({tot_num_nopen / total:.2%}).")


def eval_nopen_external_ids(dir_paths, print_subdir_info=False):
    """
    Analyzes external sources of non-open access papers.

    args:
        ''' (see eval_open_access)
    """
    if type(dir_paths) == str:
        dir_paths = [dir_paths]

    indiv_src_count = defaultdict(int)  # str -> int
    src_comb_count = defaultdict(int)  # tuple[str] -> int

    def print_counts(src_count, comb_count):
        print(f"Individual sources (n={src_count['total']}):")
        # sort by descending order of value
        src_count = {k: v for k, v in sorted(src_count.items(), key=lambda item: item[1], reverse=True)}
        comb_count = {k: v for k, v in sorted(comb_count.items(), key=lambda item: item[1], reverse=True)}

        for src, count in src_count.items():
            if src != "total":
                print(f"{src}: {count} ({count / src_count['total']:.2%})")
        print("=" * 30)
        print("Source combinations:")
        for comb, count in comb_count.items():
            print(f"{comb}: {count}")

    for dir in dir_paths:
        nopen_file_path = os.path.join(dir, "nopen.json")
        nopen_data = json.load(open(nopen_file_path))

        local_src_count = defaultdict(int)
        local_comb_count = defaultdict(int)

        for paper in nopen_data:
            local_src_count["total"] += 1
            # remove corpusID, since it's not an external ID
            ext_ids = sorted(paper["externalIds"].keys())
            ext_ids.remove("CorpusId")
            if len(ext_ids) == 0:
                local_src_count["No external IDs"] += 1
            src_comb_count[tuple(ext_ids)] += 1
            for src in ext_ids:
                local_src_count[src] += 1

        if print_subdir_info:
            print(f"Directory: {dir}")
            print_counts(local_src_count, local_comb_count)
            print()

        for src, count in local_src_count.items():
            indiv_src_count[src] += count
        for comb, count in local_comb_count.items():
            src_comb_count[comb] += count

    print_counts(indiv_src_count, src_comb_count)


if __name__ == "__main__":
    dirs = []
    for dir in os.listdir("data/processed"):
        # if dir contains a nopen.json file
        if os.path.exists(f"data/processed/{dir}/nopen.json"):
            dirs.append(f"data/processed/{dir}")
    eval_nopen_external_ids(dirs)
