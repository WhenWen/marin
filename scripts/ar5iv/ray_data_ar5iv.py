import ray

import argparse
import json
import os
import time
import traceback

import fsspec
import zipfile
import requests
import datetime

from marin.utils import get_gcs_path
# from scripts.ar5iv.utils import get_ar5iv_success_path
from marin import markdown
import re
from bs4 import BeautifulSoup
import markdownify

file = ray.data.read_json("gcs://marin-data/processed/ar5iv/html_clean/2024-06-16/no-problem", arrow_open_stream_args={"compression": "gzip"})


ctx = ray.data.DataContext.get_current()
# ctx.execution_options.resource_limits.cpu = 10
# ctx.execution_options.resource_limits.gpu = 5
# ctx.execution_options.resource_limits.object_store_memory = 10e9
ctx.execution_options.preserve_order = True

def markdownify_ar5iv_html(file):
    if type(file) is dict:
        file["text"][0] = markdown.MyMarkdownConverter().convert_soup(BeautifulSoup(file["text"][0], "html.parser"))
    elif type(file) is list:
        for i in range(len(file)):
            file[i]["text"][0] = markdown.MyMarkdownConverter().convert_soup(BeautifulSoup(file[i]["text"][0], "html.parser"))
    return file

# print(file.schema())
file.map_batches(markdownify_ar5iv_html, batch_size=256).write_jsonl("gcs://marin-data/processed/ar5iv/tmp_md_out/2024-06-16/no-problem", arrow_open_stream_args={"compression": "gzip"})