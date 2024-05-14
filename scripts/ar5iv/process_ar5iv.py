# import tempfile
# from dataclasses import dataclass

# import json
# import os
# import xml.etree.ElementTree as ET
# from typing import Optional

# import fsspec
# import tqdm
# from fsspec.callbacks import TqdmCallback
from markweb import markdown
import re
from markweb.markweb import convert_page
from bs4 import BeautifulSoup
import markdownify

class Ar5ivMarkdownConverter(markdown.MyMarkdownConverter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def to_markdown(html):
    if isinstance(html, str):
        html = BeautifulSoup(html, "html.parser")
    authors = html.find('div', {'class': 'ltx_authors'})
    if authors:
        authors.decompose()
    title_page = html.find('div', {'class': 'ltx_titlepage'})
    if title_page:
        title_page.decompose()
    html.find('title').decompose()
    text = Ar5ivMarkdownConverter().convert_soup(html)
    # cleanup: replace nbsp as space
    # this isn't quite right if we preserve html in places, but we currently are not doing that
    text = text.replace("\xa0", " ").strip()
    return text
import glob
import os
from urllib.request import urlopen
import json

# for file in ["/juice4/scr4/nlp/crfm/markweb/ar5iv/error/0003/math0003131.html"]:
def process_file(file):
    paper = os.path.basename(file)[:-5]
    print(paper)
    parent_dir = file.split("/")[-2]
    output = f"/juice4/scr4/nlp/crfm/markweb/ar5iv/no-problem/{paper}.json"
    if os.path.exists(output):
        print(f"{output} already exists")
        return
    with open(file, "r") as f:
        s = f.read()
    print(f"reading {file}")
    letters = re.search(r"^([A-Za-z])*",paper)
    numbers = re.search(r"[0-9\.]*$", paper)
    url = f"https://arxiv.org/abs/{letters.group(0)}/{numbers.group(0)}"
    print(url)
    title = "]".join(BeautifulSoup(urlopen(url), "html.parser").title.string.split("] ")[1:])
    print(title)
    match = re.search(r"""<section.*class="ltx_bibliography">""", s)
    if match:
        s = s[:match.start()]
    s = to_markdown(s)
    match = re.search(r"#*\s*Abstract\s*\n", s,re.IGNORECASE)
    if match:
        s = s[match.start():]
    else:
        match = re.search(r"#*.*Introduction\n",s, re.IGNORECASE)
        if match:
            s = s[match.start():]
    match = re.search(r"#*\s*(References|Bibliography|Bibliografia|Bibliographie)\s*\n",s,re.IGNORECASE)
    if match:
        s = s[:match.start()]
    match = re.search(r"Generated on .*by \[LaTeXML", s)
    if match:
        s = s[:match.start()]
    s = f"# {title}\n\n" + s
    if len(s) < 5000:
        print(f"{file} too small")
    print(f"saving {file}")
    output = {}
    output["metadata"] = {}
    output["content"] = s
    output["metadata"]["title"] = title
    output["metadata"]["url"] = url
    output["metadata"]["length"] = len(s)
    output["id"] = hash(s)
    output["source"] = "ar5iv"
    with open(output, "w") as f:
        json.dump(output, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, help="section out of 16 to process", default=0)
    args = parser.parse_args()
    files = glob.glob('/nlp/scr/kamyar/no-problem/*/*', recursive=True)
    length_of_section = len(files) // 16 + 1
    files = files[args.section * length_of_section: (args.section + 1) * length_of_section]
    from multiprocessing import Pool
    with Pool() as p:
        p.map(process_file, files)
