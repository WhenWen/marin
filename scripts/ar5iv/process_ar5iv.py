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
    
    def convert_sub(self, el, text, convert_as_inline):
        if not text:
            return ""
        return f"_{{{text}}}"

    def convert_sup(self, el, text, convert_as_inline):
        if not text:
            return ""
        return f"^{{{text}}}"

    def convert_br(self, el, text, convert_as_inline):
        if convert_as_inline:
            return "<br>"

        if self.options['newline_style'].lower() == markdownify.BACKSLASH:
            return '\\\n'
        else:
            return '  \n'


def to_markdown(html):
    if isinstance(html, str):
        html = BeautifulSoup(html, "html.parser")
    text = Ar5ivMarkdownConverter().convert_soup(html)
    # cleanup: replace nbsp as space
    # this isn't quite right if we preserve html in places, but we currently are not doing that
    text = text.replace("\xa0", " ")
    return text, html.find('title').text

with open("/juice4/scr4/nlp/crfm/markweb/ar5iv/error/0003/nlin0003055.html", "r") as f:
    s = f.read()
    s = s.split("</head>")[1]
    match = re.search(r"""<h(.)*>References""", s)
    s = s[:match.start()]
    print(to_markdown(s))
