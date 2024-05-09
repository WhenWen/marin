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


with open("/juice4/scr4/nlp/crfm/markweb/ar5iv/error/0003/nlin0003055.html", "r") as f:
    s = f.read()
    s = s.split("</head>")[1]
    match = re.search(r"""<h(.)*>References""", s)
    s = s[:match.start()]
    print(markdown.to_markdown(s))
