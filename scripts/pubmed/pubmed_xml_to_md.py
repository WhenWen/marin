"""
file: pubmed_xml_to_md.py
--------------------------
Converts PubMed XML files to Markdown format.
"""

import os
from collections import defaultdict
from tqdm import tqdm

from util import parse_pubmed_xml, parse_pubmed_paragraph


def xml_to_md(input_file, output_dir):
    meta = parse_pubmed_xml(input_file)
    paragraphs = parse_pubmed_paragraph(input_file)

    meta_dict = {xml["pmid"]: xml for xml in meta}
    # join paragraphs by pmid
    pmid_to_para_jsons = defaultdict(list)
    for para in paragraphs:
        print(para["pmid"])
        pmid_to_para_jsons[para["pmid"]].append(para)

    print("Number of unique pmmids: ", len(meta_dict))
    print("Number of unique pmids from paragraphs: ", len(pmid_to_para_jsons))

    for pmid in tqdm(pmid_to_para_jsons.keys()):
        assoc_meta = meta_dict[pmid]
        assoc_paragraphs = pmid_to_para_jsons[pmid]

        with open(os.path.join(output_dir, pmid + ".md"), "w") as f:
            f.write("# " + assoc_meta["full_title"] + "\n\n")
            f.write("## Abstract" + "\n\n" + assoc_meta["abstract"])

            # write sectin texts
            current_section_title = ""
            if not assoc_paragraphs:
                return None

            for paragraph in assoc_paragraphs:
                section_title = paragraph["section"]
                if section_title != current_section_title and section_title.strip():
                    f.write("\n\n## " + section_title)
                    current_section_title = section_title
                f.write("\n\n" + paragraph["text"].strip())


if __name__ == "__main__":
    DATA_DIRS = "scripts/pubmed/data"
    xml_to_md(
        input_file=os.path.join(DATA_DIRS, "europe_pmc", "PMC7240001_PMC7250000.xml.gz"),
        output_dir=os.path.join(DATA_DIRS, "processed"),
    )
