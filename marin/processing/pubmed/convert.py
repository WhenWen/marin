import re

from lxml import etree

number_bracket = re.compile("( \[[0-9][0-9,\s-]*\])")
author_list = re.compile("( \([A-Z][A-Za-z;\,\.\-\s]+ [12][0-9]{3}\))")


def clean_text(text):
    new_text = re.sub(number_bracket, "", text)  # remove number brackets
    new_text = re.sub(author_list, "", new_text)  # remove author lists
    return new_text


def process_section(element, processed_set, level=2) -> list[str]:
    section = {
        "title": "",
        "body": [],
    }

    if element.tag == "sec":
        title = element.find("title")
        if title is not None:
            title_text = "".join(title.itertext())
            section["title"] = "#" * level + " " + title_text
        for child in element:
            if child.tag == "sec":
                section["body"].extend(process_section(child, processed_set, level + 1))
            elif child.tag == "p":
                paragraph_text = "".join(child.itertext()).strip()
                # dedup
                if paragraph_text in processed_set:
                    continue
                section["body"].append(paragraph_text)
                processed_set.add(paragraph_text)

    if not section["body"]:
        return []

    return [section["title"]] + section["body"]


def xml2md(xml_str):
    # Set up return text
    text = []
    # Parse the XML file
    root = etree.fromstring(xml_str)
    # Get the title
    title = root.xpath("//article-title")
    if len(title) > 0:
        title_text = "".join(title[0].itertext())
        text.append("# " + title_text)
    # Find all titles and paragraphs
    titles_and_paragraphs = root.xpath(
        "//title | //p[not(ancestor::table-wrap)] | //abstract | //sec"
    )
    processed_set = set()

    # Iterate through titles and paragraphs interleaved
    for element in titles_and_paragraphs:

        if element.tag == "title" and element.getparent().tag != "sec":
            title_text = "".join(element.itertext())
            text.append("## " + title_text)

        elif element.tag == "p":
            paragraph_text = "".join(element.itertext())
            # Remove open access attribution statement
            if "open access" in paragraph_text.lower():
                continue
            if paragraph_text in processed_set:
                continue

            text.append(paragraph_text)
            processed_set.add(paragraph_text)

        elif element.tag == "abstract":
            text.append("## Abstract")
            for child in element:
                text.extend(process_section(child, processed_set, level=3))

        elif element.tag == "sec":
            text.extend(process_section(element, processed_set))

    text = "\n\n".join(text)
    cleaned_text = clean_text(text)

    return cleaned_text
