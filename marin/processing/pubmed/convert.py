import re
import sys

from lxml import etree

# cleaning_regexes
number_bracket = re.compile("( \[[0-9][0-9,\s-]*\])")
author_list = re.compile("( \([A-Z][A-Za-z;\,\.\-\s]+ [12][0-9]{3}\))")

def clean_text(text):
    new_text = re.sub(number_bracket, "", text)
    new_text = re.sub(author_list, "", new_text)
    return new_text

def xml2md(xml_str):
    # Set up return text
    text = ""
    # Parse the XML file
    root = etree.fromstring(xml_str)
    #root = tree.getroot()
    # Get the title
    title = root.xpath('//article-title')
    if len(title) > 0:
        title_text = ''.join(title[0].itertext())
        text += ("# " + title_text + "\n")
    # Find all titles and paragraphs    
    titles_and_paragraphs = root.xpath('//title | //p[not(ancestor::table-wrap)] | //abstract')
    # Iterate through titles and paragraphs interleaved
    for element in titles_and_paragraphs:
        if element.tag == 'title':
            title_text = ''.join(element.itertext())
            text += ('\n\n## ' + title_text)
        elif element.tag == 'p':
            paragraph_text = ''.join(element.itertext())
            text += ('\n\n' + paragraph_text)
        elif element.tag == "abstract":
            text += ('\n\n' + "## Abstract")
    # Standardize newlines
    text = re.sub("[\n]{2,}", "\n\n", text)
    # Clean text of references and author list
    text = clean_text(text)
    return text

