# import tempfile
# from dataclasses import dataclass

# import json
# import os
# import xml.etree.ElementTree as ET
# from typing import Optional

# import fsspec
# import tqdm
# from fsspec.callbacks import TqdmCallback
from marin import markdown
import re
from bs4 import BeautifulSoup
import markdownify

ATX_CLOSED = 'atx_closed'
UNDERLINED = 'underlined'

class Ar5ivMarkdownConverter(markdown.MyMarkdownConverter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def convert_hn(self, n, el, text, convert_as_inline):
        if convert_as_inline:
            return text

        style = self.options['heading_style'].lower()
        text = text.strip()
        if style == UNDERLINED and n <= 2:
            line = '=' if n == 1 else '-'
            return self.underline(text, line)
        hashes = '#' * n
        if style == ATX_CLOSED:
            return '\n%s %s %s\n' % (hashes, text, hashes)
        return """\n%s %s <a name="%s"></a>\n""" % (hashes, text, el.parent.get('id')) if el.parent.get('id') else '\n%s %s\n' % (hashes, text)
    

    # def convert_figcaption(self, el, text, convert_as_inline):
    #     return f"""\n{text} <a name="{el.parent.get('id')}"></a>\n"""
    
    def convert_a(self, el, text, convert_as_inline):
        prefix, suffix, text = markdownify.chomp(text)
        if not text:
            return ''
        href = el.get('href')
        if "data" in href:
            return ""
        title = el.get('title')
        # For the replacement see #29: text nodes underscores are escaped
        if (self.options['autolinks']
                and text.replace(r'\_', '_') == href
                and not title
                and not self.options['default_title']):
            # Shortcut syntax
            return '<%s>' % href
        if self.options['default_title'] and not title:
            title = href
        title_part = ' "%s"' % title.replace('"', r'\"') if title else ''
        return '%s[%s](%s%s)%s' % (prefix, text, href, title_part, suffix) if href else text
    
    def convert_figure(self, el, text, convert_as_inline):
        if convert_as_inline:
            return text

        # find out if element has class ltx_algorithm
        if el.has_attr('class') and 'ltx_algorithm' in el['class']:
            return self.convert_pre(el, text, convert_as_inline)

        # the super doesn't handle this specifically. we basically want to be sure there's a newline
        if not text.endswith('\n\n'):
            if not text.endswith('\n'):
                text += '\n\n'
            else:
                text += '\n'
        return text
    
    def convert_td(self, el, text, convert_as_inline):
        colspan = 1
        if 'colspan' in el.attrs:
            colspan = int(el['colspan'])
        parent = el
        if convert_as_inline:
            return text + ' '
        return ' ' + text.strip().replace("\n", " ") + ' |' * colspan

    def convert_svg(self, el, text, convert_as_inline):
        return ""

    def convert_tr(self, el, text, convert_as_inline):
        if convert_as_inline:
            return text + "\n"

        if el.has_attr('class') and 'ltx_equation' in el['class'] or 'ltx_eqn_row' in el['class']:
            text = text.replace("|", " ")
            text = text.replace("<", " < ")
            text = text.replace(">", " > ")
            text = text.replace("\left ", " ")
            text = text.replace("\\right ", " ")
            return f"\n{text}\n"

        # this is also mostly copied from the parent class
        # but the logic for guessing a th isn't quite right
        cells = el.find_all(['td', 'th'])
        is_headrow = all([cell.name == 'th' for cell in cells])

        # rowspan check
        length_of_cells = len(cells)
        if el.previous_sibling:
            prev = el.previous_sibling
            count = 1
            while prev:
                if prev.name == 'tr':
                    prev_td = prev.findAll('td')
                    length_of_cells = max(length_of_cells, len(prev_td))
                prev = prev.previous_sibling
        rowspan = [0 for _ in range(length_of_cells)]
        if el.previous_sibling:
            prev = el.previous_sibling
            count = 1
            while prev:
                if prev.name == 'tr':
                    prev_td = prev.findAll('td')
                    row_span_exists = False
                    for i, td in enumerate(prev_td):
                        if 'rowspan' in td.attrs and int(td['rowspan']) > count:
                            rowspan[i] = 1
                prev = prev.previous_sibling
                count += 1
        # modify text for rowspan
        text = text.split('|')
        for i, row in enumerate(rowspan):
            if row:
                text.insert(i, '')
        text = '|'.join(text)

        # we can be a headrow if we are the first row in the table or if all our cells are th
        # find table parent
        if not is_headrow:
            parent = el.parent
            while parent and parent.name != 'table':
                parent = parent.parent

            if parent:
                first_row = parent.find('tr')
                if first_row is el:
                    is_headrow = True

        overline = ''
        underline = ''
        if is_headrow and not el.previous_sibling:
            # first row and is headline: print headline underline
            underline += '| ' + ' | '.join(['---'] * (text.count('|'))) + ' |' + '\n'
        elif (not el.previous_sibling
              and (el.parent.name == 'table'
                   or (el.parent.name == 'tbody'
                       and not el.parent.previous_sibling))):
            # first row, not headline, and:
            # - the parent is table or
            # - the parent is tbody at the beginning of a table.
            # print empty headline above this row
            overline += '| ' + ' | '.join([''] * len(cells)) + ' |' + '\n'
            overline += '| ' + ' | '.join(['---'] * len(cells)) + ' |' + '\n'
        # if not el.previous_sibling and el.parent:
        #     if el.parent.name == 'table':
        #         overline = f"""<a name="{el.parent.get('id')}"></a>\n""" + overline
        #     elif el.parent.parent.name == 'table' and not el.parent.parent.find('thead'):
        #         overline = f"""<a name="{el.parent.parent.get('id')}"></a>\n""" + overline
        return overline + '|' + text + '\n' + underline


def to_markdown(html):
    if isinstance(html, str):
        html = BeautifulSoup(html, "html.parser")
    authors = html.findAll('div', {'class': 'ltx_authors'})
    for author in authors:
        author.decompose()
    tags = html.findAll('span', {'class': 'ltx_tag_item'})
    for author in tags:
        author.decompose()
    tags = html.findAll('span', {'class': 'ltx_tag_listingline'})
    for author in tags:
        author.decompose()
    title = html.find('title')
    if title:
        real_title = html.select('.ltx_title_document')
        if real_title:
            real_title = real_title[0]
            title.string.replace_with(real_title)
        else:
            title.decompose()
    title_page = html.findAll('div', {'class': 'ltx_titlepage'})
    for tp in title_page:
        tp.decompose()
    biblio = html.findAll('section', {'id': 'bib'})
    for bib in biblio:
        bib.decompose()
    footnotes = html.findAll('div', {'class': 'ltx_role_footnote'})
    for fn in footnotes:
        fn.decompose()
    linelisting = html.findAll('div', {'class': 'ltx_listingline'})
    for fn in linelisting:
        fn.append(BeautifulSoup("<br>", "html.parser"))
    eqntables = html.findAll('table', {'class': 'ltx_eqn_table'})
    for eqn in eqntables:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()
    eqnrows = html.findAll('tr', {'class': 'ltx_eqn_row'})
    for eqn in eqnrows:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()
    eqncell = html.findAll('td', {'class': 'ltx_eqn_cell'})
    for eqn in eqncell:
        eqn.unwrap()
    footer = html.findAll('footer')
    for fn in footer:
        fn.decompose()
    biblinks = html.findAll('a', {'class': 'ltx_ref'})
    for biblink in biblinks:
        biblink.unwrap()
    section = html.find('section')
    if section:
        section = section.previous_sibling
        while section:
            new_section = section.previous_sibling
            section.extract()
            section = new_section
    text = markdown.MyMarkdownConverter().convert_soup(html)
    # cleanup: replace nbsp as space
    # this isn't quite right if we preserve html in places, but we currently are not doing that
    text = text.replace("\xa0", " ").strip()
    return text

import glob
import os
from urllib.request import urlopen
import json

def process_file(file):
    paper = os.path.basename(file)[:-5]
    print(paper)
    parent_dir = file.split("/")[-2]
    output_file = f"/Users/kamyarsalahi/Downloads/2107/{paper}.json"
    # if os.path.exists(output_file):
    #     print(f"{output_file} already exists")
    #     return
    with open(file, "r") as f:
        s = f.read()
    # old_html = s[:]
    # print(f"reading {file}")
    # letters = re.search(r"^([A-Za-z])*",paper)
    # numbers = re.search(r"[0-9\.]*$", paper)
    
    s = to_markdown(s)

    if len(s) < 5000:
        print(f"{file} too small")
    with open(f"/Users/kamyarsalahi/Downloads/markweb/scripts/ar5iv/{paper}.md", "w") as f:
        f.write(s)
    print(f"saving {file}")
    # output = {}
    # output["metadata"] = {}
    # output["content"] = [{"title": "HTML", "type": "html", "text": old_html}, {"title": "markdown", "type": "md", "text": s}]
    # output["metadata"]["title"] = title
    # output["metadata"]["url"] = url
    # output["metadata"]["length"] = len(s)
    # output["id"] = paper
    # output["source"] = "ar5iv"
    # with open("output.jsonl", "a") as f:
    #     f.write(json.dumps(output)+"\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, help="section out of 16 to process", default=0)
    args = parser.parse_args()
    # process_file("/Users/kamyarsalahi/Downloads/2107/2107.00042.html")
    files = glob.glob('/Users/kamyarsalahi/Downloads/2107/*.html', recursive=True)
    # print(files)
    # length_of_section = len(files) // 16 + 1
    # files = files[args.section * length_of_section: (args.section + 1) * length_of_section]
    from multiprocessing import Pool
    with Pool(8) as p:
        p.map(process_file, list(files))
