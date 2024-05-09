from markweb import markdown
import tempfile
import os
import fsspec
import re
from markweb.markweb import convert_page

TEXT_START_MARKERS = (
    r"""\*\*\*[\s]*START.*\*\*\*""",
    r"\*END\*THE SMALL PRINT.*\n",
    r"This etext was prepared by.*\n",
    r"E-text prepared by.*\n",
    r"Produced b.*\ny",
    r"Distributed Proofreading Team.*\n",
    r"Proofreading Team at http://www.pgdp.net.*\n",
    r"http://gallica.bnf.fr.*\n",
    r"http://archive.org/details/.*\n",
    r"http://www.pgdp.net.*\n",
    r"\n.*Internet Archive.*\n",
    r"material from the Google Print project.*\n",
    r"\*END THE SMALL PRINT.*\n",
    r"This etext was produced by.*\n",
    r"The Project Gutenberg.*\n",
    r"http://gutenberg.spiegel.de/ erreichbar..*\n",
    r"Project Runeberg publishes.*\n",
    r"Beginning of this Project Gutenberg.*\n",
    r"Project Gutenberg Online Distributed.*\n",
    r"Gutenberg Online Distributed.*\n",
    r"the Project Gutenberg Online Distributed.*\n",
    r"Project Gutenberg TEI.*\n",
    r"This eBook was prepared by.*\n",
    r"http://gutenberg2000.de erreichbar.*\n",
    r"This Etext was prepared by.*\n",
    r"This Project Gutenberg Etext was prepared by.*\n",
    r"Gutenberg Distributed Proofreaders.*\n",
    r"Project Gutenberg Distributed Proofreaders.*\n",
    r"the Project Gutenberg Online Distributed Proofreading Team.*\n",
    r"\*\*The Project Gutenberg.*\n",
    r"\*SMALL PRINT!.*\n",
    r"More information about this book is at the top of this file..*\n",
    r"tells you about restrictions in how the file may be used..*\n",
    r"l'authorization à les utilizer pour preparer ce texte..*\n",
    r"of the etext through OCR..*\n",
    r"\*\*\*\*\*These eBooks Were Prepared By Thousands .*\n",
    r"We need your donations more than ever!.*\n",
    r"\*\*\*\*\s*SMALL PRINT!.*\n",
    r'\["Small Print" V..*\n',
    r'\(http://www.ibiblio.org/gutenberg/.*\n',
    r'and the Project Gutenberg Online Distributed Proofreading Team.*\n',
    r'Mary Meehan, and the Project Gutenberg Online Distributed Proofreading.*\n',
    r'this Project Gutenberg edition..*\n',
)


TEXT_END_MARKERS = (
    r"""\*\*\*[\s]*END.*\*\*\*""",
    r"End of\s*([Tt]he|this)? Project Gutenberg",
    r"Ende dieses Proje[ck]t Gutenberg",
    r"End of this is COPYRIGHTED",
    r"Ende dieses Etextes ",
    r"Ende diese(s)* Project Gutenber",
    r"\*\*This is a COPYRIGHTED Project Gutenberg Etext",
    r"Fin de Project Gutenberg",
    r"The Project Gutenberg Etext of ",
    r"Ce document fut pr[eé]sent[eé] en lecture",
    r"More information about this book is at the top of this file.",
    r"We need your donations more than ever!",
    r"END OF PROJECT GUTENBERG",
    r"End of the Project Gutenberg",
)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python process_gutenberg.py <book.html>', file=sys.stderr)
        sys.exit(1)
    domain = sys.argv[1]
    print(f"Processing domain {domain}", file=sys.stderr)
    if not os.path.exists(os.path.basename(domain)):
        fs = fsspec.get_fs_token_paths(domain)[0]
        fs.get_file(domain, os.path.basename(domain))
    
    with open(os.path.basename(domain), "r") as f:
        html = f.read()
    title = html.split("<title>")
    if len(title) == 1:
        title = "Unknown Title"
    else:
        title = title[1].split("</title>")[0].strip()
    for start_marker in TEXT_START_MARKERS:
        match = re.search(start_marker, html)
        if match:
            html = html[match.end():]
    for end_marker in TEXT_END_MARKERS:
        match = re.search(end_marker, html)
        if match:
            html = html[:match.start()]
    
    md = markdown.to_markdown(html)
    # print(title)
    # print(md)
    # out = {
    #         'metadata': {
    #             'source': 'project_gutenberg',
    #             'domain': domain,
    #             'title': title
    #         },
    #         'text': markdown
    #     }
    # out = convert_page(html, url=domain)
    # print(out["content"])