from markweb import markdown
import tempfile
import os
import fsspec
import re



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
    match_pre = re.search(r"""</pre>""", html)
    if match_pre:
        html = html[match_pre.end():]
    match_start = re.search(r"""\*\*\* START.*\*\*\*""", html)
    if match_start:
        html = html[match_start.end():]
    match_end = re.search(r"""\*\*\* END.*\*\*\*""", html)
    if match_end:
        html = html[:match_end.start()]
    
    markdown = markdown.to_markdown(html)
    print(title)
    print(markdown)
    out = {
            'metadata': {
                'source': 'project_gutenberg',
                'domain': domain,
                'title': title
            },
            'text': markdown
        }