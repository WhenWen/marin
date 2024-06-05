"""
file: pdf2md.py
----------------
Uses the Marker tool to convert PDFs to Markdown.
For more info about Marker, see https://github.com/VikParuchuri/marker
"""

import os
import sys
import subprocess
import argparse
import re
from tqdm import tqdm

DEBUG = 0


def pdf2md_conversion(pdf_filepaths, args):
    dir_pairs = set()  # (pdf_dir, md_dir) pairs
    for pdf_file in pdf_filepaths:  # TODO: parallelizable
        basename = os.path.basename(pdf_file)[:-4]  # remove .pdf extension
        pdf_dir, _ = os.path.split(pdf_file)
        rel_path = os.path.relpath(pdf_dir, args.pdf_root_dir)
        md_dir = os.path.join(args.md_root_dir, rel_path)
        os.makedirs(os.path.join(md_dir, basename), exist_ok=True)
        dir_pairs.add((pdf_dir, md_dir))
        if DEBUG:
            print(f"basename: {basename}")
            print(f"pdf_dir: {pdf_dir}")
            print(f"rel_path: {rel_path}")
            print(f"md_dir: {md_dir}")

        if args.single_file:
            subprocess.run(
                [
                    "marker_single",
                    pdf_file,
                    md_dir,
                    "--batch_multiplier",
                    "2",
                ]
            )

        # copy pdf file to new dir
        new_pdf_file = os.path.join(md_dir, basename, f"{basename}.pdf")
        subprocess.run(["cp", pdf_file, new_pdf_file])

    # convert pdf to markdown
    if not args.single_file:
        for pdf_dir, md_dir in tqdm(dir_pairs):
            if DEBUG:
                print(f"Converting {pdf_dir} to {md_dir}...")
            subprocess.run(
                [
                    "marker",
                    pdf_dir,
                    md_dir,
                    "--workers",
                    str(args.num_workers),
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # FILEPATH ARGS
    parser.add_argument(
        "--pdf_root_dir",
        help="Directory containing PDF files to convert to Markdown",
        default="data/pdfs/",
    )
    parser.add_argument(
        "--md_root_dir",
        help="Directory to save Markdown files",
        default="data/md/",
    )
    # MARKER ARGS
    parser.add_argument(
        "--num_workers", type=int, default=5, help="Number of worker processes to use"
    )
    # MISC ARGS
    parser.add_argument(
        "--single_file", action="store_true", help="single file conversion"
    )  # my CPU has a memory leak with batch conversion
    # TODO: consider whether we need to overwrite settings with env variables
    # (esp on GPU)
    args = parser.parse_args()

    pdf_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(args.pdf_root_dir)
        for f in filenames
        if f.endswith(".pdf")
        and not os.path.exists(
            os.path.join(
                args.md_root_dir,
                os.path.relpath(dp, args.pdf_root_dir),
                f"{os.path.basename(f)[:-4]}/{os.path.basename(f)[:-4]}.md",
            )
        )
    ]
    if len(pdf_files) == 0:
        print(f"No PDF files found in {args.pdf_root_dir}")
        sys.exit(1)

    pdf2md_conversion(pdf_files, args)
