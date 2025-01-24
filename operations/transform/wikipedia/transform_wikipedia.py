"""
wikipedia/transform_wikipedia.py

Performs HTML->Text/MD conversion using the specified tools over a wiki dump save in DOLMA format.
"""

import json
import logging
import os
from dataclasses import dataclass
import re

import draccus
import fsspec
import ray
from bs4 import BeautifulSoup
from marin.core.runtime import cached_or_construct_output
from tqdm_loggable.auto import tqdm

from marin.schemas.web.convert import ExtractionConfig, HtmlToMarkdownConfig
from marin.utils import fsspec_glob
from marin.web.convert import convert_page

logger = logging.getLogger("ray")


@dataclass
class WikiExtractionConfig:
    input_path: str
    output_path: str
    revision: str
    extract_method: str
    extract_config: ExtractionConfig
    remove_reference_section: bool
    word_threshold: int
    digit_threshold: int


def remove_and_append_infobox(html: str) -> tuple[str, str | None]:
    """
    Wraps the infobox in a new section with heading 'Notes' and appends it to the end of the article.
    """
    soup = BeautifulSoup(html, "html.parser")

    infobox = soup.find("table", {"class": "infobox"})
    infobox_soup = None
    if infobox:
        # Remove the infobox from its current position
        infobox.extract()

        # Create new section with heading
        notes_section = soup.new_tag("div")
        heading = soup.new_tag("p")
        heading.string = "Notes:"
        notes_section.append(heading)
        
        # Add header row to infobox table
        header_row = soup.new_tag("tr")
        entry_header = soup.new_tag("th")
        entry_header.string = "Entry"
        desc_header = soup.new_tag("th") 
        desc_header.string = "Description"
        header_row.append(entry_header)
        header_row.append(desc_header)
        infobox.insert(0, header_row)
        
        notes_section.append(infobox)
        infobox_soup = notes_section

        # Find the body tag and append the new section
        # body = soup.find('body')
        # if body:
        #     body.append(notes_section)
        # else:
        #     soup.append(notes_section)

    return str(soup), str(infobox_soup) if infobox_soup else None


def remove_references_from_html(html: str) -> str:
    """
    Removes the references list and heading from the article.
    """
    soup = BeautifulSoup(html, "html.parser")

    reflist = soup.find("div", {"class": "reflist"})
    if reflist:
        reflist.extract()

    ref_heading = soup.find("h2", {"id": "References"})
    if ref_heading:
        ref_heading.extract()

    return str(soup)


def clean_wiki_html(html: str, remove_reference_section: bool = True) -> tuple[str, str | None]:
    """
    Cleans the HTML by removing unwanted elements.
    """

    html, infobox = remove_and_append_infobox(html)

    if remove_reference_section:
        html = remove_references_from_html(html)

    return str(html), infobox


def clean_wikitext(
    text: str,
    word_threshold: int = 30,
    digit_threshold: int = 50,
    remove_reference_tags: bool = False,
) -> str:
    """
    Cleans the wikitext by removing unwanted elements.
    """

    # Set to none if the text is empty
    if not text:
        return None
    
    # Check if majority of text is digits
    digit_count = sum(c.isdigit() for c in text)
    digit_percentage = (digit_count / len(text)) * 100
    if digit_percentage > digit_threshold:
        return None
    
    # Remove reference tags [x] from text
    if remove_reference_tags:
        text = re.sub(r"\[\d+\]", "", text)

    # Remove text if it contains less than 30 words
    if len(text.split()) < word_threshold:
        return None
    
    return text


@ray.remote(memory=2 * 1024 * 1024 * 1024)
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(
    input_file_path: str, 
    output_file_path: str, 
    extract_method: str, 
    extract_config: ExtractionConfig, 
    remove_reference_section: bool = True,
    word_threshold: int = 30,
    digit_threshold: int = 50,
) -> None:
    

    logger.info(f"Starting processing of file {input_file_path}")
    logger.info(f"Source: {input_file_path}")
    logger.info(f"Destination: {output_file_path}")
    try:
        with (
            fsspec.open(input_file_path, compression="gzip") as source,
            fsspec.open(output_file_path, "wt", compression="gzip") as output,
        ):
            for line in tqdm(source, desc="Processing lines"):
                row = json.loads(line)

                try:
                    content = None
                    if "html" not in row["article_body"].keys() and "wikitext" in row["article_body"].keys():
                        content = row["article_body"]["wikitext"]
                    elif "html" in row["article_body"]:
                        html_string = row["article_body"]["html"]
                        filtered_html, infobox = clean_wiki_html(html_string, remove_reference_section)
                        
                        content = convert_page(
                            filtered_html, extract_method=extract_method, config=extract_config
                        )["content"]

                        if infobox:
                            include_links = getattr(extract_config, "links", getattr(extract_config, "include_links", False))
                            infobox_extraction_config = HtmlToMarkdownConfig(
                                include_images=False,
                                include_links=include_links
                            )
                            infobox_content = convert_page(
                                infobox, extract_method="readability", config=infobox_extraction_config
                            )["content"]

                            content = f"{content}\n\n{infobox_content}" if infobox_content else content
                    else:
                        logger.error(f"No content found in the row: {row}")
                        continue
                    
                    content = clean_wikitext(
                        content,
                        word_threshold=word_threshold,
                        digit_threshold=digit_threshold,
                        remove_reference_tags=remove_reference_section
                    )
                    
                    if content:
                        out_dict = {
                            "id": row["identifier"],
                            "url": row["url"],
                            "title": row["name"],
                            "abstract": row.get("abstract", ""),
                            "date_created": row["date_created"] if "date_created" in row else row.get("date_modified", ""),
                            "text": content,
                        }

                        print(json.dumps(out_dict), file=output)  # Without this line, the JSON file will be corrupted

                except Exception as e:
                    logger.info(f"Keys in row: {row.keys()}")
                    logger.info(f"Article body keys: {row['article_body'].keys()}")

                    logger.exception(f"Error processing line: {e}")
                    continue

        

        logger.info("\nProcessing completed successfully!")
        logger.info(f"File available at: {output_file_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


@draccus.wrap()
def process_wiki_dump(cfg: WikiExtractionConfig) -> None:
    logger.info(f"Starting processing of Wikipedia dump in {cfg.input_path}")

    files = fsspec_glob(f"{cfg.input_path}/*.ndjson")
    logger.info(f"Found {len(files)} files to process")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 15

    for file in files:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        output_path = os.path.join(cfg.output_path, cfg.revision)
        output_file_path = os.path.join(output_path, file.split("/")[-1].replace(".ndjson", ".jsonl.gz"))
        result_refs.append(
            process_file.remote(
                file, output_file_path, cfg.extract_method, cfg.extract_config, cfg.remove_reference_section, cfg.word_threshold, cfg.digit_threshold
            )
        )
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")
