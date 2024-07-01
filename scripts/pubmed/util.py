"""
file: util.py
-------------
Modifications of pubmed_parser (https://pypi.org/project/pubmed-parser/)
functions to handle different formats (i.e., compressed, multiple XML files)

Original functions in pubmed_parser.pubmed_oa_parser.py
"""

import gzip
import tarfile
from lxml import etree
from itertools import chain


def read_xml(path, nxml=False):
    """
    Modified to handle different compression formats
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            tree = etree.parse(f)
    elif path.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            members = tar.getmembers()
            # Assuming there's only one XML file in the tar.gz
            xml_file = [m for m in members if m.name.endswith(".xml")][0]
            f = tar.extractfile(xml_file)
            tree = etree.parse(f)
    else:
        tree = etree.parse(path)

    if nxml:
        for elem in tree.getiterator():
            elem.tag = etree.QName(elem).localname
        etree.cleanup_namespaces(tree)

    return tree


def parse_pubmed_xml(path, include_path=False, nxml=False):
    """
    Given an input XML path to PubMed XML file, extract information and metadata
    from a given XML file and return parsed XML file in JSON format.
    You can check ``ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/`` to list of available files to download

    Parameters
    ----------
    path: str
        A path to a given PubMed XML file
    include_path: bool
        if True, include a key 'path_to_file' in an output dictionary
        default: False
    nxml: bool
        if True, this will strip a namespace of an XML after reading a file
        see https://stackoverflow.com/questions/18159221/remove-namespace-and-prefix-from-xml-in-python-using-lxml to
        default: False

    Return
    ------
    json_out: str
        A JSON string containing a list of dictionaries, each with keys from a parsed XML path
        'full_title', 'abstract', 'journal', 'pmid', 'pmc', 'doi',
        'publisher_id', 'author_list', 'affiliation_list', 'publication_year',
        'publication_date', 'subjects'
    """
    tree = read_xml(path, nxml)

    articles = tree.findall(".//article")
    list_of_dicts = []

    for article in articles:
        tree_title = article.find(".//title-group/article-title")
        if tree_title is not None:
            title = [t for t in tree_title.itertext()]
            sub_title = article.xpath(".//title-group/subtitle/text()")
            title.extend(sub_title)
            title = [t.replace("\n", " ").replace("\t", " ") for t in title]
            full_title = " ".join(title)
        else:
            full_title = ""

        try:
            abstracts = list()
            abstract_tree = article.findall(".//abstract")
            for a in abstract_tree:
                for t in a.itertext():
                    text = t.replace("\n", " ").replace("\t", " ").strip()
                    abstracts.append(text)
            abstract = " ".join(abstracts)
        except BaseException:
            abstract = ""

        journal_node = article.findall(".//journal-title")
        if journal_node is not None:
            journal = " ".join([j.text for j in journal_node])
        else:
            journal = ""

        dict_article_meta = parse_article_meta(article)
        pub_year_node = article.find(".//pub-date/year")
        pub_year = pub_year_node.text if pub_year_node is not None else ""
        pub_month_node = article.find(".//pub-date/month")
        pub_month = pub_month_node.text if pub_month_node is not None else "01"
        pub_day_node = article.find(".//pub-date/day")
        pub_day = pub_day_node.text if pub_day_node is not None else "01"

        subjects_node = article.findall(".//article-categories//subj-group/subject")
        subjects = list()
        if subjects_node is not None:
            for s in subjects_node:
                subject = " ".join([s_.strip() for s_ in s.itertext()]).strip()
                subjects.append(subject)
            subjects = "; ".join(subjects)
        else:
            subjects = ""

        # create affiliation dictionary
        affil_id = article.xpath(".//aff[@id]/@id")
        if len(affil_id) > 0:
            affil_id = list(map(str, affil_id))
        else:
            affil_id = [""]  # replace id with empty list

        affil_name = article.xpath(".//aff[@id]")
        affil_name_list = list()
        for e in affil_name:
            name = stringify_affiliation_rec(e)
            name = name.strip().replace("\n", " ")
            affil_name_list.append(name)
        affiliation_list = [[idx, name] for idx, name in zip(affil_id, affil_name_list)]

        tree_author = article.xpath('.//contrib-group/contrib[@contrib-type="author"]')
        author_list = list()
        for author in tree_author:
            author_aff = author.findall('xref[@ref-type="aff"]')
            try:
                ref_id_list = [str(a.attrib["rid"]) for a in author_aff]
            except BaseException:
                ref_id_list = ""
            try:
                author_list.append(
                    [
                        author.find("name/surname").text,
                        author.find("name/given-names").text,
                        ref_id_list,
                    ]
                )
            except BaseException:
                author_list.append(["", "", ref_id_list])
        author_list = flatten_zip_author(author_list)

        coi_statement = "\n".join(parse_coi_statements(article))

        dict_out = {
            "full_title": full_title.strip(),
            "abstract": abstract,
            "journal": journal,
            "pmid": dict_article_meta["pmid"],
            "pmc": dict_article_meta["pmc"],
            "doi": dict_article_meta["doi"],
            "publisher_id": dict_article_meta["publisher_id"],
            "author_list": author_list,
            "affiliation_list": affiliation_list,
            "publication_year": pub_year,
            "publication_date": "{}-{}-{}".format(pub_day, pub_month, pub_year),
            "subjects": subjects,
            "coi_statement": coi_statement,
        }
        if include_path:
            dict_out["path_to_file"] = path

        list_of_dicts.append(dict_out)

    return list_of_dicts


def parse_article_meta(tree):
    """Helper function to parse article meta information"""
    pmid = tree.find(".//article-id[@pub-id-type='pmid']")
    pmc = tree.find(".//article-id[@pub-id-type='pmc']")
    doi = tree.find(".//article-id[@pub-id-type='doi']")
    publisher_id = tree.find(".//article-id[@pub-id-type='publisher-id']")

    return {
        "pmid": pmid.text if pmid is not None else "",
        "pmc": pmc.text if pmc is not None else "",
        "doi": doi.text if doi is not None else "",
        "publisher_id": publisher_id.text if publisher_id is not None else "",
    }


def flatten_zip_author(author_list):
    """Helper function to format author list"""
    return [{"surname": a[0], "given_names": a[1], "affiliation_ids": a[2]} for a in author_list]


def stringify_affiliation_rec(aff_element):
    """Helper function to convert affiliation element to string"""
    return " ".join(aff_element.itertext())


def parse_coi_statements(tree):
    """Helper function to parse conflict of interest statements"""
    statements = tree.findall(".//fn[@fn-type='coi-statement']")
    return [s.text for s in statements if s is not None]


def read_xml(path, nxml=False):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            tree = etree.parse(f)
    elif path.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            members = tar.getmembers()
            # Assuming there's only one XML file in the tar.gz
            xml_file = [m for m in members if m.name.endswith(".xml")][0]
            f = tar.extractfile(xml_file)
            tree = etree.parse(f)
    else:
        tree = etree.parse(path)

    if nxml:
        for elem in tree.getiterator():
            elem.tag = etree.QName(elem).localname
        etree.cleanup_namespaces(tree)

    return tree


def parse_pubmed_paragraph(path, all_paragraph=False, nxml=False):
    """
    Give path to a given PubMed OA file, parse and return
    a list of dictionaries, each containing paragraph text and its metadata.

    Parameters
    ----------
    path: str
        A string to an XML path.
    all_paragraph: bool
        By default, this function will only append a paragraph if there is at least
        one reference made in a paragraph (to avoid noisy parsed text).
        A boolean indicating if you want to include paragraph with no references made or not.
        if True, include all paragraphs.
        if False, include only paragraphs that have references.
        default: False.
    nxml: bool
        if True, this will strip a namespace of an XML after reading a file.
        see https://stackoverflow.com/questions/18159221/remove-namespace-and-prefix-from-xml-in-python-using-lxml to
        default: False.

    Return
    ------
    json_out: str
        A JSON string containing a list of dictionaries, each with paragraph text and its metadata.
        Metadata includes 'pmc' of an article, 'pmid' of an article,
        'reference_ids' which is a list of reference ``rid`` made in a paragraph,
        'section' name of an article, and section 'text'.
    """
    tree = read_xml(path, nxml)

    # Parse metadata for each article
    articles = tree.xpath("//article")
    list_of_dicts = []

    for article in articles:
        article_meta = parse_article_meta(article)
        pmid = article_meta["pmid"]
        pmc = article_meta["pmc"]

        paragraphs = article.xpath(".//body//p")
        for paragraph in paragraphs:
            paragraph_text = stringify_children(paragraph)
            section = paragraph.find("../title")
            if section is not None:
                section = stringify_children(section).strip()
            else:
                section = ""

            ref_ids = [ref.attrib["rid"] for ref in paragraph.findall(".//*[@rid]")]

            dict_par = {
                "pmc": pmc,
                "pmid": pmid,
                "reference_ids": ref_ids,
                "section": section,
                "text": paragraph_text,
            }
            if len(ref_ids) >= 1 or all_paragraph:
                list_of_dicts.append(dict_par)

    return list_of_dicts


def parse_article_meta(tree):
    """Helper function to parse article meta information"""
    pmid = tree.find(".//article-id[@pub-id-type='pmid']")
    pmc = tree.find(".//article-id[@pub-id-type='pmc']")
    doi = tree.find(".//article-id[@pub-id-type='doi']")
    publisher_id = tree.find(".//article-id[@pub-id-type='publisher-id']")

    return {
        "pmid": pmid.text if pmid is not None else "",
        "pmc": pmc.text if pmc is not None else "",
        "doi": doi.text if doi is not None else "",
        "publisher_id": publisher_id.text if publisher_id is not None else "",
    }


def stringify_children(node):
    """Helper function to convert all children nodes to string"""
    parts = (
        [node.text]
        + list(
            chain(
                *(
                    [c.text, etree.tostring(c, with_tail=False).decode("utf-8"), c.tail]
                    for c in node.getchildren()
                )
            )
        )
        + [node.tail]
    )
    return "".join(filter(None, parts))
