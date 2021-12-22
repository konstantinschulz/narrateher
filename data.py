import os.path
import re
import time

import bibtexparser
import nltk as nltk
import requests as requests
from bibtexparser.bibdatabase import BibDatabase
from bs4 import BeautifulSoup, Tag
from tika import parser
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, TensorType

from config import Config
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
# disable SSL verification because the genderopen.de server does not have a valid configuration
verify_ssl: bool = False


class FeministDataset(Dataset):
    def __init__(self) -> None:
        with open(Config.dataset_path) as f:
            self._len: int = sum([1 for x in f.readlines()])

    def __getitem__(self, idx: int) -> BatchEncoding:
        with open(Config.dataset_path) as f:
            for i, line in enumerate(f.readlines()):
                if i == idx:
                    encodings: BatchEncoding = Config.tokenizer()(
                        line, truncation=True, padding="max_length", max_length=Config.max_sequence_length,
                        return_tensors=TensorType.PYTORCH)
                    for k, v in encodings.items():
                        encodings[k] = v.squeeze().to(Config.device)
                    encodings["labels"] = encodings["input_ids"]
                    return encodings

    def __len__(self) -> int:
        return self._len


def aggregate_dataset() -> None:
    if not os.path.exists(Config.dataset_path):
        with open(Config.dataset_path, "a+") as f:
            for file in tqdm([x for x in os.listdir(Config.tok_dir) if x.endswith(".txt")]):
                with open(os.path.join(Config.tok_dir, file)) as f2:
                    content: str = f2.read()
                    f.write(content)


def check_cache(relevant_anchors: list[Tag]) -> list[Tag]:
    doi_dict: dict[int, set[int]] = {}
    for file in tqdm([os.path.join(Config.bibtex_dir, x) for x in os.listdir(Config.bibtex_dir) if x.endswith(".bib")]):
        with open(file) as f:
            bib_database: BibDatabase = bibtexparser.load(f)
            doi: str = bib_database.entries[0]["doi"]
            doi_parts: list[str] = doi.split("/")
            key: int = int(doi_parts[-2].split(".")[1])
            doi_dict[key] = doi_dict.get(key, set())
            doi_dict[key].add(int(doi_parts[-1]))
    ret_val: list[Tag] = []
    for ra in relevant_anchors:
        link: str = ra.get("href")
        link_parts: list[str] = link.split("/")
        key: int = int(link_parts[-2])
        # handles have an offset of 6 compared to the DOI ending
        val: int = int(link_parts[-1]) - 6
        if key in doi_dict and val not in doi_dict[key]:
            ret_val.append(ra)
    return ret_val


def crawl_genderopen():
    counter: int = 7
    while True:
        counter += 1
        process_page(counter)


def download_bibtex(anchors: list[Tag], base_url: str) -> str:
    for anchor in anchors:
        if anchor.string == "BibTex":
            url: str = f"{base_url}{anchor.get('href')}"
            response: requests.Response = get_response(url)
            # replace double commas to avoid confusion for the parser
            bibtex_string: str = response.text.replace(",,", ",")
            # replace whitespace in BibTex ID for better parsing
            first_linebreak_idx: int = bibtex_string.find("\n")
            # remove additional commas in the ID for better parsing
            first_line: str = bibtex_string[:first_linebreak_idx]
            first_comma_index: int = first_line.index(",")
            while first_comma_index < len(first_line) - 2:
                first_line = first_line[:first_comma_index] + "_" + first_line[first_comma_index + 1:]
                first_comma_index = first_line.index(",")
            first_line = first_line.replace(" ", "")
            bibtex_string = first_line + bibtex_string[first_linebreak_idx:]
            bib_database: BibDatabase = bibtexparser.loads(bibtex_string)
            bibtex_id: str = bib_database.entries[0]["ID"]
            # remove slashes in the ID for better parsing
            bibtex_id = bibtex_id.replace("/", "_")
            bib_database.entries[0]["ID"] = bibtex_id
            bibtex_path: str = os.path.join(Config.bibtex_dir, f"{bibtex_id}.bib")
            if not os.path.exists(bibtex_path):
                with open(bibtex_path, "w+") as f:
                    bibtexparser.dump(bib_database, f)
            return bibtex_id


def download_pdf(anchors: list[Tag], pdf_path: str, base_url: str):
    for anchor in anchors:
        if anchor.string == "Herunterladen":
            url: str = f"{base_url}{anchor.get('href')}"
            response: requests.Response = get_response(url)
            with open(pdf_path, "wb+") as f:
                f.write(response.content)


def extract_text_from_pdf():
    for file in tqdm([x for x in os.listdir(Config.pdf_dir) if x.endswith(".pdf")]):
        raw: dict = parser.from_file(os.path.join(Config.pdf_dir, file), service="text", xmlContent=True)
        content: str = raw['content'].replace("\n", " ").strip()
        start_idx: int = content.rfind("www.genderopen.de")
        content = content[start_idx + 17:]
        content = re.sub(r" +", " ", content)
        txt_file_name: str = file[:-4] + ".txt"
        with open(os.path.join(Config.txt_dir, txt_file_name), "w+") as f:
            f.write(content)


def get_response(url: str) -> requests.Response:
    time.sleep(0.5)
    return requests.get(url, verify=verify_ssl, timeout=60)


def get_soup(url: str) -> BeautifulSoup:
    response: requests.Response = get_response(url)
    return BeautifulSoup(response.text, 'html.parser')


def process_anchor(ra: Tag, base_url: str) -> None:
    url: str = base_url + ra.get("href")
    soup = get_soup(url)
    spans: list[Tag] = soup.find_all("span")
    language: str = ""
    for span in spans:
        if span.string == "Sprache":
            parent: Tag = span.parent
            next_sibling: Tag = parent.find_next_sibling("div")
            language = next_sibling.get_text().strip()
            break
    if language == "deutsch":
        anchors = soup.find_all("a")
        bibtex_id = download_bibtex(anchors, base_url)
        pdf_path: str = os.path.join(Config.pdf_dir, f"{bibtex_id}.pdf")
        if not os.path.exists(pdf_path):
            download_pdf(anchors, pdf_path, base_url)


def process_page(counter: int):
    base_url: str = "https://www.genderopen.de"
    crawling_base_url: str = f"{base_url}/discover?rpp=100&etal=0&query=feminismus&scope=&group_by=none&page="
    soup: BeautifulSoup = get_soup(f"{crawling_base_url}{counter}")
    anchors: list[Tag] = soup.find_all("a")
    relevant_anchors: list[Tag] = []
    class_string: str = "class"
    for anchor in anchors:
        parent: Tag = anchor.parent
        grandparent: Tag = parent.parent
        if grandparent.has_attr(class_string) and "item-result" in grandparent.attrs[class_string]:
            relevant_anchors.append(anchor)
    relevant_anchors = check_cache(relevant_anchors)
    for ra in tqdm(relevant_anchors):
        process_anchor(ra, base_url)


def tokenize():
    language: str = "german"
    for file in tqdm([x for x in os.listdir(Config.txt_dir) if x.endswith(".txt")]):
        with open(os.path.join(Config.txt_dir, file)) as f:
            content: str = f.read()
            with open(os.path.join(Config.tok_dir, file), "w+") as f2:
                new_content: str = ""
                sentences: list[str] = nltk.sent_tokenize(content, language=language)
                for sentence in sentences:
                    tokens: list[str] = nltk.word_tokenize(sentence, language=language)
                    new_content += " ".join(tokens) + "\n"
                f2.write(new_content)

# extract_text_from_pdf()
# tokenize()
# aggregate_dataset()
