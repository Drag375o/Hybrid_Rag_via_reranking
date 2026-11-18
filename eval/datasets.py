"""Download and read BEIR benchmark datasets (https://github.com/beir-cellar/beir).

A BEIR dataset is a zip with three parts:
  corpus.jsonl      the documents to search
  queries.jsonl     the questions
  qrels/<split>.tsv which documents are relevant to which question (the answer key)

Files are read straight out of the zip; nothing is extracted to disk.
"""

import csv
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict[str, str]]  # doc id -> {"title": ..., "text": ...}
    queries: dict[str, str]  # query id -> question (only questions that have judgements)
    qrels: dict[str, dict[str, int]]  # query id -> {relevant doc id: relevance grade}


def download(name: str, dest_dir: Path, retries: int = 3) -> Path:
    """Download ``<name>.zip`` into dest_dir once; later calls reuse the cached file."""
    path = dest_dir / f"{name}.zip"
    if path.exists():
        return path
    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".zip.part")
    url = BEIR_URL.format(name=name)
    error: Exception | None = None
    for _ in range(retries):
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(path)  # only a complete download gets the real name
            return path
        except OSError as exc:
            error = exc
    raise RuntimeError(f"Could not download {url} after {retries} attempts: {error}")


def _read_lines(zf: zipfile.ZipFile, member: str):
    with zf.open(member) as raw:
        yield from io.TextIOWrapper(raw, encoding="utf-8")


def read_beir_zip(path: Path, name: str, split: str = "test") -> BeirDataset:
    with zipfile.ZipFile(path) as zf:
        corpus = {}
        for line in _read_lines(zf, f"{name}/corpus.jsonl"):
            doc = json.loads(line)
            corpus[str(doc["_id"])] = {"title": doc.get("title", ""), "text": doc.get("text", "")}

        qrels: dict[str, dict[str, int]] = {}
        for row in csv.DictReader(_read_lines(zf, f"{name}/qrels/{split}.tsv"), delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

        queries = {}
        for line in _read_lines(zf, f"{name}/queries.jsonl"):
            query = json.loads(line)
            if str(query["_id"]) in qrels:
                queries[str(query["_id"])] = query["text"]

    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def load_beir(name: str, cache_dir: Path, split: str = "test") -> BeirDataset:
    return read_beir_zip(download(name, cache_dir), name, split)
