import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md"}

def _candidate_data_dirs():
    return [
        BASE_DIR / "data",
        BASE_DIR / "documents",
        BASE_DIR,
        Path.cwd() / "data",
        Path.cwd() / "documents",
        Path.cwd(),
    ]

def _load_file(path):
    ext = path.suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()

    if ext in {".html", ".htm"}:
        return BSHTMLLoader(str(path), bs_kwargs={"features": "html.parser"}).load()

    if ext in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()

    return []

def load_documents(data_dir=None):
    docs = []
    searched_dirs = []

    candidate_dirs = [Path(data_dir)] if data_dir else _candidate_data_dirs()

    for data_path in candidate_dirs:
        if not data_path.is_absolute():
            data_path = BASE_DIR / data_path

        searched_dirs.append(str(data_path))

        if not data_path.is_dir():
            continue

        for root, _, files in os.walk(data_path):
            for file_name in sorted(files):
                path = Path(root) / file_name

                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    docs.extend(_load_file(path))

        if docs:
            return docs

    raise FileNotFoundError(
        "No supported documents found. Checked these folders: "
        + ", ".join(searched_dirs)
    )

def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
