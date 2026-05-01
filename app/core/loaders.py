from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}


def load_documents_from_path(file_path: Path, source: str) -> list[Document]:
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        docs = PyPDFLoader(str(file_path)).load()
    elif extension == ".docx":
        docs = Docx2txtLoader(str(file_path)).load()
    elif extension == ".txt":
        docs = TextLoader(str(file_path), encoding="utf-8").load()
    elif extension in {".csv", ".xlsx"}:
        docs = _load_tabular(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    for doc in docs:
        doc.metadata["source"] = source
    return docs


def _load_tabular(file_path: Path) -> list[Document]:
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    content = df.to_csv(index=False)
    return [Document(page_content=content, metadata={"source": file_path.name})]
