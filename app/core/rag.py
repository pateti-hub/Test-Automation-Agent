from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class RAGStore:
    def __init__(self, embedding_model: str) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
        )
        self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self._vectorstore: FAISS | None = None

    def ingest(self, text: str, source: str) -> int:
        docs = [Document(page_content=text, metadata={"source": source})]
        return self.ingest_documents(docs)

    def ingest_documents(self, docs: list[Document]) -> int:
        chunks = self._splitter.split_documents(docs)
        if not chunks:
            return 0
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(chunks, self._embeddings)
        else:
            self._vectorstore.add_documents(chunks)
        return len(chunks)

    def query(self, question: str, k: int = 4) -> list[Document]:
        if self._vectorstore is None:
            return []
        return self._vectorstore.similarity_search(question, k=k)
