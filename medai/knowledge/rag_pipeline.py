"""RAG pipeline for guideline-grounded responses using ChromaDB."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

CHROMA_COLLECTION = "medai_guidelines"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


@dataclass
class RetrievedChunk:
    content: str
    source: str
    chunk_id: int
    relevance_score: float


class RAGPipeline:
    """
    RAG pipeline over clinical guidelines (NICE / WHO PDFs).
    Activates fully once PDFs are loaded via load_guidelines().
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._ready = False

        db_path = persist_dir or str(
            Path(__file__).parent.parent / "data" / "knowledge_base" / "chroma_db"
        )

        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=db_path,
                    anonymized_telemetry=False,
                )
            )
        except Exception:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=db_path)
            except Exception as exc:
                logger.warning("ChromaDB unavailable: %s", exc)
                return

        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL)

            import chromadb.utils.embedding_functions as ef
            self._embedding_fn = ef.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
        except Exception as exc:
            logger.warning("Embedding model unavailable: %s", exc)
            return

        try:
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                embedding_function=self._embedding_fn,
            )
            if self._collection.count() > 0:
                self._ready = True
                logger.info(
                    "RAG pipeline ready with %d chunks.", self._collection.count()
                )
        except Exception as exc:
            logger.warning("ChromaDB collection error: %s", exc)

    def load_guidelines(self, pdf_dir: str) -> int:
        """
        Load PDF files, chunk, embed and store.
        Returns number of chunks indexed.
        """
        if self._collection is None or self._embedding_fn is None:
            raise RuntimeError("ChromaDB or embedding model not available.")

        try:
            import pypdf
        except ImportError:
            try:
                import PyPDF2 as pypdf  # type: ignore
            except ImportError:
                raise ImportError("Install pypdf or PyPDF2 to load guidelines.")

        pdf_files = list(Path(pdf_dir).glob("**/*.pdf"))
        if not pdf_files:
            logger.warning("No PDFs found in %s", pdf_dir)
            return 0

        total_chunks = 0
        for pdf_path in pdf_files:
            chunks = self._extract_chunks(pdf_path, pypdf)
            if not chunks:
                continue

            ids = [f"{pdf_path.stem}_{i}" for i in range(total_chunks, total_chunks + len(chunks))]
            metadatas = [{"source": pdf_path.name, "chunk_id": i} for i in range(len(chunks))]

            self._collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
            )
            total_chunks += len(chunks)
            logger.info("Indexed %d chunks from %s", len(chunks), pdf_path.name)

        if total_chunks > 0:
            self._ready = True
        return total_chunks

    def _extract_chunks(self, pdf_path: Path, pypdf_module) -> List[str]:
        chunks: List[str] = []
        try:
            reader = pypdf_module.PdfReader(str(pdf_path))
            full_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as exc:
            logger.warning("Could not read %s: %s", pdf_path, exc)
            return []

        words = full_text.split()
        i = 0
        while i < len(words):
            chunk_words = words[i: i + CHUNK_SIZE]
            chunks.append(" ".join(chunk_words))
            i += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    def query(self, question: str, n_results: int = 5) -> List[RetrievedChunk]:
        if not self._ready or self._collection is None:
            return []

        try:
            results = self._collection.query(
                query_texts=[question],
                n_results=min(n_results, self._collection.count()),
            )
            chunks: List[RetrievedChunk] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                chunks.append(
                    RetrievedChunk(
                        content=doc,
                        source=meta.get("source", "unknown"),
                        chunk_id=meta.get("chunk_id", 0),
                        relevance_score=1.0 - float(dist),
                    )
                )
            return chunks
        except Exception as exc:
            logger.warning("RAG query error: %s", exc)
            return []

    def is_ready(self) -> bool:
        return self._ready
