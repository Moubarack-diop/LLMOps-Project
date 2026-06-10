import logging
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[dict[str, Any]], chunk_size: int = 512, chunk_overlap: int = 50
) -> list[Document]:
    if not documents:
        raise ValueError("La liste de documents est vide.")
    logger.info(
        "Découpage de %d documents (chunk_size=%d, chunk_overlap=%d)...",
        len(documents),
        chunk_size,
        chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    all_chunks: list[Document] = []
    for doc in documents:
        note_text = doc.get("note", "")
        if not note_text.strip():
            logger.warning("Document %s a une note vide, ignoré.", doc.get("id"))
            continue
        metadata = {
            "note_id": doc.get("id", "unknown"),
            "question": doc.get("question", ""),
            "answer": doc.get("answer", ""),
            "source": f"asclepius/{doc.get('id', 'unknown')}",
        }
        chunks = splitter.create_documents(texts=[note_text], metadatas=[metadata])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        all_chunks.extend(chunks)
    logger.info(
        "Découpage terminé : %d chunks générés depuis %d documents.",
        len(all_chunks),
        len(documents),
    )
    return all_chunks


def get_chunk_stats(chunks: list[Document]) -> dict[str, Any]:
    if not chunks:
        return {"total": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
    lengths = [len(c.page_content) for c in chunks]
    return {
        "total": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }
