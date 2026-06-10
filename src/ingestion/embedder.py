import logging
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MedicalEmbedder:

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        logger.info("Chargement du modèle d'embedding : %s", model_name)
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_embedding_dimension()
            logger.info(
                "Modèle chargé. Dimension des embeddings : %d", self.embedding_dim
            )
        except Exception as exc:
            logger.error("Erreur lors du chargement du modèle : %s", exc)
            raise RuntimeError(
                f"Impossible de charger le modèle {model_name}: {exc}"
            ) from exc

    def embed_documents(self, chunks: list[Document]) -> list[list[float]]:
        if not chunks:
            raise ValueError("La liste de chunks est vide.")
        logger.info("Génération des embeddings pour %d chunks...", len(chunks))
        texts = [chunk.page_content for chunk in chunks]
        try:
            embeddings = self.model.encode(
                texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
            )
            logger.info(
                "Embeddings générés : %d vecteurs de dimension %d.",
                len(embeddings),
                self.embedding_dim,
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.error("Erreur lors de la génération des embeddings : %s", exc)
            raise RuntimeError(f"Erreur d'embedding : {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("La requête ne peut pas être vide.")
        try:
            embedding = self.model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as exc:
            logger.error("Erreur lors de l'embedding de la requête : %s", exc)
            raise RuntimeError(f"Erreur d'embedding de requête : {exc}") from exc

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim


def run_embedding_stage(
    input_path: str = "data/raw/asclepius_documents.json",
    output_dir: str = "data/processed",
) -> str:
    """Stage DVC : charge les documents bruts, les découpe et les vectorise.

    Produit deux fichiers dans ``output_dir`` : les chunks avec leurs
    métadonnées (JSON) et la matrice d'embeddings (NumPy ``.npy``).
    """
    import json
    import os

    import numpy as np

    from src.ingestion.chunker import chunk_documents

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {input_path}. Exécutez d'abord le stage "
            "'ingest' (dvc repro ingest)."
        )
    with open(input_path, encoding="utf-8") as f:
        documents = json.load(f)
    chunks = chunk_documents(
        documents=documents,
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
    )
    embedder = MedicalEmbedder(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embeddings = embedder.embed_documents(chunks)
    os.makedirs(output_dir, exist_ok=True)
    chunks_path = os.path.join(output_dir, "chunks.json")
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
            f,
            ensure_ascii=False,
        )
    np.save(embeddings_path, np.asarray(embeddings, dtype="float32"))
    logger.info(
        "Stage embed terminé : %d chunks dans %s, embeddings dans %s.",
        len(chunks),
        chunks_path,
        embeddings_path,
    )
    return output_dir


if __name__ == "__main__":
    run_embedding_stage()
