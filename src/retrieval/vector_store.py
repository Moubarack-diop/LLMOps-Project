import logging
import uuid
from typing import Any
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantStore:

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "medassist",
    ) -> None:
        self.collection_name = collection_name
        logger.info("Connexion à Qdrant sur %s:%d...", host, port)
        try:
            self.client = QdrantClient(host=host, port=port, timeout=10)
            self.client.get_collections()
            logger.info("Connexion à Qdrant établie avec succès.")
        except Exception as exc:
            logger.error("Impossible de se connecter à Qdrant : %s", exc)
            raise ConnectionError(
                f"Connexion Qdrant échouée ({host}:{port}) : {exc}"
            ) from exc

    def create_collection(self, vector_size: int) -> None:
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection_name in existing:
                logger.info("Collection '%s' déjà existante.", self.collection_name)
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size, distance=qdrant_models.Distance.COSINE
                ),
            )
            logger.info(
                "Collection '%s' créée (dim=%d).", self.collection_name, vector_size
            )
        except UnexpectedResponse as exc:
            logger.error("Erreur Qdrant lors de la création de collection : %s", exc)
            raise RuntimeError(f"Création collection Qdrant échouée : {exc}") from exc

    def upsert_documents(
        self, chunks: list[Document], embeddings: list[list[float]]
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch : {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )
        logger.info("Insertion de %d documents dans Qdrant...", len(chunks))
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            payload = {
                "page_content": chunk.page_content,
                "note_id": chunk.metadata.get("note_id", "unknown"),
                "question": chunk.metadata.get("question", ""),
                "answer": chunk.metadata.get("answer", ""),
                "source": chunk.metadata.get("source", ""),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
            }
            points.append(
                qdrant_models.PointStruct(
                    id=point_id, vector=embedding, payload=payload
                )
            )
        try:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                logger.info(
                    "Batch %d/%d inséré (%d points).",
                    i // batch_size + 1,
                    (len(points) - 1) // batch_size + 1,
                    len(batch),
                )
            logger.info(
                "Insertion terminée : %d points dans la collection '%s'.",
                len(points),
                self.collection_name,
            )
            return len(points)
        except Exception as exc:
            logger.error("Erreur lors de l'insertion dans Qdrant : %s", exc)
            raise RuntimeError(f"Upsert Qdrant échoué : {exc}") from exc

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        note_id: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            query_filter = None
            if note_id:
                query_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="note_id",
                            match=qdrant_models.MatchValue(value=note_id),
                        )
                    ]
                )
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
            )
            documents = []
            for hit in response.points:
                doc = {
                    "page_content": hit.payload.get("page_content", ""),
                    "note_id": hit.payload.get("note_id", "unknown"),
                    "score": hit.score,
                    "question": hit.payload.get("question", ""),
                    "answer": hit.payload.get("answer", ""),
                    "source": hit.payload.get("source", ""),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                }
                documents.append(doc)
            logger.info("Recherche terminée : %d résultats trouvés.", len(documents))
            return documents
        except Exception as exc:
            logger.error("Erreur lors de la recherche Qdrant : %s", exc)
            raise RuntimeError(f"Recherche Qdrant échouée : {exc}") from exc

    def list_notes(self) -> list[str]:
        try:
            note_ids: set[str] = set()
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=["note_id"],
                    with_vectors=False,
                )
                for point in points:
                    nid = (point.payload or {}).get("note_id")
                    if nid:
                        note_ids.add(nid)
                if offset is None:
                    break
            return sorted(note_ids)
        except Exception as exc:
            logger.error("Erreur lors du listing des notes : %s", exc)
            raise RuntimeError(f"Listing des notes Qdrant échoué : {exc}") from exc

    def get_collection_info(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": info.status,
                "vector_size": info.config.params.vectors.size,
            }
        except Exception as exc:
            logger.warning("Impossible d'obtenir les infos de collection : %s", exc)
            return {"name": self.collection_name, "error": str(exc)}

    def is_healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
