import logging
import os
import httpx
from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    NotesResponse,
    QueryRequest,
    QueryResponse,
)
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import MedicalEmbedder
from src.ingestion.loader import load_asclepius_dataset
from src.monitoring.mlflow_tracker import MLflowTracker
from src.retrieval.rag_pipeline import MedicalRAGPipeline
from src.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_qdrant_store() -> QdrantStore:
    return QdrantStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("QDRANT_COLLECTION", "medassist"),
    )


def _get_embedder() -> MedicalEmbedder:
    return MedicalEmbedder(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )


def _get_mlflow_tracker() -> MLflowTracker:
    return MLflowTracker(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )


@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    qdrant_status = "ok"
    mlflow_status = "ok"
    llm_status = "ok"
    try:
        store = _get_qdrant_store()
        if not store.is_healthy():
            qdrant_status = "error"
    except Exception as exc:
        logger.warning("Qdrant non disponible : %s", exc)
        qdrant_status = "error"
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{mlflow_uri}/health")
            if resp.status_code != 200:
                mlflow_status = "error"
    except Exception as exc:
        logger.warning("MLflow non disponible : %s", exc)
        mlflow_status = "error"
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY absente : le LLM Claude est indisponible.")
        llm_status = "error"
    all_ok = qdrant_status == "ok" and mlflow_status == "ok" and (llm_status == "ok")
    overall = "ok" if all_ok else "degraded"
    return HealthResponse(
        status=overall, qdrant=qdrant_status, mlflow=mlflow_status, llm=llm_status
    )


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_dataset(request: IngestRequest) -> IngestResponse:
    logger.info(
        "Démarrage de l'ingestion : n_samples=%d, chunk_size=%d",
        request.n_samples,
        request.chunk_size,
    )
    try:
        documents = load_asclepius_dataset(n_samples=request.n_samples)
        chunks = chunk_documents(
            documents=documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        embedder = _get_embedder()
        embeddings = embedder.embed_documents(chunks)
        store = _get_qdrant_store()
        store.create_collection(vector_size=embedder.get_embedding_dimension())
        n_inserted = store.upsert_documents(chunks=chunks, embeddings=embeddings)
        try:
            tracker = _get_mlflow_tracker()
            tracker.log_ingestion(
                n_docs=len(documents),
                chunk_size=request.chunk_size,
                model_name=os.getenv(
                    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                ),
            )
        except Exception as mlflow_exc:
            logger.warning("MLflow logging échoué (non bloquant) : %s", mlflow_exc)
        return IngestResponse(
            status="success",
            documents_ingested=len(documents),
            chunks_created=n_inserted,
            message=f"Ingestion terminée : {len(documents)} documents, {n_inserted} chunks.",
        )
    except Exception as exc:
        logger.error("Erreur lors de l'ingestion : %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Ingestion échouée : {exc}"
        ) from exc


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest) -> QueryResponse:
    logger.info("Requête RAG reçue : '%s'", request.question[:80])
    try:
        store = _get_qdrant_store()
        embedder = _get_embedder()
        pipeline = MedicalRAGPipeline(
            vector_store=store,
            embedder=embedder,
            llm_model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
            top_k=request.top_k,
        )
        result = pipeline.query(question=request.question, note_id=request.note_id)
        try:
            tracker = _get_mlflow_tracker()
            tracker.log_query(
                question=request.question,
                answer=result["answer"],
                sources=result["source_documents"],
            )
        except Exception as mlflow_exc:
            logger.warning("MLflow logging échoué (non bloquant) : %s", mlflow_exc)
        return QueryResponse(
            answer=result["answer"],
            sources=result["source_documents"],
            question=result["question"],
        )
    except Exception as exc:
        logger.error("Erreur pipeline RAG : %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Erreur pipeline RAG : {exc}"
        ) from exc


@router.get("/notes", response_model=NotesResponse, tags=["RAG"])
async def list_notes() -> NotesResponse:
    try:
        store = _get_qdrant_store()
        notes = store.list_notes()
        return NotesResponse(count=len(notes), notes=notes)
    except Exception as exc:
        logger.error("Erreur listing des notes : %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Erreur listing des notes : {exc}"
        ) from exc


@router.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics() -> MetricsResponse:
    try:
        tracker = _get_mlflow_tracker()
        best_run = tracker.get_best_run()
        if not best_run:
            raise HTTPException(
                status_code=404,
                detail="Aucune métrique RAGAS disponible. Lancez d'abord une évaluation.",
            )
        return MetricsResponse(
            run_id=best_run.get("run_id", ""),
            faithfulness=best_run.get("faithfulness", 0.0),
            answer_relevancy=best_run.get("answer_relevancy", 0.0),
            context_precision=best_run.get("context_precision", 0.0),
            experiment_name=best_run.get("experiment_name", "MedAssist-RAG"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Erreur récupération métriques : %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Erreur récupération métriques : {exc}"
        ) from exc
