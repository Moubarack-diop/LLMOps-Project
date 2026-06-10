"""Évaluation RAGAS du pipeline RAG, loggée dans MLflow.

Construit un jeu d'évaluation à partir des paires question/réponse de
référence du dataset Asclepius (stockées dans les payloads Qdrant), fait
répondre le pipeline RAG à chaque question, puis calcule les métriques
RAGAS (faithfulness, answer relevancy, context precision) avec Claude
comme LLM juge.

Usage : python -m src.monitoring.run_evaluation [n_samples]
"""

import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

from src.ingestion.embedder import MedicalEmbedder
from src.monitoring.mlflow_tracker import MLflowTracker
from src.monitoring.ragas_eval import RAGASEvaluator, build_evaluation_dataset
from src.retrieval.rag_pipeline import MedicalRAGPipeline
from src.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)


def fetch_eval_samples(store: QdrantStore, n_samples: int) -> list[dict[str, Any]]:
    """Récupère une question + réponse de référence par note distincte."""
    samples: list[dict[str, Any]] = []
    seen_notes: set[str] = set()
    offset = None
    while len(samples) < n_samples:
        points, offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=256,
            offset=offset,
            with_payload=["note_id", "question", "answer"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            note_id = payload.get("note_id")
            question = (payload.get("question") or "").strip()
            ground_truth = (payload.get("answer") or "").strip()
            if note_id and question and ground_truth and note_id not in seen_notes:
                seen_notes.add(note_id)
                samples.append(
                    {
                        "note_id": note_id,
                        "question": question,
                        "ground_truth": ground_truth,
                    }
                )
                if len(samples) >= n_samples:
                    break
        if offset is None:
            break
    return samples


def main() -> None:
    # Console Windows en cp1252 : MLflow imprime des emojis en fin de run,
    # ce qui lèverait UnicodeEncodeError sans ce passage en UTF-8.
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )
    # Trace tous les appels LangChain — y compris ceux du juge RAGAS : ses
    # verdicts justifiés (statement/verdict/reason) deviennent lisibles dans
    # l'onglet Traces de MLflow.
    if os.getenv("MEDASSIST_TRACING", "1") == "1":
        try:
            import mlflow

            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
            mlflow.set_experiment("MedAssist-RAG")
            mlflow.langchain.autolog()
            logger.info("Tracing MLflow activé pour l'évaluation.")
        except Exception as exc:
            logger.warning("Tracing MLflow non activé (non bloquant) : %s", exc)
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    top_k = int(os.getenv("TOP_K", "5"))
    llm_model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

    store = QdrantStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("QDRANT_COLLECTION", "medassist"),
    )
    samples = fetch_eval_samples(store, n_samples)
    if not samples:
        raise SystemExit(
            "Aucun échantillon d'évaluation trouvé dans Qdrant. "
            "Lancez d'abord une ingestion (/ingest)."
        )
    logger.info("Jeu d'évaluation : %d questions.", len(samples))

    embedder = MedicalEmbedder(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    pipeline = MedicalRAGPipeline(
        vector_store=store, embedder=embedder, llm_model=llm_model, top_k=top_k
    )

    results: list[dict[str, Any]] = []
    for i, sample in enumerate(samples, 1):
        logger.info(
            "[%d/%d] Question sur %s : %s",
            i,
            len(samples),
            sample["note_id"],
            sample["question"][:80],
        )
        result = pipeline.query(
            question=sample["question"], note_id=sample["note_id"]
        )
        result["ground_truth"] = sample["ground_truth"]
        results.append(result)

    data = build_evaluation_dataset(results)
    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate(
        questions=data["questions"],
        answers=data["answers"],
        contexts=data["contexts"],
        ground_truths=data["ground_truths"],
    )

    tracker = MLflowTracker(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    run_id = tracker.log_ragas_metrics(
        scores,
        params={
            "n_samples": len(samples),
            "top_k": top_k,
            "llm_model": llm_model,
            "judge_model": llm_model,
            "embedding_model": embedder.model_name,
        },
    )
    print("\n=== Scores RAGAS ===")
    for name, value in scores.items():
        print(f"{name:>20} : {value:.4f}")
    print(f"\nRun MLflow : {run_id}")


if __name__ == "__main__":
    main()
