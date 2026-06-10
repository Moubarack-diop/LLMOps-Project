import logging
import os
from typing import Any
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowTracker:

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "MedAssist-RAG",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        logger.info("Connexion à MLflow sur %s...", tracking_uri)
        try:
            mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient(tracking_uri=tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow connecté. Expérience : '%s'", experiment_name)
        except Exception as exc:
            logger.error("Impossible de se connecter à MLflow : %s", exc)
            raise RuntimeError(f"Connexion MLflow échouée : {exc}") from exc

    def log_ingestion(self, n_docs: int, chunk_size: int, model_name: str) -> str:
        logger.info("Logging ingestion dans MLflow (n_docs=%d)...", n_docs)
        try:
            with mlflow.start_run(run_name="ingestion") as run:
                mlflow.set_tag("run_type", "ingestion")
                mlflow.log_params(
                    {
                        "n_documents": n_docs,
                        "chunk_size": chunk_size,
                        "embedding_model": model_name,
                        "dataset": "Asclepius-Synthetic-Clinical-Notes",
                    }
                )
                mlflow.log_metric("documents_ingested", n_docs)
                run_id = run.info.run_id
            logger.info("Ingestion loggée. Run ID : %s", run_id)
            return run_id
        except Exception as exc:
            logger.error("Erreur MLflow log_ingestion : %s", exc)
            raise RuntimeError(f"MLflow log_ingestion échoué : {exc}") from exc

    def log_ragas_metrics(
        self, metrics: dict[str, float], params: dict[str, Any] | None = None
    ) -> str:
        logger.info("Logging métriques RAGAS dans MLflow : %s", metrics)
        try:
            with mlflow.start_run(run_name="ragas_evaluation") as run:
                mlflow.set_tag("run_type", "evaluation")
                if params:
                    mlflow.log_params(params)
                for metric_name, score in metrics.items():
                    mlflow.log_metric(metric_name, score)
                run_id = run.info.run_id
            logger.info("Métriques RAGAS loggées. Run ID : %s", run_id)
            return run_id
        except Exception as exc:
            logger.error("Erreur MLflow log_ragas_metrics : %s", exc)
            raise RuntimeError(f"MLflow log_ragas_metrics échoué : {exc}") from exc

    def log_query(self, question: str, answer: str, sources: list[str]) -> str:
        try:
            with mlflow.start_run(run_name="query") as run:
                mlflow.set_tag("run_type", "query")
                mlflow.log_params(
                    {
                        "question_length": len(question),
                        "answer_length": len(answer),
                        "n_sources": len(sources),
                        "llm_model": os.getenv(
                            "LLM_MODEL", "claude-haiku-4-5-20251001"
                        ),
                    }
                )
                mlflow.log_text(question, "question.txt")
                mlflow.log_text(answer, "answer.txt")
                run_id = run.info.run_id
            return run_id
        except Exception as exc:
            logger.warning("MLflow log_query échoué (non bloquant) : %s", exc)
            return ""

    def get_best_run(self) -> dict[str, Any]:
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                logger.warning(
                    "Expérience '%s' introuvable dans MLflow.", self.experiment_name
                )
                return {}
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.run_type = 'evaluation'",
                order_by=["metrics.faithfulness DESC"],
                max_results=1,
            )
            if not runs:
                logger.info("Aucun run d'évaluation trouvé dans MLflow.")
                return {}
            best = runs[0]
            metrics = best.data.metrics
            return {
                "run_id": best.info.run_id,
                "faithfulness": metrics.get("faithfulness", 0.0),
                "answer_relevancy": metrics.get("answer_relevancy", 0.0),
                "context_precision": metrics.get("context_precision", 0.0),
                "experiment_name": self.experiment_name,
            }
        except Exception as exc:
            logger.error("Erreur lors de la récupération du meilleur run : %s", exc)
            return {}

    def is_healthy(self) -> bool:
        try:
            self.client.search_experiments()
            return True
        except Exception:
            return False
