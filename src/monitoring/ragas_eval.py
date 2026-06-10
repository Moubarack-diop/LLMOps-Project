import logging
import os
from typing import Any
from datasets import Dataset
from src.monitoring import _ragas_compat  # noqa: F401  (avant ragas)
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

logger = logging.getLogger(__name__)


class RAGASEvaluator:

    def __init__(
        self,
        metrics: list | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        if metrics is None:
            self.metrics = [faithfulness, answer_relevancy, context_precision]
        else:
            self.metrics = metrics
        self.llm_model = llm_model or os.getenv(
            "LLM_MODEL", "claude-haiku-4-5-20251001"
        )
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(
            "RAGASEvaluator initialisé avec les métriques : %s (juge LLM : %s)",
            [m.name for m in self.metrics],
            self.llm_model,
        )

    def _build_judges(self) -> tuple[Any, Any]:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY non définie : impossible d'utiliser "
                "Claude comme LLM juge pour RAGAS."
            )
        from langchain_anthropic import ChatAnthropic
        from langchain_huggingface import HuggingFaceEmbeddings

        judge_llm = ChatAnthropic(
            model=self.llm_model, temperature=0.0, max_tokens=1024, timeout=60
        )
        judge_embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        return (judge_llm, judge_embeddings)

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> dict[str, float]:
        if not len(questions) == len(answers) == len(contexts) == len(ground_truths):
            raise ValueError(
                "Les listes doivent avoir la même longueur. Reçu : "
                f"questions={len(questions)}, answers={len(answers)}, "
                f"contexts={len(contexts)}, "
                f"ground_truths={len(ground_truths)}"
            )
        if not questions:
            raise ValueError("La liste de questions est vide.")
        logger.info("Évaluation RAGAS sur %d exemples...", len(questions))
        try:
            eval_dataset = Dataset.from_dict(
                {
                    "question": questions,
                    "answer": answers,
                    "contexts": contexts,
                    "ground_truth": ground_truths,
                }
            )
            judge_llm, judge_embeddings = self._build_judges()
            result = evaluate(
                dataset=eval_dataset,
                metrics=self.metrics,
                llm=judge_llm,
                embeddings=judge_embeddings,
            )
            scores: dict[str, float] = {}
            result_df = result.to_pandas()
            for metric in self.metrics:
                metric_name = metric.name
                if metric_name in result_df.columns:
                    scores[metric_name] = float(result_df[metric_name].mean())
                else:
                    logger.warning("Métrique '%s' absente des résultats.", metric_name)
                    scores[metric_name] = 0.0
            logger.info(
                "Évaluation RAGAS terminée. Scores : %s",
                {k: f"{v:.4f}" for k, v in scores.items()},
            )
            return scores
        except Exception as exc:
            logger.error("Erreur lors de l'évaluation RAGAS : %s", exc)
            raise RuntimeError(f"Évaluation RAGAS échouée : {exc}") from exc

    def evaluate_single(
        self, question: str, answer: str, context: list[str], ground_truth: str
    ) -> dict[str, float]:
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[context],
            ground_truths=[ground_truth],
        )

    def get_metric_names(self) -> list[str]:
        return [m.name for m in self.metrics]


def build_evaluation_dataset(
    pipeline_results: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    for result in pipeline_results:
        questions.append(result.get("question", ""))
        answers.append(result.get("answer", ""))
        context_used = result.get("context_used", "")
        context_passages = [p.strip() for p in context_used.split("---") if p.strip()]
        contexts.append(context_passages if context_passages else [context_used])
        ground_truths.append(result.get("ground_truth", ""))
    return {
        "questions": questions,
        "answers": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
    }
