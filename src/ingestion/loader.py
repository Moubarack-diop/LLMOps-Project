import logging
import os
from typing import Any
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def load_asclepius_dataset(n_samples: int = 500) -> list[dict[str, Any]]:
    logger.info("Chargement du dataset Asclepius (n_samples=%d)...", n_samples)
    try:
        dataset = load_dataset(
            "starmpcc/Asclepius-Synthetic-Clinical-Notes",
            split="train",
        )
    except Exception as exc:
        logger.error("Erreur lors du chargement du dataset : %s", exc)
        raise RuntimeError(f"Impossible de charger le dataset : {exc}") from exc
    total_available = len(dataset)
    if n_samples > total_available:
        logger.warning(
            "n_samples (%d) dépasse la taille du dataset (%d). "
            "Chargement de tous les exemples disponibles.",
            n_samples,
            total_available,
        )
        n_samples = total_available
    subset = dataset.select(range(n_samples))
    logger.info(
        "Sélection de %d exemples sur %d disponibles.", n_samples, total_available
    )
    documents: list[dict[str, Any]] = []
    for idx, row in enumerate(subset):
        note_text = row.get("note", row.get("text", ""))
        question = row.get("question", "")
        answer = row.get("answer", row.get("output", ""))
        doc = {
            "note": note_text,
            "question": question,
            "answer": answer,
            "id": f"note_{idx:05d}",
        }
        documents.append(doc)
    logger.info(
        "Dataset chargé avec succès : %d documents prêts pour l'ingestion.",
        len(documents),
    )
    return documents


def save_documents_to_disk(
    documents: list[dict[str, Any]], output_dir: str = "data/raw"
) -> str:
    import json

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "asclepius_documents.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logger.info("Documents sauvegardés dans %s", output_path)
    return output_path


if __name__ == "__main__":
    docs = load_asclepius_dataset(n_samples=500)
    save_documents_to_disk(docs)
    logger.info("Ingestion terminée : %d documents sauvegardés.", len(docs))
