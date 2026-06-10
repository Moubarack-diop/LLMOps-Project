import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("=== Démarrage de MedAssist API ===")
    logger.info(
        "Configuration : QDRANT=%s:%s | MLFLOW=%s | LLM=%s",
        os.getenv("QDRANT_HOST", "localhost"),
        os.getenv("QDRANT_PORT", "6333"),
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
    )
    try:
        from src.retrieval.vector_store import QdrantStore

        store = QdrantStore(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "medassist"),
        )
        if store.is_healthy():
            logger.info("Qdrant : connexion OK")
        else:
            logger.warning("Qdrant : connexion DEGRADÉE")
    except Exception as exc:
        logger.warning("Qdrant non disponible au démarrage : %s", exc)
    try:
        import httpx

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{mlflow_uri}/health")
            if resp.status_code == 200:
                logger.info("MLflow : connexion OK")
            else:
                logger.warning("MLflow : statut HTTP %d", resp.status_code)
    except Exception as exc:
        logger.warning("MLflow non disponible au démarrage : %s", exc)
    # Tracing GenAI : chaque requête RAG produit une trace LangChain complète
    # (prompt, contexte, réponse, latences) dans l'onglet Traces de MLflow.
    if os.getenv("MEDASSIST_TRACING", "1") == "1":
        try:
            import mlflow

            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
            mlflow.set_experiment("MedAssist-RAG")
            mlflow.langchain.autolog()
            logger.info("MLflow : tracing LangChain activé.")
        except Exception as exc:
            logger.warning("Tracing MLflow non activé (non bloquant) : %s", exc)
    llm_model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Anthropic : clé API détectée (modèle '%s')", llm_model)
    else:
        logger.warning(
            "Anthropic : ANTHROPIC_API_KEY absente. Les requêtes RAG "
            "échoueront. Renseignez votre clé dans le fichier .env."
        )
    # Pré-charge le modèle d'embedding et la connexion Qdrant pour que la
    # première requête ne paie pas le coût d'initialisation (désactivable,
    # notamment pour les tests).
    if os.getenv("MEDASSIST_EAGER_INIT", "0") == "1":
        from src.api.routes import warmup_resources

        warmup_resources()
    logger.info("=== MedAssist API prête ===")
    yield
    logger.info("=== Arrêt de MedAssist API ===")


app = FastAPI(
    title="MedAssist — Assistant Médical Intelligent",
    description=(
        "Système RAG (Retrieval-Augmented Generation) permettant à un "
        "clinicien d'interroger des dossiers patients en langage naturel."
        "\n\n**Dataset** : Asclepius Synthetic Clinical Notes (HuggingFace)"
        "\n\n**LLM** : Claude via Anthropic (claude-haiku-4-5)"
        "\n\n**Embeddings** : sentence-transformers/all-MiniLM-L6-v2"
        "\n\n**Base vectorielle** : Qdrant"
        "\n\n*Projet académique MLOps — Auteur : Mouhamed Diop | "
        "Encadrant : Mously DIAW*"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
# Origines autorisées, séparées par des virgules (ex: "http://localhost:8501").
# "*" par défaut pour l'usage local ; à restreindre en déploiement.
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # Le combo credentials + wildcard est interdit par la spec CORS.
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="")
logger.info("Application MedAssist configurée avec succès.")
