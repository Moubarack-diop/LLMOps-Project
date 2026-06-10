import logging
import os
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from src.ingestion.embedder import MedicalEmbedder
from src.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)
MEDICAL_SYSTEM_PROMPT = "Tu es MedAssist, un assistant médical intelligent spécialisé dans l'analyse de dossiers patients.\n\nRÈGLES ABSOLUES :\n1. Tu réponds UNIQUEMENT à partir du contexte de notes cliniques fourni ci-dessous.\n2. Si l'information demandée n'est pas présente dans le contexte, tu dois répondre exactement : \"Je ne trouve pas cette information dans les dossiers patients disponibles.\"\n3. Tu ne fais JAMAIS d'hypothèses médicales au-delà du contexte fourni.\n4. Tu cites toujours les identifiants des notes sources utilisées (note_id).\n5. Tu utilises un langage clair, précis et professionnel adapté aux cliniciens.\n\nCONTEXTE DES NOTES CLINIQUES :\n{context}\n\nRéponds maintenant à la question du clinicien en citant les sources pertinentes."
HUMAN_PROMPT = "Question : {question}"


class MedicalRAGPipeline:

    def __init__(
        self,
        vector_store: QdrantStore,
        embedder: MedicalEmbedder,
        llm_model: str = "claude-haiku-4-5-20251001",
        top_k: int = 5,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.llm_model = llm_model
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY non définie. Renseignez votre clé API Anthropic dans le fichier .env ou les variables d'environnement."
            )
        logger.info("Initialisation du LLM Anthropic (%s)...", llm_model)
        try:
            self.llm = ChatAnthropic(
                model=llm_model, temperature=0.1, max_tokens=1024, timeout=60
            )
        except Exception as exc:
            logger.error("Erreur d'initialisation du LLM Anthropic : %s", exc)
            raise RuntimeError(
                f"LLM Anthropic non initialisé ({llm_model}) : {exc}"
            ) from exc
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", MEDICAL_SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )
        self.chain = self.prompt | self.llm
        logger.info("Pipeline RAG initialisé avec succès (modèle : %s).", llm_model)

    def query(self, question: str, note_id: str | None = None) -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")
        logger.info("Traitement de la requête : '%s'", question[:100])
        try:
            query_embedding = self.embedder.embed_query(question)
        except Exception as exc:
            logger.error("Erreur embedding requête : %s", exc)
            raise RuntimeError(f"Embedding de la requête échoué : {exc}") from exc
        try:
            retrieved_docs = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                note_id=note_id,
            )
        except Exception as exc:
            logger.error("Erreur retrieval Qdrant : %s", exc)
            raise RuntimeError(f"Retrieval échoué : {exc}") from exc
        context_parts = []
        source_documents = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_note_id = doc.get("note_id", "unknown")
            content = doc.get("page_content", "")
            score = doc.get("score", 0.0)
            context_parts.append(
                f"[SOURCE {i} — {doc_note_id} (score: {score:.3f})]\n{content}"
            )
            if doc_note_id not in source_documents:
                source_documents.append(doc_note_id)
        context_used = "\n\n---\n\n".join(context_parts)
        if not context_used:
            logger.warning("Aucun contexte récupéré pour la requête.")
            context_used = "Aucune note clinique pertinente trouvée."
        try:
            response = self.chain.invoke(
                {"context": context_used, "question": question}
            )
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("Erreur génération LLM Anthropic : %s", exc)
            raise RuntimeError(f"Génération LLM échouée : {exc}") from exc
        logger.info(
            "Réponse générée. Sources utilisées : %s", ", ".join(source_documents)
        )
        return {
            "answer": answer,
            "source_documents": source_documents,
            "question": question,
            "context_used": context_used,
        }
