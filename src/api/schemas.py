from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question médicale à soumettre au pipeline RAG.",
        examples=["Quels sont les antécédents cardiovasculaires du patient ?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre de documents contextuels à récupérer.",
    )
    note_id: str | None = Field(
        default=None,
        description=(
            "Restreint la recherche à une note précise (ex: 'note_00391'). "
            "Si absent, la recherche porte sur toutes les notes."
        ),
        examples=["note_00391"],
    )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Réponse générée par le pipeline RAG.")
    sources: list[str] = Field(
        default_factory=list, description="Identifiants des notes cliniques sources."
    )
    question: str = Field(..., description="Question originale.")


class IngestRequest(BaseModel):
    n_samples: int = Field(
        default=500,
        ge=1,
        le=157000,
        description="Nombre d'exemples à ingérer depuis Asclepius.",
    )
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Taille maximale des chunks en caractères.",
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=200, description="Chevauchement entre chunks consécutifs."
    )


class IngestResponse(BaseModel):
    status: str = Field(..., description="Statut de l'ingestion.")
    documents_ingested: int = Field(
        default=0, description="Nombre de documents source ingérés."
    )
    chunks_created: int = Field(
        default=0, description="Nombre de chunks vectorisés et stockés."
    )
    message: str = Field(default="", description="Message informatif.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Statut global de l'API.")
    qdrant: str = Field(..., description="Statut de Qdrant ('ok' ou 'error').")
    mlflow: str = Field(..., description="Statut de MLflow ('ok' ou 'error').")
    llm: str = Field(
        default="unknown",
        description="Statut du LLM Claude/Anthropic ('ok' ou 'error').",
    )


class NotesResponse(BaseModel):
    count: int = Field(default=0, description="Nombre de notes distinctes.")
    notes: list[str] = Field(
        default_factory=list, description="Identifiants des notes disponibles."
    )


class NoteDetailResponse(BaseModel):
    note_id: str = Field(..., description="Identifiant de la note clinique.")
    content: str = Field(..., description="Texte complet reconstitué de la note.")
    n_chunks: int = Field(default=0, description="Nombre de chunks indexés.")
    reference_question: str = Field(
        default="", description="Question de référence du dataset Asclepius."
    )
    reference_answer: str = Field(
        default="", description="Réponse de référence du dataset Asclepius."
    )
    source: str = Field(default="", description="Source du document.")


class MetricsResponse(BaseModel):
    run_id: str = Field(default="", description="Identifiant du run MLflow.")
    faithfulness: float = Field(
        default=0.0, description="Score RAGAS faithfulness (0-1)."
    )
    answer_relevancy: float = Field(
        default=0.0, description="Score RAGAS answer_relevancy (0-1)."
    )
    context_precision: float = Field(
        default=0.0, description="Score RAGAS context_precision (0-1)."
    )
    experiment_name: str = Field(
        default="MedAssist-RAG", description="Nom de l'expérience MLflow."
    )
