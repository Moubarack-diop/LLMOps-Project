from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    with patch.dict(
        "os.environ",
        {
            "ANTHROPIC_API_KEY": "test-key-123",
            "LLM_MODEL": "claude-haiku-4-5-20251001",
            "QDRANT_HOST": "localhost",
            "QDRANT_PORT": "6333",
            "QDRANT_COLLECTION": "medassist_test",
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
            # Pas de serveur MLflow dans les tests : évite la tentative de
            # connexion du tracing au démarrage de l'app.
            "MEDASSIST_TRACING": "0",
        },
    ):
        from src.api.main import app
        from src.api.routes import reset_resources

        # Vide le cache de singletons pour que chaque test instancie ses
        # propres mocks (les ressources sont partagées entre les requêtes).
        reset_resources()
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client
        reset_resources()


class TestHealthEndpoint:

    @patch("src.api.routes.QdrantStore")
    @patch("httpx.AsyncClient")
    def test_health_endpoint_returns_200(
        self, mock_httpx_class, mock_qdrant_class, test_client
    ):
        mock_store = MagicMock()
        mock_store.is_healthy.return_value = True
        mock_qdrant_class.return_value = mock_store
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=None)
        mock_async_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_class.return_value = mock_async_client
        response = test_client.get("/health")
        assert response.status_code == 200

    @patch("src.api.routes.QdrantStore")
    @patch("httpx.AsyncClient")
    def test_health_endpoint_returns_correct_keys(
        self, mock_httpx_class, mock_qdrant_class, test_client
    ):
        mock_store = MagicMock()
        mock_store.is_healthy.return_value = True
        mock_qdrant_class.return_value = mock_store
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=None)
        mock_async_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_class.return_value = mock_async_client
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data, "Clé 'status' manquante."
        assert "qdrant" in data, "Clé 'qdrant' manquante."
        assert "mlflow" in data, "Clé 'mlflow' manquante."

    @patch("src.api.routes.QdrantStore")
    @patch("httpx.AsyncClient")
    def test_health_endpoint_degraded_when_qdrant_down(
        self, mock_httpx_class, mock_qdrant_class, test_client
    ):
        mock_qdrant_class.side_effect = ConnectionError("Qdrant unreachable")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=None)
        mock_async_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_class.return_value = mock_async_client
        response = test_client.get("/health")
        data = response.json()
        assert data["qdrant"] == "error"
        assert data["status"] == "degraded"


class TestQueryEndpoint:

    @patch("src.api.routes.MedicalRAGPipeline")
    @patch("src.api.routes.MedicalEmbedder")
    @patch("src.api.routes.QdrantStore")
    @patch("src.api.routes.MLflowTracker")
    def test_query_endpoint_valid_input(
        self,
        mock_mlflow_class,
        mock_qdrant_class,
        mock_embedder_class,
        mock_pipeline_class,
        test_client,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "The patient was diagnosed with inferior STEMI.",
            "source_documents": ["note_00000", "note_00001"],
            "question": "What was the diagnosis?",
            "context_used": "Some clinical context.",
        }
        mock_pipeline_class.return_value = mock_pipeline
        mock_mlflow_class.return_value = MagicMock()
        response = test_client.post(
            "/query", json={"question": "What was the patient's diagnosis?", "top_k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "question" in data

    @patch("src.api.routes.MedicalRAGPipeline")
    @patch("src.api.routes.MedicalEmbedder")
    @patch("src.api.routes.QdrantStore")
    @patch("src.api.routes.MLflowTracker")
    def test_query_endpoint_returns_sources(
        self,
        mock_mlflow_class,
        mock_qdrant_class,
        mock_embedder_class,
        mock_pipeline_class,
        test_client,
    ):
        expected_sources = ["note_00000", "note_00001"]
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "Answer text.",
            "source_documents": expected_sources,
            "question": "Question text.",
            "context_used": "Context.",
        }
        mock_pipeline_class.return_value = mock_pipeline
        mock_mlflow_class.return_value = MagicMock()
        response = test_client.post(
            "/query", json={"question": "Some medical question?"}
        )
        data = response.json()
        assert data["sources"] == expected_sources

    def test_query_endpoint_invalid_input_empty_question(self, test_client):
        response = test_client.post("/query", json={"question": "a"})
        assert response.status_code == 422

    def test_query_endpoint_missing_question_field(self, test_client):
        response = test_client.post("/query", json={"top_k": 3})
        assert response.status_code == 422

    def test_query_endpoint_invalid_top_k(self, test_client):
        response = test_client.post(
            "/query", json={"question": "Valid medical question here?", "top_k": 0}
        )
        assert response.status_code == 422


class TestIngestEndpoint:

    @patch("src.api.routes.load_asclepius_dataset")
    @patch("src.api.routes.chunk_documents")
    @patch("src.api.routes.MedicalEmbedder")
    @patch("src.api.routes.QdrantStore")
    @patch("src.api.routes.MLflowTracker")
    def test_ingest_endpoint_returns_200(
        self,
        mock_mlflow_class,
        mock_qdrant_class,
        mock_embedder_class,
        mock_chunk_func,
        mock_load_func,
        test_client,
    ):
        mock_load_func.return_value = [
            {"note": "text", "id": "note_00000", "question": "q", "answer": "a"}
        ] * 10
        mock_chunk_func.return_value = [MagicMock()] * 20
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1] * 384] * 20
        mock_embedder.get_embedding_dimension.return_value = 384
        mock_embedder_class.return_value = mock_embedder
        mock_store = MagicMock()
        mock_store.upsert_documents.return_value = 20
        mock_qdrant_class.return_value = mock_store
        mock_mlflow_class.return_value = MagicMock()
        response = test_client.post("/ingest", json={"n_samples": 10})
        assert response.status_code == 200

    @patch("src.api.routes.load_asclepius_dataset")
    @patch("src.api.routes.chunk_documents")
    @patch("src.api.routes.MedicalEmbedder")
    @patch("src.api.routes.QdrantStore")
    @patch("src.api.routes.MLflowTracker")
    def test_ingest_endpoint_returns_correct_counts(
        self,
        mock_mlflow_class,
        mock_qdrant_class,
        mock_embedder_class,
        mock_chunk_func,
        mock_load_func,
        test_client,
    ):
        n_docs = 5
        n_chunks = 15
        mock_load_func.return_value = [MagicMock()] * n_docs
        mock_chunk_func.return_value = [MagicMock()] * n_chunks
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1] * 384] * n_chunks
        mock_embedder.get_embedding_dimension.return_value = 384
        mock_embedder_class.return_value = mock_embedder
        mock_store = MagicMock()
        mock_store.upsert_documents.return_value = n_chunks
        mock_qdrant_class.return_value = mock_store
        mock_mlflow_class.return_value = MagicMock()
        response = test_client.post("/ingest", json={"n_samples": n_docs})
        data = response.json()
        assert data["documents_ingested"] == n_docs
        assert data["chunks_created"] == n_chunks
        assert data["status"] == "success"


class TestNotesEndpoint:

    @patch("src.api.routes.QdrantStore")
    def test_notes_endpoint_returns_list(self, mock_qdrant_class, test_client):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = ["note_00000", "note_00391"]
        mock_qdrant_class.return_value = mock_store
        response = test_client.get("/notes")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["notes"] == ["note_00000", "note_00391"]


class TestNoteDetailEndpoint:

    @patch("src.api.routes.QdrantStore")
    def test_note_detail_returns_content(self, mock_qdrant_class, test_client):
        mock_store = MagicMock()
        mock_store.get_note.return_value = {
            "note_id": "note_00042",
            "content": "Patient admitted with chest pain.",
            "n_chunks": 3,
            "reference_question": "What was the diagnosis?",
            "reference_answer": "STEMI.",
            "source": "asclepius/note_00042",
        }
        mock_qdrant_class.return_value = mock_store
        response = test_client.get("/notes/note_00042")
        assert response.status_code == 200
        data = response.json()
        assert data["note_id"] == "note_00042"
        assert data["content"] == "Patient admitted with chest pain."
        assert data["n_chunks"] == 3

    @patch("src.api.routes.QdrantStore")
    def test_note_detail_returns_404_when_missing(self, mock_qdrant_class, test_client):
        mock_store = MagicMock()
        mock_store.get_note.return_value = None
        mock_qdrant_class.return_value = mock_store
        response = test_client.get("/notes/note_99999")
        assert response.status_code == 404


class TestApiKeyAuth:

    @patch("src.api.routes.QdrantStore")
    def test_request_rejected_without_key(self, mock_qdrant_class, test_client):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        mock_qdrant_class.return_value = mock_store
        with patch.dict("os.environ", {"MEDASSIST_API_KEY": "secret-123"}):
            response = test_client.get("/notes")
            assert response.status_code == 401

    @patch("src.api.routes.QdrantStore")
    def test_request_accepted_with_key(self, mock_qdrant_class, test_client):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = ["note_00000"]
        mock_qdrant_class.return_value = mock_store
        with patch.dict("os.environ", {"MEDASSIST_API_KEY": "secret-123"}):
            response = test_client.get("/notes", headers={"X-API-Key": "secret-123"})
            assert response.status_code == 200

    @patch("src.api.routes.QdrantStore")
    @patch("httpx.AsyncClient")
    def test_health_stays_open_without_key(
        self, mock_httpx_class, mock_qdrant_class, test_client
    ):
        mock_store = MagicMock()
        mock_store.is_healthy.return_value = True
        mock_qdrant_class.return_value = mock_store
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=None)
        mock_async_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_class.return_value = mock_async_client
        with patch.dict("os.environ", {"MEDASSIST_API_KEY": "secret-123"}):
            response = test_client.get("/health")
            assert response.status_code == 200


class TestQueryNoteFilter:

    @patch("src.api.routes.MedicalRAGPipeline")
    @patch("src.api.routes.MedicalEmbedder")
    @patch("src.api.routes.QdrantStore")
    @patch("src.api.routes.MLflowTracker")
    def test_query_forwards_note_id(
        self,
        mock_mlflow_class,
        mock_qdrant_class,
        mock_embedder_class,
        mock_pipeline_class,
        test_client,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "Answer.",
            "source_documents": ["note_00391"],
            "question": "Q?",
            "context_used": "Ctx.",
        }
        mock_pipeline_class.return_value = mock_pipeline
        mock_mlflow_class.return_value = MagicMock()
        response = test_client.post(
            "/query",
            json={"question": "What was the diagnosis?", "note_id": "note_00391"},
        )
        assert response.status_code == 200
        call_kwargs = mock_pipeline.query.call_args[1]
        assert call_kwargs.get("note_id") == "note_00391"
