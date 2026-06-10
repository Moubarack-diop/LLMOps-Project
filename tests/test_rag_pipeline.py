from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_vector_store():
    mock_store = MagicMock()
    mock_store.is_healthy.return_value = True
    mock_store.search.return_value = [
        {
            "page_content": "Patient admitted with severe chest pain. ECG showed ST elevation in leads II, III, aVF. Diagnosed with inferior STEMI. Treated with primary PCI.",
            "note_id": "note_00000",
            "score": 0.92,
            "question": "What was the cardiac diagnosis?",
            "answer": "Inferior STEMI.",
            "source": "asclepius/note_00000",
        },
        {
            "page_content": "Aspirin 300mg loading dose administered. Clopidogrel 600mg given. Patient transferred to cath lab. Successful stent placement in right coronary artery.",
            "note_id": "note_00001",
            "score": 0.85,
            "question": "What treatment was given?",
            "answer": "Primary PCI with stent.",
            "source": "asclepius/note_00001",
        },
    ]
    return mock_store


@pytest.fixture
def mock_embedder():
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.1] * 384
    mock_emb.get_embedding_dimension.return_value = 384
    return mock_emb


@pytest.fixture
def mock_llm_response():
    mock_response = MagicMock()
    mock_response.content = "D'après les notes cliniques disponibles, le patient a été diagnostiqué avec un STEMI inférieur (note_00000). Le traitement administré comprenait de l'aspirine 300mg, du clopidogrel 600mg, et une angioplastie primaire avec pose de stent dans l'artère coronaire droite (note_00001)."
    return mock_response


class TestMedicalRAGPipeline:

    @patch("src.retrieval.rag_pipeline.ChatAnthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_pipeline_initializes_correctly(
        self, mock_anthropic_class, mock_vector_store, mock_embedder
    ):
        mock_anthropic_class.return_value = MagicMock()
        from src.retrieval.rag_pipeline import MedicalRAGPipeline

        pipeline = MedicalRAGPipeline(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_model="claude-haiku-4-5-20251001",
            top_k=5,
        )
        assert pipeline is not None

    @patch("src.retrieval.rag_pipeline.ChatAnthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_pipeline_returns_answer_and_sources(
        self, mock_anthropic_class, mock_vector_store, mock_embedder, mock_llm_response
    ):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_llm_response
        mock_anthropic_class.return_value = mock_llm
        from src.retrieval.rag_pipeline import MedicalRAGPipeline

        pipeline = MedicalRAGPipeline(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_model="claude-haiku-4-5-20251001",
            top_k=2,
        )
        pipeline.chain = mock_chain
        result = pipeline.query("What was the cardiac diagnosis?")
        assert isinstance(result, dict), "Le résultat doit être un dictionnaire."
        assert "answer" in result, "La clé 'answer' est manquante."
        assert "source_documents" in result, "La clé 'source_documents' est manquante."
        assert "question" in result, "La clé 'question' est manquante."
        assert "context_used" in result, "La clé 'context_used' est manquante."

    @patch("src.retrieval.rag_pipeline.ChatAnthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_pipeline_sources_contain_note_ids(
        self, mock_anthropic_class, mock_vector_store, mock_embedder, mock_llm_response
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_llm_response
        mock_anthropic_class.return_value = MagicMock()
        from src.retrieval.rag_pipeline import MedicalRAGPipeline

        pipeline = MedicalRAGPipeline(
            vector_store=mock_vector_store, embedder=mock_embedder
        )
        pipeline.chain = mock_chain
        result = pipeline.query("What treatment was administered?")
        assert isinstance(result["source_documents"], list)
        for source in result["source_documents"]:
            assert isinstance(source, str), "Chaque source doit être une chaîne."
            assert source.startswith("note_"), f"Source inattendue : {source}"

    @patch("src.retrieval.rag_pipeline.ChatAnthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_pipeline_handles_empty_context(
        self, mock_anthropic_class, mock_embedder, mock_llm_response
    ):
        empty_store = MagicMock()
        empty_store.search.return_value = []
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_llm_response
        mock_anthropic_class.return_value = MagicMock()
        from src.retrieval.rag_pipeline import MedicalRAGPipeline

        pipeline = MedicalRAGPipeline(vector_store=empty_store, embedder=mock_embedder)
        pipeline.chain = mock_chain
        result = pipeline.query("Any question that yields no results?")
        assert result["source_documents"] == [], "Sources vides si contexte vide."
        assert "Aucune note clinique" in result["context_used"] or result["answer"]

    @patch("src.retrieval.rag_pipeline.ChatAnthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_query_response_format(
        self, mock_anthropic_class, mock_vector_store, mock_embedder, mock_llm_response
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_llm_response
        mock_anthropic_class.return_value = MagicMock()
        from src.retrieval.rag_pipeline import MedicalRAGPipeline

        pipeline = MedicalRAGPipeline(
            vector_store=mock_vector_store, embedder=mock_embedder
        )
        pipeline.chain = mock_chain
        question = "What medications were prescribed?"
        result = pipeline.query(question)
        assert result["question"] == question, "La question doit être préservée."
        assert isinstance(result["answer"], str), "La réponse doit être une chaîne."
        assert len(result["answer"]) > 0, "La réponse ne doit pas être vide."
        assert isinstance(result["source_documents"], list)
        assert isinstance(result["context_used"], str)

    @patch("src.retrieval.rag_pipeline.ChatAnthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_query_raises_on_empty_question(
        self, mock_anthropic_class, mock_vector_store, mock_embedder
    ):
        mock_anthropic_class.return_value = MagicMock()
        from src.retrieval.rag_pipeline import MedicalRAGPipeline

        pipeline = MedicalRAGPipeline(
            vector_store=mock_vector_store, embedder=mock_embedder
        )
        with pytest.raises(ValueError, match="vide"):
            pipeline.query("")

    @patch("src.retrieval.rag_pipeline.ChatAnthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_pipeline_uses_configured_claude_model(
        self, mock_anthropic_class, mock_vector_store, mock_embedder
    ):
        mock_anthropic_class.return_value = MagicMock()
        from src.retrieval.rag_pipeline import MedicalRAGPipeline

        pipeline = MedicalRAGPipeline(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_model="claude-haiku-4-5-20251001",
        )
        assert pipeline.llm_model == "claude-haiku-4-5-20251001"
        mock_anthropic_class.assert_called_once()
        call_kwargs = mock_anthropic_class.call_args[1]
        assert call_kwargs.get("model") == "claude-haiku-4-5-20251001"

    @patch.dict("os.environ", {}, clear=True)
    def test_pipeline_raises_without_api_key(self, mock_vector_store, mock_embedder):
        from src.retrieval.rag_pipeline import MedicalRAGPipeline

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            MedicalRAGPipeline(vector_store=mock_vector_store, embedder=mock_embedder)

    @patch("src.retrieval.rag_pipeline.ChatAnthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    def test_pipeline_passes_note_id_filter(
        self, mock_anthropic_class, mock_vector_store, mock_embedder, mock_llm_response
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_llm_response
        mock_anthropic_class.return_value = MagicMock()
        from src.retrieval.rag_pipeline import MedicalRAGPipeline

        pipeline = MedicalRAGPipeline(
            vector_store=mock_vector_store, embedder=mock_embedder
        )
        pipeline.chain = mock_chain
        pipeline.query("What was the diagnosis?", note_id="note_00000")
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs.get("note_id") == "note_00000"
