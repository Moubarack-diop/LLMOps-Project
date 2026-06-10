from typing import Any
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    return [
        {
            "note": "Patient admitted with chest pain. History of hypertension. Prescribed metoprolol 50mg twice daily. Discharged after 3 days. Follow-up scheduled in 2 weeks. Blood pressure controlled.",
            "question": "What medication was prescribed?",
            "answer": "Metoprolol 50mg twice daily.",
            "id": "note_00000",
        },
        {
            "note": "Elderly patient with type 2 diabetes. HbA1c at 8.2%. Started on metformin 500mg once daily with meals. Diet counseling provided. Return visit in 3 months.",
            "question": "What is the HbA1c value?",
            "answer": "8.2%",
            "id": "note_00001",
        },
        {
            "note": "Post-operative report following appendectomy. No complications observed. Patient tolerating oral fluids. Wound healing satisfactory. Discharged on day 2.",
            "question": "What surgery was performed?",
            "answer": "Appendectomy.",
            "id": "note_00002",
        },
        {
            "note": "Asthma exacerbation treated with nebulized salbutamol. Oxygen saturation improved to 98%. Prednisolone course started. Patient education on inhaler technique provided.",
            "question": "What treatment was given for asthma?",
            "answer": "Nebulized salbutamol and prednisolone.",
            "id": "note_00003",
        },
        {
            "note": "Acute UTI treated with trimethoprim for 7 days. Urine culture pending. Advised to increase fluid intake. Symptoms resolved within 48 hours.",
            "question": "How long is the antibiotic course?",
            "answer": "7 days.",
            "id": "note_00004",
        },
    ]


@pytest.fixture
def fake_hf_dataset(sample_documents: list[dict[str, Any]]):
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=len(sample_documents))
    mock_dataset.select = MagicMock(return_value=sample_documents)
    mock_hf_rows = []
    for doc in sample_documents:
        row = {
            "note": doc["note"],
            "question": doc["question"],
            "answer": doc["answer"],
        }
        mock_hf_rows.append(row)
    mock_dataset.__iter__ = MagicMock(return_value=iter(mock_hf_rows))
    mock_dataset.select.return_value = mock_hf_rows
    return mock_dataset


class TestLoadAsclepiusDataset:

    @patch("src.ingestion.loader.load_dataset")
    def test_load_dataset_returns_correct_format(
        self, mock_load_dataset, fake_hf_dataset, sample_documents
    ):
        mock_load_dataset.return_value = fake_hf_dataset
        from src.ingestion.loader import load_asclepius_dataset

        docs = load_asclepius_dataset(n_samples=5)
        assert isinstance(docs, list), "Le résultat doit être une liste."
        assert len(docs) == 5, "Doit retourner exactement 5 documents."
        for doc in docs:
            assert "note" in doc, "Chaque document doit avoir une clé 'note'."
            assert "question" in doc, "Chaque document doit avoir une clé 'question'."
            assert "answer" in doc, "Chaque document doit avoir une clé 'answer'."
            assert "id" in doc, "Chaque document doit avoir une clé 'id'."

    @patch("src.ingestion.loader.load_dataset")
    def test_load_dataset_ids_are_unique(self, mock_load_dataset, fake_hf_dataset):
        mock_load_dataset.return_value = fake_hf_dataset
        from src.ingestion.loader import load_asclepius_dataset

        docs = load_asclepius_dataset(n_samples=5)
        ids = [d["id"] for d in docs]
        assert len(ids) == len(set(ids)), "Les identifiants doivent être uniques."

    @patch("src.ingestion.loader.load_dataset")
    def test_load_dataset_raises_on_failure(self, mock_load_dataset):
        mock_load_dataset.side_effect = Exception("Network error")
        from src.ingestion.loader import load_asclepius_dataset

        with pytest.raises(RuntimeError, match="Impossible de charger le dataset"):
            load_asclepius_dataset(n_samples=5)


class TestChunkDocuments:

    def test_chunk_documents_returns_langchain_documents(
        self, sample_documents: list[dict[str, Any]]
    ):
        from src.ingestion.chunker import chunk_documents

        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=10)
        assert isinstance(chunks, list), "La sortie doit être une liste."
        assert len(chunks) > 0, "Il doit y avoir au moins un chunk."
        for chunk in chunks:
            assert isinstance(chunk, Document), "Chaque élément doit être un Document."

    def test_chunk_documents_respects_size(
        self, sample_documents: list[dict[str, Any]]
    ):
        chunk_size = 150
        from src.ingestion.chunker import chunk_documents

        chunks = chunk_documents(
            sample_documents, chunk_size=chunk_size, chunk_overlap=20
        )
        oversized = [c for c in chunks if len(c.page_content) > chunk_size + 50]
        assert (
            len(oversized) == 0
        ), f"{len(oversized)} chunks dépassent la taille maximale."

    def test_chunk_documents_preserves_metadata(
        self, sample_documents: list[dict[str, Any]]
    ):
        from src.ingestion.chunker import chunk_documents

        chunks = chunk_documents(sample_documents, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert "note_id" in chunk.metadata, "Métadonnée 'note_id' manquante."
            assert "question" in chunk.metadata, "Métadonnée 'question' manquante."
            assert "answer" in chunk.metadata, "Métadonnée 'answer' manquante."

    def test_chunk_documents_raises_on_empty_list(self):
        from src.ingestion.chunker import chunk_documents

        with pytest.raises(ValueError, match="vide"):
            chunk_documents([], chunk_size=512, chunk_overlap=50)

    def test_chunk_documents_produces_more_chunks_with_smaller_size(
        self, sample_documents: list[dict[str, Any]]
    ):
        from src.ingestion.chunker import chunk_documents

        chunks_small = chunk_documents(
            sample_documents, chunk_size=100, chunk_overlap=10
        )
        chunks_large = chunk_documents(
            sample_documents, chunk_size=1000, chunk_overlap=50
        )
        assert len(chunks_small) >= len(
            chunks_large
        ), "Des chunks plus petits doivent produire au moins autant de chunks."


class TestMedicalEmbedder:

    @pytest.fixture
    def mock_sentence_transformer(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(5, 384).astype("float32")
        return mock_model

    @patch("src.ingestion.embedder.SentenceTransformer")
    def test_embedder_initializes_correctly(
        self, mock_st_class, mock_sentence_transformer
    ):
        mock_st_class.return_value = mock_sentence_transformer
        from src.ingestion.embedder import MedicalEmbedder

        embedder = MedicalEmbedder(model_name="test-model")
        assert embedder.embedding_dim == 384

    @patch("src.ingestion.embedder.SentenceTransformer")
    def test_embedder_returns_correct_dimension(
        self, mock_st_class, mock_sentence_transformer, sample_documents
    ):
        import numpy as np

        mock_st_class.return_value = mock_sentence_transformer
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.embedder import MedicalEmbedder

        embedder = MedicalEmbedder(model_name="test-model")
        chunks = chunk_documents(sample_documents, chunk_size=300, chunk_overlap=20)
        n_chunks = len(chunks)
        mock_sentence_transformer.encode.return_value = np.random.rand(
            n_chunks, 384
        ).astype("float32")
        embeddings = embedder.embed_documents(chunks)
        assert isinstance(embeddings, list), "Les embeddings doivent être une liste."
        assert (
            len(embeddings) == n_chunks
        ), f"Doit y avoir {n_chunks} embeddings, reçu {len(embeddings)}."
        for emb in embeddings:
            assert len(emb) == 384, f"Dimension attendue 384, obtenue {len(emb)}."

    @patch("src.ingestion.embedder.SentenceTransformer")
    def test_embed_query_returns_vector(self, mock_st_class, mock_sentence_transformer):
        import numpy as np

        mock_st_class.return_value = mock_sentence_transformer
        mock_sentence_transformer.encode.return_value = np.random.rand(384).astype(
            "float32"
        )
        from src.ingestion.embedder import MedicalEmbedder

        embedder = MedicalEmbedder(model_name="test-model")
        vector = embedder.embed_query("What is the patient's diagnosis?")
        assert isinstance(vector, list), "L'embedding doit être une liste."
        assert len(vector) == 384, f"Dimension attendue 384, obtenue {len(vector)}."

    @patch("src.ingestion.embedder.SentenceTransformer")
    def test_embed_query_raises_on_empty_string(
        self, mock_st_class, mock_sentence_transformer
    ):
        mock_st_class.return_value = mock_sentence_transformer
        from src.ingestion.embedder import MedicalEmbedder

        embedder = MedicalEmbedder(model_name="test-model")
        with pytest.raises(ValueError, match="vide"):
            embedder.embed_query("")
