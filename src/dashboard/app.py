import os
import httpx
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def call_api(method: str, endpoint: str, timeout: float = 120.0, **kwargs) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=timeout) as client:
            func = getattr(client, method)
            response = func(url, **kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        raise ConnectionError(
            f"Impossible de se connecter à l'API MedAssist ({API_BASE_URL}). Vérifiez que le serveur est démarré."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Erreur API ({exc.response.status_code}) : {exc.response.text}"
        ) from exc


@st.cache_data(ttl=60, show_spinner=False)
def fetch_notes() -> list[str]:
    try:
        data = call_api("get", "/notes")
        return data.get("notes", [])
    except Exception:
        return []


def render_sidebar() -> None:
    st.sidebar.title("Configuration")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Statut des services")
    if st.sidebar.button("Vérifier le statut", key="health_check"):
        try:
            health = call_api("get", "/health")
            qdrant_icon = "✅" if health["qdrant"] == "ok" else "❌"
            mlflow_icon = "✅" if health["mlflow"] == "ok" else "❌"
            llm_icon = "✅" if health.get("llm") == "ok" else "❌"
            st.sidebar.success(f"{qdrant_icon} Qdrant : {health['qdrant']}")
            st.sidebar.info(f"{mlflow_icon} MLflow : {health['mlflow']}")
            st.sidebar.info(f"{llm_icon} Claude (LLM) : {health.get('llm', 'unknown')}")
        except ConnectionError as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:
            st.sidebar.error(f"Erreur : {exc}")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ingestion des données")
    n_samples = st.sidebar.slider(
        "Nombre d'exemples",
        min_value=100,
        max_value=1000,
        value=500,
        step=100,
        help="Nombre de notes cliniques à ingérer depuis Asclepius.",
    )
    chunk_size = st.sidebar.slider(
        "Taille des chunks",
        min_value=128,
        max_value=1024,
        value=512,
        step=64,
        help="Taille maximale de chaque chunk de texte.",
    )
    if st.sidebar.button("Ingérer les données", type="primary", key="ingest_btn"):
        with st.sidebar:
            with st.spinner(f"Ingestion de {n_samples} notes cliniques..."):
                try:
                    result = call_api(
                        "post",
                        "/ingest",
                        timeout=900.0,
                        json={
                            "n_samples": n_samples,
                            "chunk_size": chunk_size,
                            "chunk_overlap": 50,
                        },
                    )
                    st.success(
                        f"Ingestion terminée !\n\nDocuments : {result['documents_ingested']}\n\nChunks : {result['chunks_created']}"
                    )
                except ConnectionError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Erreur lors de l'ingestion : {exc}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "*MedAssist v1.0.0*\n\nAuteur : Mouhamed Diop\n\nEncadrant : Mously DIAW"
    )


def render_query_tab() -> None:
    st.subheader("Interrogez les dossiers patients")
    st.markdown(
        "Posez votre question médicale en langage naturel. Le système RAG analysera les notes cliniques disponibles et vous fournira une réponse sourcée."
    )
    question = st.text_area(
        "Votre question",
        placeholder="Ex: Quels sont les antécédents cardiovasculaires du patient ? Quelle est la posologie prescrite pour ce diabétique ?",
        height=100,
        key="query_input",
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.number_input("Top-K sources", min_value=1, max_value=20, value=5)
    with col2:
        options = ["Toutes les notes"] + fetch_notes()
        selected_note = st.selectbox(
            "Cibler une note (patient)",
            options,
            index=0,
            help="Restreint la recherche à une seule note clinique.",
        )
    note_id = None if selected_note == "Toutes les notes" else selected_note
    if st.button("Envoyer la requête", type="primary", disabled=not question.strip()):
        if question.strip():
            with st.spinner("Analyse des dossiers patients en cours..."):
                try:
                    payload = {"question": question, "top_k": top_k}
                    if note_id:
                        payload["note_id"] = note_id
                    result = call_api("post", "/query", json=payload)
                    st.markdown("---")
                    st.subheader("Réponse")
                    st.markdown(result["answer"])
                    if result.get("sources"):
                        st.markdown("---")
                        st.subheader("Sources citées")
                        for source in result["sources"]:
                            st.code(source, language=None)
                    else:
                        st.info("Aucune source identifiée.")
                except ConnectionError as exc:
                    st.error(str(exc))
                    st.info(
                        "Assurez-vous que l'API MedAssist est démarrée avec :\n\n`uvicorn src.api.main:app --reload`"
                    )
                except Exception as exc:
                    st.error(f"Erreur : {exc}")


def render_metrics_tab() -> None:
    st.subheader("Métriques de qualité RAGAS")
    st.markdown(
        "Visualisez les scores de qualité du pipeline RAG calculés lors des dernières évaluations MLflow."
    )
    if st.button("Charger les métriques", key="load_metrics"):
        try:
            metrics = call_api("get", "/metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Faithfulness",
                    f"{metrics['faithfulness']:.2%}",
                    help="La réponse est-elle fidèle au contexte ?",
                )
            with col2:
                st.metric(
                    "Answer Relevancy",
                    f"{metrics['answer_relevancy']:.2%}",
                    help="La réponse répond-elle à la question ?",
                )
            with col3:
                st.metric(
                    "Context Precision",
                    f"{metrics['context_precision']:.2%}",
                    help="Le contexte récupéré est-il pertinent ?",
                )
            scores_df = pd.DataFrame(
                {
                    "Métrique": [
                        "Faithfulness",
                        "Answer Relevancy",
                        "Context Precision",
                    ],
                    "Score": [
                        metrics["faithfulness"],
                        metrics["answer_relevancy"],
                        metrics["context_precision"],
                    ],
                }
            )
            st.bar_chart(scores_df.set_index("Métrique"))
            st.caption(
                f"Run MLflow : `{metrics['run_id']}` | Expérience : `{metrics['experiment_name']}`"
            )
        except ConnectionError as exc:
            st.error(str(exc))
        except Exception as exc:
            if "404" in str(exc):
                st.warning(
                    "Aucune métrique disponible. Lancez d'abord une évaluation RAGAS via l'API."
                )
            else:
                st.error(f"Erreur : {exc}")
    st.markdown("---")
    st.markdown(
        "### Description des métriques RAGAS\n\n| Métrique | Description | Plage |\n|---|---|---|\n| **Faithfulness** | Mesure si la réponse est supportée par le contexte récupéré. Score = 1 si toutes les affirmations sont vérifiables. | [0, 1] |\n| **Answer Relevancy** | Évalue si la réponse répond réellement à la question posée. Pénalise les réponses incomplètes ou hors-sujet. | [0, 1] |\n| **Context Precision** | Vérifie si les passages récupérés sont pertinents pour répondre à la question. Évalue la qualité du retrieval. | [0, 1] |\n"
    )


def main() -> None:
    st.set_page_config(
        page_title="MedAssist — Assistant Médical",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("MedAssist — Assistant Médical Intelligent")
    st.markdown(
        "Système RAG permettant aux cliniciens d'interroger des dossiers patients en langage naturel, basé sur le dataset **Asclepius Synthetic Clinical Notes**."
    )
    st.markdown("---")
    render_sidebar()
    tab_query, tab_metrics = st.tabs(["Requête médicale", "Métriques RAGAS"])
    with tab_query:
        render_query_tab()
    with tab_metrics:
        render_metrics_tab()


if __name__ == "__main__":
    main()
