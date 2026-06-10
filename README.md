# MedAssist — Assistant Médical Intelligent

> Système RAG (Retrieval-Augmented Generation) permettant à un clinicien d'interroger des dossiers patients en langage naturel.

**Auteur** : Mouhamed Diop | **Encadrant** : Mously DIAW
**Dataset** : [Asclepius Synthetic Clinical Notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes) — 157 000 notes cliniques synthétiques

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLINICIEN                                │
│                   (Question en langage naturel)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   React UI      │  servie par l'API
                    │  (frontend/)    │  http://localhost:8000/
                    └────────┬────────┘
                             │ HTTP
                    ┌────────▼────────┐
                    │   FastAPI API   │  port 8000
                    │  /query /ingest │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
 ┌────────▼────────┐ ┌───────▼───────┐ ┌───────▼───────┐
 │  MedicalEmbedder│ │  QdrantStore  │ │ ChatAnthropic │
 │ all-MiniLM-L6   │ │  (Retrieval)  │ │  (Generation) │
 │   dim=384       │ │   port 6333   │ │  claude-haiku │
 └────────┬────────┘ └───────┬───────┘ └───────────────┘
          │                  │
          └──────────────────┘
                    │
          ┌─────────▼─────────┐
          │     MLflow UI     │  port 5000
          │  (RAGAS Tracking) │
          └───────────────────┘

Dataset : HuggingFace Asclepius → DVC → Qdrant Vector Store
```

---

## Stack Technique

| Composant | Outil |
|---|---|
| LLM | Claude via Anthropic (`claude-haiku-4-5`) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Base vectorielle | Qdrant |
| Orchestration RAG | LangChain |
| API | FastAPI |
| Interface | React + Vite (TypeScript) |
| Monitoring qualité | RAGAS |
| Tracking ML | MLflow |
| Tests | Pytest |
| CI/CD | GitHub Actions |
| Containerisation | Docker + docker-compose |
| Versioning données | DVC |
| Linting | flake8 + black |

---

## Installation

### Prérequis

- Python 3.11+
- Docker + Docker Compose
- Une clé API Anthropic ([console.anthropic.com](https://console.anthropic.com))

### 0. Obtenir une clé API Anthropic

Créez une clé API sur [console.anthropic.com](https://console.anthropic.com), puis
renseignez-la dans le fichier `.env` (étape 2). Aucun modèle à télécharger : le LLM
Claude est appelé via l'API.

Modèles compatibles (variable `LLM_MODEL`) :

```text
claude-haiku-4-5-20251001   # rapide et économique (défaut)
claude-sonnet-4-6           # plus capable, ~4x plus cher
```

### 1. Cloner et configurer

```bash
git clone <votre-repo>
cd medassist

# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer le package en mode développement
pip install -e .
```

### 2. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditez .env et renseignez votre ANTHROPIC_API_KEY
```

Contenu du `.env` :

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5-20251001
# LLM_MODEL=claude-sonnet-4-6   # plus capable, 4x plus cher
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=medassist
MLFLOW_TRACKING_URI=http://localhost:5000
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
```

Variables optionnelles :

```env
# Protège les endpoints (sauf /health) : les clients doivent envoyer la clé
# dans l'en-tête HTTP X-API-Key. Absente = API ouverte (usage local).
MEDASSIST_API_KEY=changez-moi
# Origines CORS autorisées, séparées par des virgules ("*" par défaut).
CORS_ORIGINS=http://localhost:5173
# Pré-charge le modèle d'embedding et la connexion Qdrant au démarrage de
# l'API (activé dans docker-compose) au lieu de la première requête.
MEDASSIST_EAGER_INIT=1
```

> ⚠️ Le fichier `.env` contient votre clé secrète : il ne doit **jamais** être commité
> (il est listé dans `.gitignore`). Ne partagez jamais cette clé.

---

## Lancement avec Docker Compose

```bash
# Démarrer tous les services (Qdrant + MLflow + API)
docker-compose up -d

# Vérifier que les services sont actifs
docker-compose ps

# Voir les logs
docker-compose logs -f api
```

Services disponibles :
- **Interface web** : http://localhost:8000/
- **API** : http://localhost:8000
- **Qdrant Dashboard** : http://localhost:6333/dashboard
- **MLflow UI** : http://localhost:5000

---

## Interface web (React)

L'interface principale est une application React (dossier `frontend/`) de type
gestion de dossiers médicaux : annuaire des patients avec recherche, fiche
dossier structurée, et interrogation du dossier par l'assistant IA avec
citations des sources.

```bash
# Développement (rechargement à chaud, proxy API automatique vers :8000)
cd frontend
npm install
npm run dev          # http://localhost:5173

# Production : le build est servi par FastAPI sur http://localhost:8000/
npm run build        # génère frontend/dist, détecté au démarrage de l'API
```

L'image Docker construit l'interface automatiquement (build multi-étapes) :
avec `docker-compose up -d`, l'interface est disponible sur http://localhost:8000/.

---

## Lancement en développement local

```bash
# Terminal 1 — Démarrer Qdrant et MLflow uniquement
docker-compose up -d qdrant mlflow

# Terminal 2 — Démarrer l'API FastAPI
uvicorn src.api.main:app --reload --port 8000

# Terminal 3 (optionnel) — Frontend en mode développement (hot reload)
cd frontend && npm run dev    # http://localhost:5173
```

> Sans le terminal 3, l'interface de production (`frontend/dist`) est servie
> directement par l'API sur http://localhost:8000/.

---

## Utilisation

### Étape 1 : Ingérer les données

Via l'API :
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 500, "chunk_size": 512, "chunk_overlap": 50}'
```


### Étape 2 : Interroger le système

Via l'API :
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les antécédents cardiovasculaires du patient ?", "top_k": 5}'
```

Via l'interface web : http://localhost:8000/ — sélectionnez un dossier patient
puis utilisez le panneau **« Interroger ce dossier »**.

### Étape 3 : Évaluation RAGAS (optionnel)

```python
from src.monitoring.ragas_eval import RAGASEvaluator
from src.monitoring.mlflow_tracker import MLflowTracker

evaluator = RAGASEvaluator()
scores = evaluator.evaluate(
    questions=["What is the diagnosis?"],
    answers=["Inferior STEMI."],
    contexts=[["Patient admitted with chest pain..."]],
    ground_truths=["The patient had inferior STEMI."]
)

tracker = MLflowTracker()
tracker.log_ragas_metrics(scores)
```

---

## Endpoints API

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Statut de l'API et des services |
| `POST` | `/ingest` | Ingestion du dataset Asclepius |
| `POST` | `/query` | Requête RAG en langage naturel |
| `GET` | `/metrics` | Dernières métriques RAGAS |
| `GET` | `/docs` | Documentation Swagger UI |
| `GET` | `/redoc` | Documentation ReDoc |

### Exemple de requête `/query`

**Request :**
```json
{
  "question": "What medications were prescribed for the cardiac patient?",
  "top_k": 5
}
```

**Response :**
```json
{
  "answer": "D'après les notes cliniques (note_00042, note_00087)...",
  "sources": ["note_00042", "note_00087"],
  "question": "What medications were prescribed for the cardiac patient?"
}
```

---

## Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec couverture de code
pytest tests/ --cov=src --cov-report=term-missing

# Un fichier spécifique
pytest tests/test_ingestion.py -v
pytest tests/test_rag_pipeline.py -v
pytest tests/test_api.py -v
```

---

## Linting et formatage

```bash
# Vérifier le formatage black
black src/ tests/ --check

# Appliquer le formatage
black src/ tests/

# Vérifier le style flake8
flake8 src/ tests/
```

---

## Pipeline DVC

```bash
# Initialiser DVC (première fois)
dvc init

# Exécuter le pipeline d'ingestion
dvc repro ingest

# Exécuter le pipeline complet
dvc repro
```

---

## Métriques RAGAS

| Métrique | Description | Valeur cible |
|---|---|---|
| **Faithfulness** | Mesure si la réponse est supportée par le contexte récupéré. Score = 1 si toutes les affirmations sont vérifiables dans les sources. | ≥ 0.80 |
| **Answer Relevancy** | Évalue si la réponse répond réellement à la question posée. Pénalise les réponses incomplètes ou hors-sujet. | ≥ 0.75 |
| **Context Precision** | Vérifie si les passages récupérés sont pertinents pour répondre à la question. Évalue la qualité du retrieval. | ≥ 0.70 |

Les métriques sont trackées automatiquement dans MLflow après chaque évaluation.
Accès au dashboard MLflow : http://localhost:5000

---

## Structure du projet

```
medassist/
├── src/
│   ├── ingestion/          # Chargement, chunking, embedding
│   ├── retrieval/          # Qdrant + pipeline RAG LangChain
│   ├── api/                # FastAPI (routes, schemas)
│   └── monitoring/         # RAGAS + MLflow
├── frontend/               # Interface web React + Vite
├── tests/                  # Tests unitaires pytest
├── notebooks/              # Exploration du dataset
├── .github/workflows/      # CI/CD GitHub Actions
├── data/
│   ├── raw/                # Données brutes (DVC)
│   └── processed/          # Embeddings (DVC)
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

## CI/CD

Le pipeline GitHub Actions s'exécute automatiquement sur chaque `push` et `pull_request` vers `main` :

1. **lint** : Vérification du style avec `black` et `flake8`
2. **test** : Exécution de `pytest` avec rapport de couverture
3. **build** : Construction et validation de l'image Docker

Pour exécuter une évaluation réelle (non mockée) en CI, configurez le secret
`ANTHROPIC_API_KEY` dans les Settings GitHub du dépôt. Les tests unitaires mockent
tous les appels externes et n'en ont pas besoin.

---

*Projet académique MLOps/LLMOps — 2026*
