# RAG Classique — Explication Générale du Projet

## 1. Présentation

Ce projet implémente un système **RAG (Retrieval-Augmented Generation)** classique, destiné à répondre à des questions sur des **documents juridiques et fiscaux français** (notamment le Code Général des Impôts, articles liés à la facturation électronique, directives européennes, etc.).

Le principe du RAG est simple : au lieu de demander au LLM de mémoriser toutes les lois, on lui fournit à chaque requête les passages de texte les plus pertinents extraits d'une base documentaire. Le LLM répond ensuite **uniquement à partir de ces passages**, ce qui garantit que les réponses sont fondées sur les textes officiels et non sur des hallucinations.

---

## 2. Architecture Globale

```
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 1 — INGESTION (offline)             │
│                                                             │
│  PDFs  ──►  DocumentLoader  ──►  Chunker  ──►  Embedder    │
│  (CGI, etc.)  (Unstructured)   (1000 chars)  (BERT 768d)   │
│                                        │                    │
│                                        ▼                    │
│                               FAISSVectorStore              │
│                            (index FAISS + BM25)             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PHASE 2 — QUERY (online)                  │
│                                                             │
│  Question  ──►  Embedder  ──►  VectorStore.hybrid_search   │
│                                        │                    │
│                               top_k chunks pertinents       │
│                                        │                    │
│                                        ▼                    │
│                               OllamaLLM.generate            │
│                             (llama3.1:8b, prompt RAG)       │
│                                        │                    │
│                                        ▼                    │
│                                    Réponse                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PHASE 3 — ÉVALUATION                      │
│                                                             │
│  test_questions.yaml  ──►  RAGEvaluator  ──►  RAGAS Score  │
│  (10 questions + GT)     (3 métriques LLM-as-judge)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Technologies Utilisées

| Composant | Technologie | Pourquoi |
|-----------|-------------|----------|
| Extraction PDF | **Unstructured** (strategy="fast") | Gère texte, tableaux et figures ; plus robuste que pdfplumber |
| Découpage | **LangChain RecursiveCharacterTextSplitter** | Coupe intelligemment aux phrases/paragraphes |
| Embeddings | **paraphrase-multilingual-mpnet-base-v2** (768d) | Multilingue, excellent pour le français |
| Index vectoriel | **FAISS IndexFlatIP** (cosine) | Recherche sémantique ultra-rapide |
| Recherche lexicale | **BM25** (rank_bm25) | Capture les mots-clés exacts (numéros d'articles, etc.) |
| LLM | **Ollama llama3.1:8b** | Tourne en local, pas de coût API, modèle multilingue fort |
| API | **FastAPI** | Framework Python moderne, auto-documentation |
| Évaluation | **RAGAS custom** (LLM-as-judge) | Adapté au corpus juridique français |

---

## 4. Structure du Projet

```
RAG benchmark_RAG-ClassiqueV0/
│
├── config.yaml                   # Configuration centralisée
├── api.py                        # Serveur FastAPI (point d'entrée HTTP)
├── ingest_documents.py           # Script d'ingestion des PDFs
├── run_evaluation.py             # Script d'évaluation RAGAS
├── test_questions.yaml           # 10 questions de test + vérités terrain
│
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py    # Chargement et extraction des PDFs
│   │   ├── chunker.py            # Découpage en chunks
│   │   └── embedder.py           # Génération des embeddings
│   │
│   ├── retrieval/
│   │   └── vector_store.py       # Index FAISS + BM25, recherche hybride
│   │
│   ├── generation/
│   │   └── llm_interface.py      # Interface Ollama, prompt RAG
│   │
│   ├── evaluation/
│   │   ├── metrics.py            # Context Precision, Recall, Faithfulness
│   │   └── evaluator.py          # Orchestrateur de l'évaluation
│   │
│   └── utils/
│       └── helpers.py            # Fonctions utilitaires (config, logging)
│
├── data/
│   └── vector_store/             # Index FAISS persistant (faiss_index.index)
│
├── données rag/                  # PDFs sources à ingérer
│
└── logs/
    └── evaluation_report.json    # Résultats du dernier run RAGAS
```

---

## 5. Flux d'Exécution

### 5.1 — Ingestion (une seule fois, ou quand de nouveaux PDFs arrivent)

```bash
python ingest_documents.py
# ou
run_ingest.bat
```

1. Lit tous les PDFs du dossier `données rag/`
2. Extrait le texte (tableaux balisés `[TABLEAU]...[/TABLEAU]`)
3. Découpe en chunks de 1000 caractères (200 de chevauchement)
4. Calcule les embeddings (768 dimensions)
5. Sauvegarde l'index FAISS sur disque dans `data/vector_store/`

### 5.2 — Requête via API

```bash
run_api.bat        # démarre le serveur sur http://localhost:8000
```

Puis POST `/query` avec `{"query": "Votre question juridique"}`.

### 5.3 — Évaluation

```bash
python run_evaluation.py
# ou
run_evaluation.bat
```

Évalue les 10 questions de `test_questions.yaml` et génère `logs/evaluation_report.json`.

---

## 6. Configuration Principale (`config.yaml`)

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `chunk_size` | 1000 | Taille des chunks en caractères |
| `chunk_overlap` | 200 | Chevauchement entre chunks voisins |
| `top_k` | 8 | Nombre de chunks récupérés par requête |
| `similarity_threshold` | 0.25 | Score minimum pour inclure un chunk |
| `hybrid_alpha` | 0.5 | Poids FAISS / BM25 (0.5 = équilibre) |
| `hybrid_candidate_factor` | 5 | Facteur de sur-récupération FAISS avant fusion |
| `model_name` | paraphrase-multilingual-mpnet-base-v2 | Modèle d'embedding |
| `llm.model` | llama3.1:8b | Modèle LLM via Ollama |
| `temperature` | 0.2 | Faible = réponses déterministes et précises |
| `max_tokens` | 800 | Longueur maximale de la réponse générée |

---

## 7. Métriques d'Évaluation (RAGAS)

Le système est évalué avec trois métriques inspirées de RAGAS, toutes utilisant le LLM comme juge :

| Métrique | Description | Ce qu'elle mesure |
|----------|-------------|-------------------|
| **Context Precision** | Les chunks pertinents sont-ils bien classés en haut ? | Qualité du classement de la recherche |
| **Context Recall** | Le contexte couvre-t-il la vérité terrain ? | Complétude de la récupération |
| **Faithfulness** | La réponse est-elle fidèle aux chunks ? | Absence d'hallucinations |
| **RAGAS Score** | Moyenne harmonique des 3 métriques | Score global |

**Score actuel après 4 itérations d'amélioration : ~0.63 / 1.0**

---

## 8. Dossier `explications/`

Chaque fichier de ce dossier documente un module spécifique :

| Fichier | Module documenté |
|---------|-----------------|
| `document_loader_explication.md` | `src/ingestion/document_loader.py` |
| `chunker_explication.md` | `src/ingestion/chunker.py` |
| `embedder_explication.md` | `src/ingestion/embedder.py` |
| `vector_store_explication.md` | `src/retrieval/vector_store.py` |
| `llm_interface_explication.md` | `src/generation/llm_interface.py` |
| `metrics_explication.md` | `src/evaluation/metrics.py` |
| `evaluator_explication.md` | `src/evaluation/evaluator.py` |
| `api_explication.md` | `api.py` |
| `helpers_explication.md` | `src/utils/helpers.py` |
| `ingest_documents_explication.md` | `ingest_documents.py` |
