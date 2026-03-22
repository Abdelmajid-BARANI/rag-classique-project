# API — Explication

**Fichier :** `api.py`  
**Rôle :** Exposer le pipeline RAG sous forme d'une API HTTP REST via FastAPI.

---

## 1. Framework : FastAPI

**FastAPI** est un framework Python moderne basé sur les standards OpenAPI et JSON Schema. Il fournit automatiquement :

- **Documentation interactive** sur `http://localhost:8000/docs` (Swagger UI)
- **Validation automatique** des paramètres d'entrée via Pydantic
- **Performances élevées** (basé sur Starlette + asyncio)

---

## 2. Démarrage du Serveur

```bash
run_api.bat          # Windows — démarre sur port 8000
# ou
uvicorn api:app --reload --port 8000
```

### Initialisation au Démarrage (`lifespan`)

Le décorateur `@asynccontextmanager lifespan` exécute du code **avant** de recevoir les premières requêtes :

1. Charge `BERTEmbedder` (modèle sentence-transformer en mémoire)
2. Charge `FAISSVectorStore` depuis `data/vector_store/` (index FAISS + BM25)
3. Connecte `OllamaLLM` à `localhost:11434`
4. Lit les paramètres depuis `config.yaml` (`top_k`, `alpha`, `threshold`, etc.)

Si Ollama n'est pas disponible, l'API démarre quand même — les requêtes `/query` retourneront une erreur 503 mais `/health` et `/stats` fonctionnent.

---

## 3. Endpoints

### `GET /health` — Santé du Système

```json
{
    "status": "healthy",
    "ollama_connected": true,
    "vector_store_loaded": true,
    "num_documents": 142,
    "model": "llama3.1:8b"
}
```

À appeler avant de faire des requêtes pour vérifier que tous les composants sont opérationnels.

---

### `GET /stats` — Statistiques

```json
{
    "num_chunks": 142,
    "embedding_dim": 768,
    "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
    "llm_model": "llama3.1:8b",
    "top_k": 8,
    "search_mode": "hybrid",
    "hybrid_alpha": 0.5,
    "similarity_threshold": 0.25
}
```

---

### `POST /query` — Requête RAG (Principal)

**Corps de la requête :**
```json
{
    "query": "Quelles sont les conditions pour qu'une plateforme soit agréée PDP ?",
    "top_k": 8,
    "temperature": 0.2,
    "max_tokens": 800
}
```

**Paramètres :**

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `query` | *requis* | La question en langage naturel |
| `top_k` | 8 (config) | Nombre de chunks à récupérer |
| `temperature` | 0.2 | Créativité du LLM (0 = déterministe) |
| `max_tokens` | 800 | Longueur max de la réponse |

**Réponse :**
```json
{
    "query": "Quelles sont les conditions...",
    "answer": "Pour être agréée PDP, une plateforme doit... [Document 1]",
    "context_chunks": [
        {
            "text": "Art. 289 bis — Les plateformes de dématérialisation partenaires...",
            "chunk_id": "cgi_289bis_chunk_2",
            "score": 0.847,
            "semantic_score": 0.923,
            "bm25_score": 14.2,
            "rank": 1,
            "metadata": { "source": "...", "filename": "CGI.pdf" }
        }
    ],
    "num_chunks_retrieved": 8,
    "latency_seconds": 4.2
}
```

**Pipeline interne :**
```
1. Encoder la question → vecteur 768d
2. hybrid_search() → 8 chunks les plus pertinents
3. Filtrer par similarity_threshold (0.25)
4. Construire le prompt RAG avec les chunks
5. OllamaLLM.generate() → réponse
6. Retourner réponse + chunks + latence
```

---

### `POST /evaluate/single` — Évaluer une Requête

Évalue la qualité d'une réponse RAG avec les métriques RAGAS.

**Corps :**
```json
{
    "question": "...",
    "answer": "...",
    "context_chunks": [...],
    "ground_truth": "Réponse attendue...",
    "compute_precision": true,
    "compute_recall": true,
    "compute_faithfulness": true
}
```

**Retourne :** les scores Context Precision, Context Recall, Faithfulness et RAGAS.

---

### `POST /evaluate/dataset` — Évaluer tout le Dataset

Lance l'évaluation complète des 10 questions de `test_questions.yaml`.

**Durée estimée :** 30-60 minutes sur CPU (5-6 appels LLM par question × 10 questions).

**Corps :**
```json
{
    "questions_file": "test_questions.yaml",
    "max_questions": 0,
    "temperature": 0.2,
    "max_tokens": 800
}
```

---

## 4. CORS

L'API autorise les requêtes depuis n'importe quelle origine (`allow_origins=["*"]`). Cela permet d'appeler l'API depuis Postman, depuis un navigateur web ou depuis une autre application.

---

## 5. Validation Automatique

FastAPI valide les requêtes entrantes via les modèles Pydantic :

```python
class QueryRequest(BaseModel):
    query: str                              # requis
    top_k: Optional[int] = Field(None)     # optionnel
    temperature: float = Field(0.2, ge=0.0, le=2.0)  # entre 0 et 2
    max_tokens: int = Field(300, ge=50, le=2000)       # entre 50 et 2000
```

Si `temperature=5.0` est envoyé, FastAPI retourne automatiquement une erreur 422 avec les détails de validation.

---

## 6. Documentation Interactive

Accéder à `http://localhost:8000/docs` pour :
- Voir tous les endpoints avec leur documentation
- Tester les requêtes directement depuis le navigateur (Swagger UI)
- Voir les schémas JSON de requête et réponse
