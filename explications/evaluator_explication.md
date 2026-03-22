# Evaluator — Explication

**Fichier :** `src/evaluation/evaluator.py`  
**Rôle :** Orchestrer l'évaluation RAGAS complète d'un pipeline RAG.

---

## 1. Rôle dans l'Architecture

L'`evaluator.py` est le chef d'orchestre de l'évaluation. Il ne calcule pas lui-même les métriques — il délègue à `metrics.py` — mais il gère :

- Le flux d'évaluation question par question
- L'appel au pipeline RAG (retrieval + génération) si nécessaire
- L'agrégation des résultats
- La génération du rapport JSON final

---

## 2. Classe `RAGEvaluator`

### Initialisation

```python
evaluator = RAGEvaluator(
    llm=ollama_llm,          # Pour la génération ET le jugement
    embedder=bert_embedder,  # Pour encoder les questions
    vector_store=faiss_store # Pour récupérer les chunks
)
```

Note : `embedder` et `vector_store` sont optionnels. Ils ne sont nécessaires que si on évalue une question sans réponse préalable (mode "pipeline complet").

---

## 3. Mode d'Évaluation : `evaluate_single()`

Évalue **une seule question** avec une réponse et des chunks déjà fournis.

### Entrées

```python
result = evaluator.evaluate_single(
    question="Quelles sont les obligations des PDP ?",
    answer="Selon [Document 1], les PDP doivent...",
    context_chunks=[{"text": "...", "chunk_id": "..."}, ...],
    ground_truth="Les PDP doivent obtenir une certification ISO/IEC 27001...",
    compute_precision=True,
    compute_recall=True,
    compute_faithfulness=True
)
```

### Sorties

```python
{
    "question": "Quelles sont les obligations des PDP ?",
    "answer_preview": "Selon [Document 1], les PDP doivent...",
    "num_context_chunks": 8,
    "ground_truth_provided": True,
    "context_precision": {
        "score": 0.875,
        "num_relevant": 7,
        "total_chunks": 8
    },
    "context_recall": {
        "score": 1.00,
        "statements_supported": 3,
        "total_statements": 3
    },
    "faithfulness": {
        "score": 0.833,
        "claims_supported": 5,
        "total_claims": 6
    },
    "ragas_score": 0.902,
    "evaluation_time_seconds": 245.3
}
```

---

## 4. Mode d'Évaluation en Lot : `evaluate_dataset()`

Évalue toutes les questions d'un fichier YAML.

### Flux d'Exécution

```
Pour chaque question du dataset :
    1. Encoder la question (BERTEmbedder)
    2. Recherche hybride (FAISSVectorStore)
    3. Générer la réponse (OllamaLLM)
    4. Calculer les 3 métriques (metrics.py)
    5. Stocker les résultats

Résumé final :
    - Moyennes par métrique
    - Score RAGAS global
    - Question avec le meilleur/pire score
```

---

## 5. Génération du Rapport : `generate_report()`

Produit un rapport JSON structuré dans `logs/evaluation_report.json` :

```json
{
    "benchmark_date": "2025-01-15T14:32:00",
    "num_questions": 10,
    "config": {
        "top_k": 8,
        "model": "llama3.1:8b",
        "embedding_model": "paraphrase-multilingual-mpnet-base-v2"
    },
    "summary": {
        "avg_context_precision": 0.742,
        "avg_context_recall": 0.685,
        "avg_faithfulness": 0.810,
        "avg_ragas_score": 0.627,
        "best_question": {"id": 5, "score": 1.00},
        "worst_question": {"id": 3, "score": 0.33}
    },
    "results": [
        { "question_id": 1, "question": "...", "scores": {...} },
        ...
    ]
}
```

---

## 6. Décisions de Conception

### Métriques Optionnelles

```python
compute_precision=True
compute_recall=True
compute_faithfulness=True
```

On peut désactiver une métrique pour aller plus vite. Par exemple, si on veut uniquement tester la fidélité de la génération, on met `compute_precision=False, compute_recall=False`.

### Chronométrage

Chaque évaluation est chronométrée. Le temps est inclus dans le rapport. Sur CPU, une évaluation complète de 10 questions prend généralement **30-60 minutes** (3-6 min par question, 5-6 appels LLM chacune).

---

## 7. Position dans le Pipeline

```
run_evaluation.py
      │
      ▼
[RAGEvaluator]
      │
      ├──► BERTEmbedder.embed_text()
      ├──► FAISSVectorStore.hybrid_search()
      ├──► OllamaLLM.generate_with_context()
      │
      └──► ContextPrecision.compute()
      └──► ContextRecall.compute()
      └──► Faithfulness.compute()
      │
      ▼
logs/evaluation_report.json
```
