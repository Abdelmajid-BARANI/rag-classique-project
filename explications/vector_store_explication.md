# Vector Store — Explication

**Fichier :** `src/retrieval/vector_store.py`  
**Rôle :** Stocker les vecteurs et retrouver les chunks les plus pertinents pour une question donnée.

---

## 1. Deux Moteurs de Recherche Combinés

Le `FAISSVectorStore` implémente une **recherche hybride** qui fusionne deux approches complémentaires :

```
Question
   │
   ├──► Recherche Sémantique (FAISS)  ──► "Que dit la loi sur la facturation ?"
   │    (comprend le sens)                 → trouve des paraphrases
   │
   └──► Recherche Lexicale (BM25)     ──► "art. 242 nonies B"
        (cherche les mots exacts)          → trouve le numéro d'article précis
   
        ────────── Fusion ──────────
              Score Hybride Final
```

---

## 2. Composant 1 : FAISS (Recherche Sémantique)

### Qu'est-ce que FAISS ?

**FAISS** (Facebook AI Similarity Search) est une bibliothèque développée par Meta pour la recherche de vecteurs à grande échelle.

### Type d'Index : `IndexFlatIP`

- **Flat** : tous les vecteurs sont stockés et comparés directement (pas d'approximation)
- **IP** : Inner Product (produit scalaire)

Avec des vecteurs normalisés L2, le produit scalaire = similarité cosinus. Cela donne une mesure de similitude entre 0 et 1 (1 = identique, 0 = orthogonal).

### Avantage
Recherche **exacte** (pas approchée) — pour un corpus de quelques milliers de chunks comme nos textes juridiques, c'est suffisamment rapide et garantit de trouver les vrais plus proches voisins.

---

## 3. Composant 2 : BM25 (Recherche Lexicale)

**BM25** (Best Match 25) est l'algorithme de référence pour la recherche textuelle (utilisé par Elasticsearch, Solr, etc.).

### Principe
BM25 score un document en fonction de la fréquence des mots de la requête dans le document, avec deux corrections :
1. **Saturation de fréquence** : un mot qui apparaît 100 fois ne vaut pas 100× plus qu'un mot qui apparaît 1 fois
2. **Normalisation par longueur** : un document long ne bénéficie pas d'un avantage artificiel

### Pourquoi c'est crucial pour notre corpus ?

Les textes juridiques contiennent des **identifiants très spécifiques** :
- `art. 242 nonies B`, `art. 289 bis`, `ISO/IEC 27001`, `Directive 2010/45/UE`

La recherche sémantique peut rater ces références exactes si la formulation varie. BM25 les trouve toujours car il cherche l'occurrence exacte du token.

---

## 4. Recherche Hybride : Fusion des Scores

### Algorithme (méthode `hybrid_search`)

```
1. FAISS récupère top_k × candidate_factor candidats sémantiques
   (ex: 8 × 5 = 40 candidats)

2. BM25 calcule les scores pour TOUS les chunks

3. Union des candidats FAISS + top BM25

4. Normalisation min-max de chaque score en [0, 1]

5. Score hybride = α × score_sémantique + (1-α) × score_bm25
                    (0.5)                      (0.5)

6. Tri décroissant → top_k résultats finaux (= 8)
```

### Paramètre α (alpha)

| Valeur α | Comportement |
|---------|--------------|
| 1.0 | 100% sémantique (FAISS pur) |
| 0.5 | **Équilibre** (valeur actuelle) |
| 0.0 | 100% lexical (BM25 pur) |

Avec `alpha=0.5`, le système est robuste à la fois pour les questions exprimées différemment des textes ET pour les requêtes contenant des numéros d'articles précis.

---

## 5. Structure du Stockage

Les chunks et l'index sont persistés sur disque :

```
data/vector_store/
├── faiss_index.index   # Index binaire FAISS (vecteurs)
├── chunks.json         # Métadonnées de chaque chunk (texte, chunk_id, source)
└── config.json         # Paramètres utilisés lors de la création
```

### Chargement au Démarrage de l'API

```python
store = FAISSVectorStore(embedder)
store.load("data/vector_store/faiss_index")
# → prêt en quelques secondes
```

---

## 6. Méthodes Principales

| Méthode | Description |
|---------|-------------|
| `add_chunks(chunks)` | Indexe une liste de chunks (avec leurs embeddings) |
| `search(query_embedding, top_k)` | Recherche sémantique FAISS seule |
| `bm25_search(query_text, top_k)` | Recherche lexicale BM25 seule |
| `hybrid_search(embedding, text, top_k, alpha)` | **Recherche hybride** (principale) |
| `save(filename)` | Sauvegarde sur disque |
| `load(filename)` | Chargement depuis disque |

---

## 7. Format de Sortie

Chaque chunk retourné par la recherche contient :

```python
{
    "text": "Art. 289 bis CGI — La facturation électronique est...",
    "chunk_id": "cgi_289bis_chunk_3",
    "score": 0.847,            # score hybride normalisé
    "semantic_score": 0.923,   # score FAISS brut (cosine)
    "bm25_score": 14.2,        # score BM25 brut
    "rank": 1,                 # position dans les résultats
    "metadata": { "source": "...", "filename": "..." }
}
```

---

## 8. Position dans le Pipeline

```
Embedder  ──►  [FAISSVectorStore]  ──►  OllamaLLM
               (stocke + recherche)    (génère la réponse)
```
