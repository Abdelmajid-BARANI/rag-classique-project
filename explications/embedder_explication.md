# Embedder — Explication

**Fichier :** `src/ingestion/embedder.py`  
**Rôle :** Transformer du texte en vecteurs numériques (embeddings) pour la recherche sémantique.

---

## 1. Qu'est-ce qu'un Embedding ?

Un embedding est une **représentation numérique d'un texte** sous forme d'un vecteur de nombres réels. L'idée fondamentale : deux textes qui ont un sens similaire auront des vecteurs proches dans l'espace mathématique.

```
"Facturation électronique obligatoire"  →  [0.12, -0.34, 0.87, ..., 0.45]  (768 valeurs)
"E-invoicing requis par la loi"         →  [0.11, -0.36, 0.85, ..., 0.43]  (similaire !)
"Le chat mange une souris"              →  [-0.23, 0.67, -0.12, ..., 0.91]  (différent)
```

---

## 2. Modèle : `paraphrase-multilingual-mpnet-base-v2`

### Pourquoi ce modèle ?

| Critère | Valeur |
|---------|--------|
| Dimensions | **768** |
| Langues | **50+ langues**, dont le français |
| Architecture | MPNet (supérieur à BERT pour la similarité sémantique) |
| Poids | ~278 Mo |
| Spécialité | **Paraphrases** — excellente pour trouver des formulations équivalentes |

Le modèle a été entraîné spécifiquement pour reconnaître quand deux phrases disent la même chose avec des mots différents — parfait pour les textes juridiques qui reformulent souvent les mêmes règles.

### Alternative écartée : `paraphrase-multilingual-MiniLM-L12-v2`
- Plus petit (384d, ~118 Mo), mais moins précis
- Ce projet utilise la version 768d pour une meilleure qualité

---

## 3. Classe `BERTEmbedder`

### Initialisation

```python
embedder = BERTEmbedder(
    model_name="paraphrase-multilingual-mpnet-base-v2",
    device="cpu"  # ou "cuda" si GPU disponible
)
```

Le modèle est chargé en mémoire une seule fois et réutilisé pour toutes les requêtes.

### Méthodes Principales

#### `embed_text(text: str) -> np.ndarray`

Transforme un texte unique en vecteur (768 dimensions).

```python
vector = embedder.embed_text("Quelles sont les obligations de facturation ?")
# → array de shape (768,)
```

#### `embed_batch(texts: List[str]) -> np.ndarray`

Traite plusieurs textes d'un coup (plus efficace que appels successifs).

```python
vectors = embedder.embed_batch(["texte 1", "texte 2", "texte 3"])
# → array de shape (3, 768)
```

#### `embed_chunks(chunks: List[Dict]) -> List[Dict]`

Enrichit chaque chunk avec son vecteur d'embedding :

```python
# Entrée
{"text": "Art. 289 bis...", "chunk_id": "cgi_chunk_0", ...}

# Sortie (chunk enrichi)
{"text": "Art. 289 bis...", "chunk_id": "cgi_chunk_0", "embedding": array([...]), ...}
```

---

## 4. Normalisation L2

Avant d'être stockés dans FAISS, les vecteurs sont **normalisés L2** (divisés par leur norme). Cela transforme la **distance euclidienne** en **similarité cosinus** :

$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||}$$

Après normalisation : $||A|| = ||B|| = 1$, donc $\text{cosine\_similarity} = A \cdot B$ (produit scalaire).

La similarité cosinus est plus robuste que la distance euclidienne pour les textes de longueurs différentes.

---

## 5. Position dans le Pipeline

### Lors de l'Ingestion

```
Chunker  ──►  [Embedder]  ──►  FAISSVectorStore
             "embed_chunks()"   sauvegarde l'index
```

### Lors d'une Requête

```
Question  ──►  [Embedder]  ──►  FAISSVectorStore.hybrid_search()
              "embed_text()"
```

Le même modèle est utilisé dans les deux directions — la cohérence est essentielle pour que les vecteurs soient comparables.

---

## 6. Performances

| Opération | Temps approximatif (CPU) |
|-----------|--------------------------|
| Charger le modèle | ~5 secondes (une seule fois au démarrage) |
| Encoder 1 texte | ~50-100 ms |
| Encoder 1000 chunks | ~30-60 secondes (batch) |

L'encodage se fait en batch pour minimiser le temps total lors de l'ingestion.
