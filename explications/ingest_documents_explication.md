# Script d'Ingestion — Explication

**Fichier :** `ingest_documents.py`  
**Rôle :** Script principal qui lit les PDFs, les transforme en chunks vectorisés et les indexe dans FAISS.

---

## 1. Qu'est-ce que l'Ingestion ?

L'ingestion est la **phase préparatoire** du système RAG. Elle se fait **une seule fois** (ou quand de nouveaux documents sont ajoutés) et produit l'index FAISS qui sera utilisé par l'API en temps réel.

```
PDFs bruts  ──(ingestion)──►  Index FAISS  ──(API)──►  Réponses
```

---

## 2. Lancement

```bash
python ingest_documents.py
# ou
run_ingest.bat
```

Durée typique : **2-5 minutes** pour un corpus de plusieurs dizaines de PDFs juridiques (sur CPU).

---

## 3. Pipeline en 5 Étapes

### Étape 1 — Chargement de la Configuration
```python
config = load_config("config.yaml")
```

Lit tous les paramètres : dossier source, taille des chunks, modèle d'embedding, dossier de sauvegarde.

### Étape 2 — Chargement des PDFs (`DocumentLoader`)
```python
loader = DocumentLoader()
documents = loader.load_all_pdfs(pdf_dir="données rag/")
```

Pour chaque PDF du dossier `données rag/` :
- Extrait le texte avec Unstructured (strategy="fast")
- Balise les tableaux `[TABLEAU]...[/TABLEAU]`
- Retourne une liste de documents `{filename, content, metadata}`

### Étape 3 — Découpage en Chunks (`DocumentChunker`)
```python
chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_documents(documents)
```

Découpe chaque document en morceaux de ~1000 caractères avec 200 de chevauchement.

**Log typique :**
```
INFO | 15 documents chargés → 142 chunks créés
```

### Étape 4 — Génération des Embeddings (`BERTEmbedder`)
```python
embedder = BERTEmbedder(model_name="paraphrase-multilingual-mpnet-base-v2")
chunks_with_embeddings = embedder.embed_chunks(chunks)
```

Calcule un vecteur de 768 dimensions pour chaque chunk (en batch pour l'efficacité).

**Log typique :**
```
INFO | Génération de 142 embeddings...
INFO | ████████████████ 100% — Terminé en 28.4s
```

### Étape 5 — Indexation et Sauvegarde (`FAISSVectorStore`)
```python
store = FAISSVectorStore(embedding_dim=768)
store.add_chunks(chunks_with_embeddings)
store.save("./data/vector_store/faiss_index")
```

- Crée l'index FAISS (IndexFlatIP)
- Construit l'index BM25 sur les mêmes chunks
- Sauvegarde sur disque dans `data/vector_store/`

---

## 4. Fichiers Générés

```
data/vector_store/
├── faiss_index.index   # Index binaire de vecteurs FAISS
├── chunks.json         # Textes et métadonnées de chaque chunk
└── config.json         # Config utilisée lors de l'ingestion
```

Ces fichiers sont chargés à chaque démarrage de l'API en quelques secondes.

---

## 5. Quand Relancer l'Ingestion ?

| Situation | Relancer |
|-----------|----------|
| Nouveaux PDFs ajoutés dans `données rag/` | ✅ Oui |
| Changement de `chunk_size` ou `chunk_overlap` | ✅ Oui |
| Changement de modèle d'embedding | ✅ Oui |
| Modification du prompt LLM | ✗ Non (pas de lien avec l'index) |
| Modification des paramètres de recherche | ✗ Non (chargés depuis config au runtime) |

---

## 6. Log de Succès

```
INFO  | Démarrage du pipeline d'ingestion
INFO  | Dossier source: données rag/
INFO  | 3 fichiers PDF trouvés
INFO  | Chargement: CGI_article_289.pdf (42 pages)
INFO  | Chargement: directive_2010_45.pdf (28 pages)
INFO  | Chargement: annexe_decret_2023.pdf (15 pages)
INFO  | 15 documents → 142 chunks
INFO  | Génération des embeddings (paraphrase-multilingual-mpnet-base-v2)...
INFO  | Embeddings générés en 28.4 secondes
INFO  | Indexation FAISS...
SUCCESS | Index FAISS sauvegardé: data/vector_store/faiss_index.index
SUCCESS | Ingestion terminée: 142 chunks indexés
```

---

## 7. Position dans l'Architecture

```
données rag/         ← PDFs sources
     │
     ▼
ingest_documents.py  ← CE SCRIPT
     │
     ├── DocumentLoader
     ├── DocumentChunker
     ├── BERTEmbedder
     └── FAISSVectorStore.save()
     │
     ▼
data/vector_store/   ← Index persistant
     │
     ▼
api.py (FAISSVectorStore.load() au démarrage)
```
