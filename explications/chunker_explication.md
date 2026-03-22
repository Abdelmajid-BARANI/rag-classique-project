# Chunker — Explication

**Fichier :** `src/ingestion/chunker.py`  
**Rôle :** Découper le texte extrait des PDFs en morceaux (chunks) adaptés à la recherche sémantique.

---

## 1. Pourquoi Découper ?

Les modèles d'embeddings et les LLMs ont une **fenêtre de contexte limitée**. On ne peut pas envoyer un document de 100 pages entier. Le découpage résout deux problèmes :

1. **Taille** : chaque morceau tient dans le contexte du modèle d'embedding (limite : ~512 tokens)
2. **Précision** : un petit morceau ciblé est plus pertinent qu'un grand document entier

---

## 2. Algorithme : RecursiveCharacterTextSplitter

Le chunker utilise `RecursiveCharacterTextSplitter` de **LangChain** avec ces séparateurs (dans l'ordre de priorité) :

```
\n\n  →  double saut de ligne (= séparation entre paragraphes)
\n    →  saut de ligne simple
.     →  fin de phrase
      →  espace (dernier recours)
```

**Logique récursive :** si un paragraphe est trop grand, il est coupé à la phrase. Si la phrase est trop grande, il est coupé à l'espace.

### Avantage pour les textes juridiques

Les textes du CGI sont structurés en paragraphes et alinéas bien délimités par des sauts de ligne. Le découpeur préserve naturellement les articles entiers quand ils sont courts, et ne coupe jamais un article en plein milieu d'une règle.

---

## 3. Paramètres de Configuration

| Paramètre | Valeur | Explication |
|-----------|--------|-------------|
| `chunk_size` | **1000** caractères | Environ 200 mots — un article court ou un paragraphe |
| `chunk_overlap` | **200** caractères | Chevauchement entre chunks voisins |

### Rôle du chevauchement (overlap)

```
Chunk 1 : [==================================|---]
Chunk 2 :                              [---|==================================]
                                       ↑ 200 chars partagés
```

Le chevauchement garantit que si une information importante se trouve à la jonction de deux paragraphes, elle apparaît dans les deux chunks. Sans chevauchement, une question portant sur cette jonction ne trouverait rien.

---

## 4. Format de Sortie

Chaque chunk est un dictionnaire :

```python
{
    "text": "Texte du morceau (max 1000 caractères)...",
    "chunk_id": "nom_du_fichier_chunk_0",   # identifiant unique
    "n_chars": 847,                          # longueur effective
    "metadata": {
        "source": "/chemin/fichier.pdf",
        "filename": "CGI_article_289.pdf",
        "chunk_index": 0,                    # position dans le document
        "total_chunks": 15
    }
}
```

---

## 5. Exemple Concret

Document source (2 500 caractères) → découpage en chunks :

```
Chunk 0 (chars 0-1000)       : En-tête de l'article 289 bis + I.
Chunk 1 (chars 800-1800)     : I. (fin) + II. (début)   ← 200 chars en commun avec Chunk 0
Chunk 2 (chars 1600-2500)    : II. (fin) + III.          ← 200 chars en commun avec Chunk 1
```

---

## 6. Interface Publique

### `chunk_document(document: Dict) -> List[Dict]`

Découpe un seul document en chunks.

### `chunk_documents(documents: List[Dict]) -> List[Dict]`

Traite une liste de documents et retourne tous leurs chunks.

---

## 7. Position dans le Pipeline

```
DocumentLoader  ──►  [Chunker]  ──►  Embedder  ──►  FAISS
```

Le Chunker reçoit des textes structurés (avec balises `[TABLEAU]`) et produit des unités prêtes à être vectorisées.
