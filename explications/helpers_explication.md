# Helpers (Utils) — Explication

**Fichier :** `src/utils/helpers.py`  
**Rôle :** Fournir des fonctions utilitaires partagées par tous les modules du projet.

---

## 1. Fonctions Disponibles

### `load_config(config_path: str) -> Dict`

Charge la configuration depuis un fichier YAML.

```python
config = load_config("config.yaml")
# → {"ingestion": {"chunk_size": 1000, ...}, "retrieval": {...}, ...}
```

**Utilisation :** appelé au démarrage par `api.py`, `run_evaluation.py` et `ingest_documents.py`.

---

### `setup_logging(log_level: str, log_file: str = None)`

Configure le système de logging **Loguru** (une seule fois — idempotent).

```python
setup_logging(log_level="INFO", log_file="logs/app.log")
```

**Format des logs :**
```
2025-01-15 14:32:00 | INFO     | ingestion.embedder:embed_text - Embedding calculé en 52ms
```

La fonction est protégée par un flag `_logging_configured` pour éviter d'ajouter plusieurs fois les mêmes handlers si elle est appelée depuis des modules différents.

---

### `ensure_directories(directories: list)`

Crée les répertoires nécessaires s'ils n'existent pas déjà.

```python
ensure_directories(["./data/vector_store", "./logs"])
```

Appelé au démarrage de l'ingestion pour s'assurer que les dossiers de sortie existent.

---

## 2. Bibliothèque de Logging : Loguru

Le projet utilise **Loguru** plutôt que le module `logging` standard de Python.

### Avantages

| Feature | logging standard | Loguru |
|---------|-----------------|--------|
| Configuration | Verbose (handlers, formatters, levels) | Une ligne |
| Couleurs console | Non (par défaut) | Oui (automatic) |
| Rotation de fichiers | Manuel | `rotation="100 MB"` |
| Niveaux colorés | Non | `SUCCESS` en vert, `ERROR` en rouge |

### Niveaux Utilisés dans le Projet

| Niveau | Usage |
|--------|-------|
| `logger.debug(...)` | Détails techniques (nb candidats, scores BM25, etc.) |
| `logger.info(...)` | Actions normales (composant initialisé, chunk chargé) |
| `logger.success(...)` | Succès importants (index FAISS chargé, Ollama connecté) |
| `logger.warning(...)` | Problèmes récupérables (modèle pas encore téléchargé) |
| `logger.error(...)` | Erreurs sérieuses (connexion Ollama impossible) |

---

## 3. Centralisation de la Configuration

**Principe :** un seul fichier `config.yaml` est la source de vérité pour tous les paramètres. Les modules lisent la config via `load_config()` plutôt que d'avoir des constantes hardcodées.

```python
# ✓ Bonne pratique (utilisée dans le projet)
config = load_config("config.yaml")
top_k = config["retrieval"]["top_k"]

# ✗ À éviter
top_k = 8  # hardcodé, difficile à changer
```

Cela permet de modifier un paramètre (ex: `top_k=10`) sans toucher au code source.

---

## 4. Position dans le Projet

Le module `utils` est transversal — il n'appartient à aucune couche du pipeline RAG spécifiquement, mais est utilisé par toutes :

```
utils/helpers.py
     │
     ├──► api.py
     ├──► ingest_documents.py
     ├──► run_evaluation.py
     ├──► src/ingestion/*
     ├──► src/retrieval/*
     ├──► src/generation/*
     └──► src/evaluation/*
```
