# LLM Interface — Explication

**Fichier :** `src/generation/llm_interface.py`  
**Rôle :** Interagir avec le modèle de langage Ollama pour générer des réponses à partir du contexte récupéré.

---

## 1. Architecture : Ollama en Local

Le projet utilise **Ollama** pour faire tourner le LLM sur la machine locale, sans dépendance à une API cloud.

```
         HTTP Request
API.py ──────────────►  Ollama (localhost:11434)
                              │
                         llama3.1:8b
                              │
                         ◄────────────
                         HTTP Response
```

### Avantages d'Ollama

- **Confidentialité** : les documents juridiques ne quittent pas la machine
- **Pas de coût** : aucun appel API payant
- **Latence maîtrisée** : pas de dépendance réseau
- **Modèle puissant** : `llama3.1:8b` excelle en français

---

## 2. Modèle : `llama3.1:8b`

| Propriété | Valeur |
|-----------|--------|
| Paramètres | 8 milliards |
| Architecture | Transformer (Meta) |
| Langues | Multilingue (excellent en français) |
| Fenêtre contexte | 128K tokens |
| RAM requise | ~5-6 Go |

---

## 3. Classe `OllamaLLM`

### Initialisation

```python
llm = OllamaLLM(
    model="llama3.1:8b",
    host="http://localhost:11434"
)
```

Au démarrage, le constructeur vérifie que Ollama est accessible et que le modèle est téléchargé.

### Méthode Principale : `generate(prompt, temperature, max_tokens)`

| Paramètre | Valeur par défaut | Usage dans RAG |
|-----------|-----------------|---------------|
| `temperature` | 0.7 (défaut LLM) | **0.2** (réponses déterministes) |
| `max_tokens` | 512 | **800** (réponses complètes) |
| `timeout` | 600 secondes | Pour les requêtes lentes sur CPU |

#### `temperature=0.2`

Une faible température rend le modèle **conservateur et précis**. Pour des questions juridiques, on veut des réponses factuelles, pas créatives. Cela réduit aussi le risque d'hallucination.

#### `max_tokens=800`

800 tokens ≈ 600 mots, suffisant pour une réponse complète sur un article de loi avec citations.

---

## 4. Le Prompt RAG

La méthode `generate_with_context()` construit automatiquement un **prompt structuré** avant d'appeler le LLM.

### Structure du Prompt

```
[INSTRUCTIONS STRICTES]
- Commence par répondre directement en une phrase courte
- Développe avec citations [Document N]
- Si article abrogé : signale la date et l'absence de successeur
- Ne répète pas deux fois la même info
- Si l'info n'est pas dans le contexte : "L'information n'est pas présente..."
- Sois complet, ne tronque jamais ta réponse

[CONTEXTE]
[Document 1]: Art. 289 bis I. — Les assujettis...
[Document 2]: Art. 242 nonies B — Six mois avant...
...

[QUESTION]
Quelles sont les conditions pour être PDP ?
```

### Règles du Prompt (détail)

| Règle | Raison |
|-------|--------|
| **Réponse directe d'abord** | Évite les introductions longues du type "Voici les informations..." |
| **Citations obligatoires** | Traçabilité : on sait d'où vient chaque information |
| **Articles abrogés** | Le corpus contient des articles anciens et nouveaux — évite la confusion |
| **Anti-redondance** | Sans cette règle, llama3.1 répète souvent les mêmes infos 2-3 fois |
| **"L'information n'est pas présente..."** | Si le contexte ne couvre pas la question, le LLM doit le dire clairement au lieu d'inventer |

---

## 5. Système de Retry

```python
MAX_RETRIES = 2
RETRY_DELAY = 2  # secondes

for attempt in range(MAX_RETRIES + 1):
    try:
        response = requests.post(api_url, ...)
        return response
    except ConnectionError:
        wait(RETRY_DELAY)
        continue
```

Si Ollama est temporairement surchargé (CPU 100%), le client réessaie 2 fois avant de lever une exception.

---

## 6. `keep_alive: "10m"`

Le paramètre `keep_alive: "10m"` indique à Ollama de garder le modèle chargé en mémoire pendant 10 minutes après la dernière requête. Sans ça, Ollama décharge le modèle après chaque requête, ce qui provoque un temps de chargement de 30-60 secondes à chaque appel.

---

## 7. Position dans le Pipeline

```
FAISSVectorStore.hybrid_search()
        │
        ▼ (top_k chunks)
[OllamaLLM.generate_with_context()]
        │
        ▼
    Réponse finale
```

Le LLM est le dernier composant actif de la chaîne : il reçoit les chunks pertinents et produit la réponse en langage naturel.
