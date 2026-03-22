# Métriques d'Évaluation — Explication

**Fichier :** `src/evaluation/metrics.py`  
**Rôle :** Calculer les trois métriques RAGAS (Context Precision, Context Recall, Faithfulness) en utilisant le LLM comme juge.

---

## 1. Approche : LLM-as-Judge

Au lieu d'utiliser des métriques automatiques simples (comme le BLEU ou le F1 textuel), ce projet utilise le **LLM comme juge**. Le LLM (llama3.1:8b) est invité à évaluer la qualité des réponses — c'est l'approche "LLM-as-judge" popularisée par RAGAS.

### Avantage

Les métriques automatiques comme BLEU mesurent la similarité lexicale. Le LLM-as-judge comprend le sens — il reconnaît qu'une paraphrase est équivalente à l'original.

### Contrainte

Chaque évaluation nécessite plusieurs appels LLM (lents sur CPU). Les métriques utilisent donc des **prompts batchés** : un seul appel LLM pour évaluer tous les chunks en même temps.

---

## 2. Constantes Clés

| Constante | Valeur | Rôle |
|-----------|--------|------|
| `CHUNK_PREVIEW_LEN` | 1000 chars | Longueur de chunk envoyée au juge (= chunk complet) |
| `MAX_CONTEXT_LEN` | 10 000 chars | Longueur max du contexte concaténé |
| `JUDGE_MAX_TOKENS` | 500 tokens | Limite de réponse du LLM-juge |
| `JUDGE_TIMEOUT` | 600 secondes | Timeout pour les appels d'évaluation |
| `DECOMPOSE_MAX_TOKENS` | 400 tokens | Pour décomposer la ground truth en énoncés |

---

## 3. Métrique 1 : Context Precision

### Définition

**Les chunks retournés sont-ils vraiment pertinents pour la question ?**

Score 1.0 = tous les chunks pertinents sont classés avant les non-pertinents  
Score 0.0 = aucun chunk pertinent

### Calcul (1 seul appel LLM)

```
Prompt → "Pour cette question, chacun de ces passages est-il pertinent ?"
         Question: ...
         Réponse attendue: ...
         Passage 1: ...
         Passage 2: ...
         ...

Réponse LLM → ["oui", "non", "oui", "oui", "non", "non", "oui", "non"]
```

### Formule (Precision@k pondérée)

$$CP = \frac{\sum_{k=1}^{n} P@k \cdot \text{isRelevant}(k)}{\text{nbRelevant}}$$

Où $P@k = \frac{\text{nbRelevant jusqu'à k}}{k}$

**Exemple :**

| Rank | Chunk | Pertinent ? | P@k | Contribution |
|------|-------|------------|-----|--------------|
| 1 | Chunk A | oui | 1/1 = 1.00 | 1.00 |
| 2 | Chunk B | non | - | - |
| 3 | Chunk C | oui | 2/3 = 0.67 | 0.67 |

Score = (1.00 + 0.67) / 2 = **0.835**

Un chunk pertinent classé en position 1 vaut plus qu'un chunk pertinent classé en position 8.

---

## 4. Métrique 2 : Context Recall

### Définition

**Le contexte récupéré couvre-t-il toutes les informations de la vérité terrain ?**

Score 1.0 = tout ce que la ground truth dit est présent dans les chunks  
Score 0.0 = le contexte ne contient rien de la ground truth

### Calcul (2 appels LLM)

**Appel 1 — Décomposition :**
```
Prompt → "Décompose cette réponse attendue en énoncés factuels courts"
Ground Truth: "L'entrée en vigueur est le 31 décembre 2023 (LOI n°2023-1322)"

→ ["L'entrée en vigueur est le 31 décembre 2023.",
   "La loi concernée est la LOI n°2023-1322."]
```

**Appel 2 — Vérification :**
```
Prompt → "Chacun de ces énoncés est-il soutenu par le contexte ?"
Contexte: [chunks 1 à 8 concaténés]
Énoncé 1: "L'entrée en vigueur est le 31 décembre 2023."
Énoncé 2: "La loi concernée est la LOI n°2023-1322."

→ ["oui", "oui"]
```

**Score = 2/2 = 1.0**

---

## 5. Métrique 3 : Faithfulness (Fidélité)

### Définition

**La réponse du RAG est-elle fidèle au contexte fourni ? Contient-elle des hallucinations ?**

Score 1.0 = chaque affirmation de la réponse est vérifiable dans les chunks  
Score 0.0 = la réponse invente des informations absentes du contexte

### Calcul (2 appels LLM)

**Appel 1 — Extraction des affirmations :**
```
Prompt → "Extrait les affirmations factuelles de cette réponse"
Réponse RAG: "La facturation électronique est obligatoire depuis le 01/01/2024.
             Selon l'article 289 bis, les PDP doivent être certifiées ISO 27001."

→ ["La facturation électronique est obligatoire depuis le 01/01/2024.",
   "Les PDP doivent être certifiées ISO 27001 selon l'article 289 bis."]
```

**Appel 2 — Vérification dans les chunks :**
```
→ ["oui", "oui"]
```

**Score = 2/2 = 1.0**

---

## 6. Score RAGAS Global

```python
ragas_score = (3 × CP × CR × F) / (CP + CR + F)  # Moyenne harmonique
```

La moyenne harmonique pénalise fortement les cas où une seule métrique est nulle. Si Faithfulness = 0 (hallucination), le score RAGAS tombe à 0 même si CP et CR sont bons.

---

## 7. Robustesse du Parsing

Le LLM ne retourne pas toujours du JSON propre. Les fonctions `_parse_statements()` et `_parse_verdicts()` implémentent plusieurs niveaux de fallback :

**`_parse_statements()` — 4 niveaux :**
1. JSON array de strings `["...", "..."]`
2. JSON avec dicts `[{"texte": "..."}, ...]`
3. Regex sur les accolades `{...}`
4. Lignes numérotées / tirets (texte libre)

**`_parse_verdicts()` — 4 niveaux :**
1. JSON array `["oui", "non", "oui"]`
2. Lignes numérotées `1. oui\n2. non`
3. Séparateurs virgule `oui, non, oui`
4. Regex globale `\b(oui|non|yes|no)\b`

---

## 8. Paramétrage des Prompts de Jugement

Les prompts de jugement utilisent des formulations **permissives** pour éviter les faux négatifs :

- "même formulée différemment ou partiellement" → si la GT dit "A" et le chunk dit "A et B", c'est "oui"
- "non uniquement si totalement absente" → le LLM ne doit pas dire "non" juste parce que la formulation diffère

This calibration is critical: prompts trop stricts → scores artificiellement bas ; prompts trop permissifs → scores artificiellement hauts.
