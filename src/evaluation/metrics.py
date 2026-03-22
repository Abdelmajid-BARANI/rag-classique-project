"""
RAGAS Evaluation Metrics
Implémente les métriques RAGAS pour évaluer un système RAG :
  - Context Precision  : les chunks pertinents sont-ils bien classés ?
  - Context Recall     : le contexte récupéré couvre-t-il la vérité terrain ?
  - Faithfulness       : la réponse est-elle fidèle au contexte récupéré ?
  - RAGAS Score        : score agrégé (moyenne harmonique)

Chaque métrique utilise le LLM (Ollama) comme juge pour l'évaluation.
Toutes les vérifications sont **batchées** en un seul appel LLM par étape
afin de minimiser le nombre de requêtes (crucial pour CPU lent).
"""
import re
import json
import time
from typing import List, Dict, Optional, Any
from loguru import logger


# ---------------------------------------------------------------------------
# Constants – limites pour les prompts
# ---------------------------------------------------------------------------
CHUNK_PREVIEW_LEN = 1000     # caractères par chunk (= chunk_size complet)
MAX_CONTEXT_LEN = 10000      # contexte concaténé total
JUDGE_TIMEOUT = 600           # secondes par appel LLM de jugement
DECOMPOSE_MAX_TOKENS = 400   # pour décomposition / extraction
JUDGE_MAX_TOKENS = 500        # pour les verdicts batchés


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_verdicts(text: str, n_expected: int) -> List[bool]:
    """
    Parse une liste de verdicts depuis la sortie LLM.
    Supporte les formats :
      - JSON array : ["oui", "non", "oui"]
      - Lignes numérotées : 1. oui  2. non  3. oui
      - Séparés par virgule : oui, non, oui
    """
    verdicts: List[bool] = []

    # 1. Tenter JSON array
    json_match = re.search(r'\[.*?\]', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list) and len(parsed) >= n_expected:
                for v in parsed[:n_expected]:
                    verdicts.append(str(v).strip().lower() in ("oui", "yes", "1", "true", "vrai"))
                return verdicts
        except json.JSONDecodeError:
            pass

    # 2. Tenter lignes numérotées / tirets
    lines_with_verdict = []
    for line in text.split("\n"):
        line = line.strip()
        # Retirer le numéro / tiret
        cleaned = re.sub(r'^[\d]+[\.\)\-:\s]+', '', line)
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned).strip().lower()
        if cleaned and len(cleaned) < 50:
            lines_with_verdict.append(cleaned)
    if len(lines_with_verdict) >= n_expected:
        for v in lines_with_verdict[:n_expected]:
            verdicts.append(v.startswith(("oui", "yes", "1", "true", "vrai"))
                           or "oui" in v or "yes" in v)
        return verdicts

    # 3. Fallback : séparés par virgule
    parts = re.split(r'[,;]+', text.lower())
    if len(parts) >= n_expected:
        for v in parts[:n_expected]:
            v = v.strip()
            verdicts.append(v.startswith(("oui", "yes", "1", "true", "vrai"))
                           or "oui" in v or "yes" in v)
        return verdicts

    # 4. Dernier fallback : chercher tous oui/non dans le texte dans l'ordre
    for match in re.finditer(r'\b(oui|non|yes|no)\b', text.lower()):
        verdicts.append(match.group(1) in ("oui", "yes"))
        if len(verdicts) >= n_expected:
            break

    # Pad with False si pas assez
    while len(verdicts) < n_expected:
        verdicts.append(False)

    return verdicts[:n_expected]


def _parse_statements(text: str) -> List[str]:
    """Extrait une liste d'énoncés depuis la sortie LLM."""
    statements = []
    # Tenter d'extraire un tableau JSON
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                clean = []
                for s in parsed:
                    if isinstance(s, dict):
                        # LLM retourne parfois {"texte": "..."} au lieu de "..."
                        val = s.get("texte", s.get("text", ""))
                        if not val:
                            vals = list(s.values())
                            val = vals[0] if vals else str(s)
                        clean.append(str(val).strip())
                    else:
                        clean.append(str(s).strip())
                filtered = [s for s in clean if s and len(s) > 10]
                if filtered:
                    return filtered
        except json.JSONDecodeError:
            pass

        # JSON invalide — LLM retourne [{"text"}, {"text"}] (set-like)
        # Extraire le contenu entre accolades
        items = re.findall(r'\{([^}]+)\}', raw)
        if items:
            for item in items:
                cleaned = item.strip().strip('"\u201c\u201d').strip()
                # Gérer format clé:valeur → "texte": "..."
                kv = re.match(r'["\u201c\u201d]?\w+["\u201c\u201d]?\s*:\s*["\u201c\u201d](.+)["\u201c\u201d]', cleaned, re.DOTALL)
                if kv:
                    cleaned = kv.group(1).strip()
                if cleaned and len(cleaned) > 10:
                    statements.append(cleaned)
            if statements:
                return statements

        # Dernier essai : extraire toutes les chaînes entre guillemets
        quoted = re.findall(r'"([^"]{10,})"', raw)
        if quoted:
            return [s.strip() for s in quoted if s.strip()]

    # Fallback : lignes numérotées ou tirets
    for line in text.split("\n"):
        line = line.strip()
        # Ignorer les lignes d'en-tête du LLM
        if any(skip in line.lower() for skip in ["voici", "les affirmations", "extraits de", "tableau json", "affirmations factuelles"]):
            continue
        line = re.sub(r'^[\d]+[\.\)\-]\s*', '', line)
        line = re.sub(r'^[-•*]\s*', '', line)
        if line and len(line) > 10:
            statements.append(line)
    return statements


def _safe_llm_call(
    llm,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.0,
    timeout: int = JUDGE_TIMEOUT,
    retries: int = 1,
) -> str:
    """Appel LLM avec gestion d'erreur et retry."""
    for attempt in range(retries + 1):
        try:
            return llm.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as e:
            if attempt < retries:
                logger.warning(f"Appel LLM évaluation échoué (tentative {attempt + 1}), retry... : {e}")
                import time as _time
                _time.sleep(2)
                continue
            logger.warning(f"Erreur lors de l'appel LLM pour évaluation : {e}")
            return ""


# ---------------------------------------------------------------------------
# Context Precision  (1 seul appel LLM — prompt batché)
# ---------------------------------------------------------------------------

class ContextPrecision:
    """
    Context Precision — mesure si les chunks pertinents sont bien classés en haut.

    Un seul appel LLM : on soumet TOUS les passages et on demande une liste
    de verdicts oui/non pour chacun.
    """

    name = "context_precision"

    BATCH_PROMPT = """Tu es un évaluateur de systèmes de recherche documentaire.
Pour la question ci-dessous, détermine si CHAQUE passage contient des informations qui aident à répondre.
Un passage est pertinent s'il mentionne le sujet de la question, même partiellement.

Question : {question}
Réponse attendue : {ground_truth}

{passages_block}

Réponds UNIQUEMENT avec une liste JSON de "oui" ou "non", un par passage, dans l'ordre.
Exemple pour 3 passages : ["oui", "non", "oui"]

Liste JSON :"""

    @staticmethod
    def compute(
        question: str,
        context_chunks: List[Dict],
        llm,
        ground_truth: Optional[str] = None,
    ) -> Dict:
        if not context_chunks:
            return {"score": 0.0, "details": [], "num_relevant": 0}

        # Construire le bloc passages
        passages_lines = []
        for i, chunk in enumerate(context_chunks):
            text = chunk.get("text", "")[:CHUNK_PREVIEW_LEN]
            passages_lines.append(f"Passage {i+1} : {text}")
        passages_block = "\n\n".join(passages_lines)

        prompt = ContextPrecision.BATCH_PROMPT.format(
            question=question,
            ground_truth=ground_truth or "Non disponible",
            passages_block=passages_block,
        )
        response = _safe_llm_call(llm, prompt, max_tokens=JUDGE_MAX_TOKENS, timeout=JUDGE_TIMEOUT)
        relevance_flags = _parse_verdicts(response, len(context_chunks))

        details = []
        for i, (chunk, is_rel) in enumerate(zip(context_chunks, relevance_flags)):
            details.append({
                "rank": i + 1,
                "relevant": is_rel,
                "chunk_preview": chunk.get("text", "")[:100],
            })

        # Calcul du score pondéré (Precision@k pondérée)
        num_relevant = sum(relevance_flags)
        if num_relevant == 0:
            score = 0.0
        else:
            weighted_sum = 0.0
            cumulative_relevant = 0
            for k, is_rel in enumerate(relevance_flags, start=1):
                if is_rel:
                    cumulative_relevant += 1
                    precision_at_k = cumulative_relevant / k
                    weighted_sum += precision_at_k
            score = weighted_sum / num_relevant

        return {
            "score": round(score, 4),
            "details": details,
            "num_relevant": num_relevant,
            "total_chunks": len(context_chunks),
            "llm_response": response.strip()[:200],
        }


# ---------------------------------------------------------------------------
# Context Recall  (2 appels LLM : décompose + batch vérification)
# ---------------------------------------------------------------------------

class ContextRecall:
    """
    Context Recall — le contexte récupéré couvre-t-il la vérité terrain ?

    Appel 1 : décomposer la ground truth en énoncés.
    Appel 2 : vérifier d'un coup si chaque énoncé est soutenu par le contexte.
    """

    name = "context_recall"

    DECOMPOSE_PROMPT = """Décompose le texte suivant en une liste d'énoncés factuels courts (phrases indépendantes en français).
Retourne UNIQUEMENT un tableau JSON de chaînes de caractères simples (PAS de dictionnaires, PAS de clés). Maximum 5 énoncés.

Exemple de format attendu : ["La TVA est de 20%.", "L'article 289 traite de la facturation."]

Texte : {ground_truth}

Tableau JSON :"""

    BATCH_SUPPORT_PROMPT = """Étant donné le contexte ci-dessous, détermine si CHAQUE énoncé est soutenu par le contexte.
Réponds "oui" si l'information est présente dans le contexte, même formulée différemment ou partiellement.
Réponds "non" uniquement si l'information est totalement absente du contexte.

Contexte :
{context}

{statements_block}

Réponds UNIQUEMENT avec une liste JSON de "oui" ou "non", un par énoncé, dans l'ordre.
Exemple pour 3 énoncés : ["oui", "non", "oui"]

Liste JSON :"""

    @staticmethod
    def compute(
        question: str,
        context_chunks: List[Dict],
        llm,
        ground_truth: str = "",
    ) -> Dict:
        if not ground_truth:
            logger.warning("Context Recall nécessite une ground_truth. Score = N/A")
            return {"score": None, "statements": [], "details": [], "reason": "ground_truth manquante"}

        if not context_chunks:
            return {"score": 0.0, "statements": [], "details": []}

        # 1. Décomposer la ground truth en énoncés
        decompose_resp = _safe_llm_call(
            llm,
            ContextRecall.DECOMPOSE_PROMPT.format(ground_truth=ground_truth),
            max_tokens=DECOMPOSE_MAX_TOKENS,
            timeout=JUDGE_TIMEOUT,
        )
        statements = _parse_statements(decompose_resp)
        # Limiter à 5 pour garder le prompt court
        statements = statements[:5]

        if not statements:
            logger.warning("Impossible de décomposer la ground_truth en énoncés")
            return {"score": None, "statements": [], "details": [], "reason": "décomposition échouée"}

        # 2. Construire le contexte concaténé
        context_text = "\n\n".join(
            f"[Chunk {i+1}]: {c.get('text', '')[:CHUNK_PREVIEW_LEN]}"
            for i, c in enumerate(context_chunks)
        )
        context_text = context_text[:MAX_CONTEXT_LEN]

        # 3. Batch vérification
        stmts_block = "\n".join(f"Énoncé {i+1} : {s}" for i, s in enumerate(statements))
        prompt = ContextRecall.BATCH_SUPPORT_PROMPT.format(
            context=context_text, statements_block=stmts_block
        )
        response = _safe_llm_call(llm, prompt, max_tokens=JUDGE_MAX_TOKENS, timeout=JUDGE_TIMEOUT)
        support_flags = _parse_verdicts(response, len(statements))

        supported_count = sum(support_flags)
        details = []
        for stmt, is_sup in zip(statements, support_flags):
            details.append({"statement": stmt, "supported": is_sup})

        score = supported_count / len(statements) if statements else 0.0

        return {
            "score": round(score, 4),
            "statements": statements,
            "details": details,
            "supported_count": supported_count,
            "total_statements": len(statements),
            "llm_response": response.strip()[:200],
        }


# ---------------------------------------------------------------------------
# Faithfulness  (2 appels LLM : extraction + batch vérification)
# ---------------------------------------------------------------------------

class Faithfulness:
    """
    Faithfulness — la réponse est-elle fidèle au contexte récupéré ?

    Appel 1 : extraire les affirmations de la réponse.
    Appel 2 : vérifier d'un coup si chaque affirmation est soutenue.
    """

    name = "faithfulness"

    EXTRACT_CLAIMS_PROMPT = """Extrais les affirmations factuelles de la réponse ci-dessous.
Retourne UNIQUEMENT un tableau JSON de phrases courtes en français (PAS de dictionnaires, PAS de clés). Maximum 5 affirmations.

Exemple de format attendu : ["Le délai est de 6 ans.", "La facture doit être conservée."]

Question : {question}
Réponse : {answer}

Tableau JSON :"""

    BATCH_VERIFY_PROMPT = """Étant donné le contexte ci-dessous, détermine si CHAQUE affirmation est soutenue par ce contexte.
Réponds "oui" si l'information est présente dans le contexte, même formulée différemment ou de manière partielle.
Réponds "non" uniquement si l'information est absente ou contredite.

Contexte :
{context}

{claims_block}

Réponds UNIQUEMENT avec une liste JSON de "oui" ou "non", une par affirmation, dans l'ordre.
Exemple pour 3 affirmations : ["oui", "non", "oui"]

Liste JSON :"""

    @staticmethod
    def compute(
        question: str,
        answer: str,
        context_chunks: List[Dict],
        llm,
        ground_truth: Optional[str] = None,
    ) -> Dict:
        if not answer or not answer.strip():
            return {"score": 0.0, "claims": [], "details": [], "reason": "réponse vide"}

        if not context_chunks:
            return {"score": 0.0, "claims": [], "details": [], "reason": "contexte vide"}

        # 1. Extraire les affirmations de la réponse
        extract_resp = _safe_llm_call(
            llm,
            Faithfulness.EXTRACT_CLAIMS_PROMPT.format(question=question, answer=answer),
            max_tokens=DECOMPOSE_MAX_TOKENS,
            timeout=JUDGE_TIMEOUT,
        )
        claims = _parse_statements(extract_resp)
        claims = claims[:5]

        if not claims:
            logger.warning("Impossible d'extraire les affirmations de la réponse")
            return {"score": None, "claims": [], "details": [], "reason": "extraction échouée"}

        # 2. Construire le contexte concaténé
        context_text = "\n\n".join(
            f"[Chunk {i+1}]: {c.get('text', '')[:CHUNK_PREVIEW_LEN]}"
            for i, c in enumerate(context_chunks)
        )
        context_text = context_text[:MAX_CONTEXT_LEN]

        # 3. Batch vérification
        claims_block = "\n".join(f"Affirmation {i+1} : {c}" for i, c in enumerate(claims))
        prompt = Faithfulness.BATCH_VERIFY_PROMPT.format(
            context=context_text, claims_block=claims_block
        )
        response = _safe_llm_call(llm, prompt, max_tokens=JUDGE_MAX_TOKENS, timeout=JUDGE_TIMEOUT)
        support_flags = _parse_verdicts(response, len(claims))

        supported_count = sum(support_flags)
        details = []
        for claim, is_sup in zip(claims, support_flags):
            details.append({"claim": claim, "supported": is_sup})

        score = supported_count / len(claims) if claims else 0.0

        return {
            "score": round(score, 4),
            "claims": claims,
            "details": details,
            "supported_count": supported_count,
            "total_claims": len(claims),
            "llm_response": response.strip()[:200],
        }


# ---------------------------------------------------------------------------
# RAGAS Score (agrégé)
# ---------------------------------------------------------------------------

class RAGASScore:
    """
    Score RAGAS agrégé — moyenne harmonique de :
      - Context Precision
      - Context Recall (si ground_truth disponible)
      - Faithfulness
    """

    name = "ragas_score"

    @staticmethod
    def compute(
        context_precision: float,
        context_recall: Optional[float],
        faithfulness: float,
    ) -> Dict:
        """
        Calcule le score RAGAS global.

        Args:
            context_precision: Score de Context Precision [0,1]
            context_recall: Score de Context Recall [0,1] ou None
            faithfulness: Score de Faithfulness [0,1]

        Returns:
            {"score": float, "components": {...}}
        """
        scores = {}
        if context_precision is not None:
            scores["context_precision"] = context_precision
        if context_recall is not None:
            scores["context_recall"] = context_recall
        if faithfulness is not None:
            scores["faithfulness"] = faithfulness

        if not scores:
            return {"score": None, "components": scores, "reason": "aucune métrique disponible"}

        # Moyenne harmonique (robuste aux None)
        valid_scores = [s for s in scores.values() if s is not None and s > 0]
        if not valid_scores:
            harmonic_mean = 0.0
        else:
            n = len(valid_scores)
            harmonic_mean = n / sum(1.0 / s for s in valid_scores)

        return {
            "score": round(harmonic_mean, 4),
            "components": scores,
            "method": "harmonic_mean",
        }
