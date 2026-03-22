"""
RAG Evaluator
Orchestre l'évaluation complète d'un pipeline RAG avec les métriques RAGAS.

Modes d'utilisation :
  1. Évaluer une seule requête (evaluate_single)
  2. Évaluer un jeu de test complet (evaluate_dataset)
  3. Générer un rapport de benchmark (generate_report)
"""
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from loguru import logger

from .metrics import ContextPrecision, ContextRecall, Faithfulness, RAGASScore


class RAGEvaluator:
    """
    Évaluateur RAGAS pour un pipeline RAG.

    Utilise le LLM comme juge pour calculer :
      - Context Precision
      - Context Recall
      - Faithfulness
      - Score RAGAS global
    """

    def __init__(self, llm, embedder=None, vector_store=None):
        """
        Args:
            llm: Instance OllamaLLM (pour la génération et le jugement)
            embedder: Instance BERTEmbedder (optionnel, pour pipeline complet)
            vector_store: Instance FAISSVectorStore (optionnel, pour pipeline complet)
        """
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("RAGEvaluator initialisé")

    # ------------------------------------------------------------------
    # Évaluation d'une seule requête
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context_chunks: List[Dict],
        ground_truth: str = "",
        compute_precision: bool = True,
        compute_recall: bool = True,
        compute_faithfulness: bool = True,
    ) -> Dict:
        """
        Évalue une seule réponse RAG.

        Args:
            question: Question posée
            answer: Réponse générée par le RAG
            context_chunks: Chunks de contexte récupérés
            ground_truth: Réponse attendue (vérité terrain)
            compute_precision: Calculer Context Precision
            compute_recall: Calculer Context Recall
            compute_faithfulness: Calculer Faithfulness

        Returns:
            Dictionnaire complet des scores et détails
        """
        start_time = time.time()
        results = {
            "question": question,
            "answer_preview": answer[:200] if answer else "",
            "num_context_chunks": len(context_chunks),
            "ground_truth_provided": bool(ground_truth),
        }

        # Context Precision
        cp_score = None
        if compute_precision:
            logger.info("  → Calcul de Context Precision...")
            cp_result = ContextPrecision.compute(
                question=question,
                context_chunks=context_chunks,
                llm=self.llm,
                ground_truth=ground_truth,
            )
            results["context_precision"] = cp_result
            cp_score = cp_result.get("score")

        # Context Recall
        cr_score = None
        if compute_recall and ground_truth:
            logger.info("  → Calcul de Context Recall...")
            cr_result = ContextRecall.compute(
                question=question,
                context_chunks=context_chunks,
                llm=self.llm,
                ground_truth=ground_truth,
            )
            results["context_recall"] = cr_result
            cr_score = cr_result.get("score")
        elif compute_recall and not ground_truth:
            results["context_recall"] = {"score": None, "reason": "ground_truth manquante"}

        # Faithfulness
        f_score = None
        if compute_faithfulness:
            logger.info("  → Calcul de Faithfulness...")
            f_result = Faithfulness.compute(
                question=question,
                answer=answer,
                context_chunks=context_chunks,
                llm=self.llm,
                ground_truth=ground_truth,
            )
            results["faithfulness"] = f_result
            f_score = f_result.get("score")

        # Score RAGAS global
        ragas = RAGASScore.compute(
            context_precision=cp_score if cp_score is not None else 0.0,
            context_recall=cr_score,
            faithfulness=f_score if f_score is not None else 0.0,
        )
        results["ragas_score"] = ragas

        results["evaluation_time_seconds"] = round(time.time() - start_time, 2)
        return results

    # ------------------------------------------------------------------
    # Pipeline complet : retrieval + generation + évaluation
    # ------------------------------------------------------------------

    def evaluate_query_end_to_end(
        self,
        question: str,
        ground_truth: str = "",
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        temperature: float = 0.2,
        max_tokens: int = 300,
        search_mode: str = "hybrid",
        hybrid_alpha: float = 0.6,
        hybrid_candidate_factor: int = 4,
    ) -> Dict:
        """
        Exécute le pipeline RAG complet et évalue le résultat.

        Args:
            question: Question à évaluer
            ground_truth: Réponse attendue
            top_k: Nombre de chunks à récupérer
            similarity_threshold: Seuil de similarité
            temperature: Température LLM
            max_tokens: Tokens max pour la réponse
            search_mode: "semantic" (FAISS seul) ou "hybrid" (FAISS + BM25)
            hybrid_alpha: Poids sémantique pour le mode hybride (0=BM25, 1=FAISS)
            hybrid_candidate_factor: Facteur de sur-récupération pour le mode hybride

        Returns:
            Résultat complet avec réponse RAG et scores d'évaluation
        """
        if not self.embedder or not self.vector_store:
            raise ValueError("embedder et vector_store requis pour l'évaluation end-to-end")

        start = time.time()

        # 1. Retrieval — sémantique ou hybride
        query_embedding = self.embedder.embed_text(question)

        if search_mode == "hybrid":
            context_chunks = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=question,
                top_k=top_k,
                alpha=hybrid_alpha,
                candidate_factor=hybrid_candidate_factor,
            )
        else:
            context_chunks = self.vector_store.search(query_embedding, top_k=top_k)

        context_chunks = [c for c in context_chunks if c.get("similarity", 0) >= similarity_threshold]

        retrieval_time = time.time() - start

        if not context_chunks:
            return {
                "question": question,
                "answer": "",
                "context_chunks": [],
                "retrieval_time": round(retrieval_time, 3),
                "error": f"Aucun chunk avec similarité ≥ {similarity_threshold}",
            }

        # 2. Generation
        gen_start = time.time()
        response_data = self.llm.generate_with_context(
            query=question,
            context_chunks=context_chunks,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        generation_time = time.time() - gen_start
        answer = response_data.get("answer", "")

        # 3. Evaluation
        eval_results = self.evaluate_single(
            question=question,
            answer=answer,
            context_chunks=context_chunks,
            ground_truth=ground_truth,
        )

        eval_results["answer"] = answer
        eval_results["retrieval_time"] = round(retrieval_time, 3)
        eval_results["generation_time"] = round(generation_time, 3)
        eval_results["total_time"] = round(time.time() - start, 3)

        return eval_results

    # ------------------------------------------------------------------
    # Évaluation d'un jeu de test complet
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        test_questions: List[Dict],
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        temperature: float = 0.2,
        max_tokens: int = 300,
        search_mode: str = "hybrid",
        hybrid_alpha: float = 0.6,
        hybrid_candidate_factor: int = 4,
    ) -> Dict:
        """
        Évalue un jeu de test complet.

        Args:
            test_questions: Liste de dicts avec clés :
                - "query" (str) : question
                - "ground_truth" (str, optionnel) : réponse attendue
            top_k: Nombre de chunks
            similarity_threshold: Seuil de similarité
            temperature: Température LLM
            max_tokens: Max tokens
            search_mode: "semantic" ou "hybrid"
            hybrid_alpha: Poids sémantique pour hybrid
            hybrid_candidate_factor: Facteur de sur-récupération

        Returns:
            Rapport complet avec scores moyens et détails par question
        """
        if not self.embedder or not self.vector_store:
            raise ValueError("embedder et vector_store requis pour evaluate_dataset")

        start_time = time.time()
        results_per_question = []
        scores_aggregate = {
            "context_precision": [],
            "context_recall": [],
            "faithfulness": [],
            "ragas_score": [],
        }

        total = len(test_questions)
        for i, tq in enumerate(test_questions, 1):
            question = tq.get("query", "")
            ground_truth = tq.get("ground_truth", "")
            qid = tq.get("id", i)

            logger.info(f"[{i}/{total}] Évaluation question #{qid} : {question[:60]}...")

            try:
                result = self.evaluate_query_end_to_end(
                    question=question,
                    ground_truth=ground_truth,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    search_mode=search_mode,
                    hybrid_alpha=hybrid_alpha,
                    hybrid_candidate_factor=hybrid_candidate_factor,
                )
                result["question_id"] = qid

                # Collecter les scores
                for metric_key in scores_aggregate:
                    metric_data = result.get(metric_key, {})
                    if isinstance(metric_data, dict):
                        s = metric_data.get("score")
                    else:
                        s = metric_data
                    if s is not None:
                        scores_aggregate[metric_key].append(s)

                results_per_question.append(result)

            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation de la question #{qid} : {e}")
                results_per_question.append({
                    "question_id": qid,
                    "question": question,
                    "error": str(e),
                })

        # Calculer les moyennes
        averages = {}
        for metric_key, values in scores_aggregate.items():
            if values:
                averages[metric_key] = round(sum(values) / len(values), 4)
            else:
                averages[metric_key] = None

        total_time = time.time() - start_time

        report = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": total,
            "num_evaluated": len([r for r in results_per_question if "error" not in r]),
            "average_scores": averages,
            "results": results_per_question,
            "total_evaluation_time_seconds": round(total_time, 2),
        }

        return report

    # ------------------------------------------------------------------
    # Génération de rapport
    # ------------------------------------------------------------------

    @staticmethod
    def generate_report(
        evaluation_results: Dict,
        output_path: str = "./logs/evaluation_report.json",
        config: Optional[Dict] = None,
    ) -> str:
        """
        Sauvegarde un rapport d'évaluation en JSON.

        Args:
            evaluation_results: Résultats de evaluate_dataset
            output_path: Chemin de sortie
            config: Configuration du modèle (optionnel)

        Returns:
            Chemin du fichier généré
        """
        report: Dict[str, Any] = {
            "report_type": "RAGAS Evaluation",
            "generated_at": datetime.now().isoformat(),
        }

        if config:
            report["model_config"] = {
                "model_name": config.get("model_name", "N/A"),
                "embedding_model": config.get("embeddings", {}).get("model_name", "N/A"),
                "llm_model": config.get("llm", {}).get("model", "N/A"),
                "chunk_size": config.get("ingestion", {}).get("chunk_size", "N/A"),
                "chunk_overlap": config.get("ingestion", {}).get("chunk_overlap", "N/A"),
                "top_k": config.get("retrieval", {}).get("top_k", "N/A"),
            }

        report["evaluation"] = evaluation_results

        # Sauvegarder
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.success(f"Rapport d'évaluation sauvegardé : {output_path}")
        return str(output)

    # ------------------------------------------------------------------
    # Affichage console
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(evaluation_results: Dict):
        """Affiche un résumé lisible des résultats."""
        avg = evaluation_results.get("average_scores", {})
        n = evaluation_results.get("num_questions", 0)
        n_eval = evaluation_results.get("num_evaluated", 0)
        total_time = evaluation_results.get("total_evaluation_time_seconds", 0)

        print("\n" + "=" * 65)
        print("  RAPPORT D'ÉVALUATION RAGAS")
        print("=" * 65)
        print(f"  Questions évaluées : {n_eval}/{n}")
        print(f"  Temps total        : {total_time:.1f}s")
        print("-" * 65)
        print(f"  {'Métrique':<25} {'Score':>10}")
        print("-" * 65)

        metrics_labels = {
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
            "faithfulness": "Faithfulness",
            "ragas_score": "RAGAS Score",
        }

        for key, label in metrics_labels.items():
            score = avg.get(key)
            if score is not None:
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  {label:<25} {score:>8.4f}  {bar}")
            else:
                print(f"  {label:<25} {'N/A':>10}")

        print("=" * 65)

        # Détails par question
        results = evaluation_results.get("results", [])
        if results:
            print(f"\n  Détails par question :")
            print("-" * 65)
            for r in results:
                qid = r.get("question_id", "?")
                question = r.get("question", "")[:50]
                if "error" in r:
                    print(f"  Q#{qid}: {question}... → ERREUR: {r['error'][:40]}")
                    continue

                cp = r.get("context_precision", {}).get("score", "N/A")
                cr = r.get("context_recall", {}).get("score", "N/A")
                ff = r.get("faithfulness", {}).get("score", "N/A")
                rg = r.get("ragas_score", {}).get("score", "N/A")

                cp_str = f"{cp:.2f}" if isinstance(cp, (int, float)) else str(cp)
                cr_str = f"{cr:.2f}" if isinstance(cr, (int, float)) else str(cr)
                ff_str = f"{ff:.2f}" if isinstance(ff, (int, float)) else str(ff)
                rg_str = f"{rg:.2f}" if isinstance(rg, (int, float)) else str(rg)

                print(f"  Q#{qid}: {question}...")
                print(f"         CP={cp_str}  CR={cr_str}  F={ff_str}  RAGAS={rg_str}")
            print("-" * 65)
