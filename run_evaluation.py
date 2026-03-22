"""
Script d'évaluation RAGAS standalone
Exécute l'évaluation complète du pipeline RAG sur le jeu de test.

Usage :
  python run_evaluation.py                    # Évalue toutes les questions
  python run_evaluation.py --max-questions 3  # Évalue seulement 3 questions
  python run_evaluation.py --question 1       # Évalue une seule question (par ID)
"""
import sys
import os
import argparse
import yaml

# Ajouter le répertoire src au path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from loguru import logger
from utils import load_config, setup_logging
from ingestion import BERTEmbedder
from retrieval import FAISSVectorStore
from generation import OllamaLLM
from evaluation import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(description="Évaluation RAGAS du pipeline RAG")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Nombre max de questions à évaluer (0 = toutes)")
    parser.add_argument("--question", type=int, default=0,
                        help="Évaluer une seule question par son ID")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Nombre de chunks à récupérer (défaut: config)")
    parser.add_argument("--output", type=str, default="./logs/evaluation_report.json",
                        help="Chemin du rapport JSON")
    args = parser.parse_args()

    # Setup
    setup_logging(log_level="INFO")
    config = load_config(os.path.join(PROJECT_DIR, "config.yaml"))

    top_k = args.top_k or config.get("retrieval", {}).get("top_k", 5)
    similarity_threshold = config.get("retrieval", {}).get("similarity_threshold", 0.3)
    search_mode = config.get("retrieval", {}).get("search_mode", "hybrid")
    hybrid_alpha = config.get("retrieval", {}).get("hybrid_alpha", 0.6)
    hybrid_candidate_factor = config.get("retrieval", {}).get("hybrid_candidate_factor", 4)
    temperature = config.get("llm", {}).get("temperature", 0.2)
    max_tokens = config.get("llm", {}).get("max_tokens", 300)

    # Charger les composants
    logger.info("Initialisation des composants...")

    embedding_config = config.get("embeddings", {})
    embedder = BERTEmbedder(
        model_name=embedding_config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2"),
        device=embedding_config.get("device", "cpu"),
    )

    vector_store_config = config.get("vector_store", {})
    vector_store = FAISSVectorStore(
        embedding_dim=embedder.get_embedding_dimension(),
        persist_directory=vector_store_config.get("persist_directory", "./data/vector_store"),
    )
    vector_store.load()

    llm_config = config.get("llm", {})
    llm_instance = OllamaLLM(
        model=llm_config.get("model", "llama3.1:8b"),
        host=llm_config.get("host", "http://localhost:11434"),
    )

    # Charger le jeu de test
    test_file = os.path.join(PROJECT_DIR, "test_questions.yaml")
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = yaml.safe_load(f)

    questions = test_data.get("questions", [])

    if args.question > 0:
        questions = [q for q in questions if q.get("id") == args.question]
        if not questions:
            logger.error(f"Question #{args.question} non trouvée dans test_questions.yaml")
            sys.exit(1)
        logger.info(f"Évaluation de la question #{args.question} uniquement")
    elif args.max_questions > 0:
        questions = questions[:args.max_questions]
        logger.info(f"Évaluation limitée à {args.max_questions} questions")

    logger.info(f"Nombre de questions à évaluer : {len(questions)}")

    # Évaluation
    evaluator = RAGEvaluator(llm=llm_instance, embedder=embedder, vector_store=vector_store)

    if len(questions) == 1 and args.question > 0:
        # Évaluation d'une seule question
        q = questions[0]
        logger.info(f"Évaluation de : {q['query']}")
        result = evaluator.evaluate_query_end_to_end(
            question=q["query"],
            ground_truth=q.get("ground_truth", ""),
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            temperature=temperature,
            max_tokens=max_tokens,
            search_mode=search_mode,
            hybrid_alpha=hybrid_alpha,
            hybrid_candidate_factor=hybrid_candidate_factor,
        )
        # Affichage
        import json
        print("\n" + "=" * 65)
        print(f"  RÉSULTAT — Question #{q['id']}")
        print("=" * 65)
        print(f"  Question : {q['query']}")
        print(f"  Réponse  : {result.get('answer', '')[:300]}")
        print("-" * 65)
        for metric in ["context_precision", "context_recall", "faithfulness", "ragas_score"]:
            data = result.get(metric, {})
            score = data.get("score", "N/A") if isinstance(data, dict) else data
            s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"  {metric:<25} : {s}")
        print("=" * 65)

        # Sauvegarder
        RAGEvaluator.generate_report(
            evaluation_results={"results": [result], "num_questions": 1, "num_evaluated": 1},
            output_path=args.output,
            config=config,
        )
    else:
        # Évaluation du dataset complet
        report = evaluator.evaluate_dataset(
            test_questions=questions,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            temperature=temperature,
            max_tokens=max_tokens,
            search_mode=search_mode,
            hybrid_alpha=hybrid_alpha,
            hybrid_candidate_factor=hybrid_candidate_factor,
        )

        # Affichage du résumé
        RAGEvaluator.print_summary(report)

        # Sauvegarder le rapport
        RAGEvaluator.generate_report(
            evaluation_results=report,
            output_path=args.output,
            config=config,
        )

    logger.success("Évaluation terminée !")


if __name__ == "__main__":
    main()
