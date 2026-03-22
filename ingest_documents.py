"""
Script Principal - Ingestion et Indexation des Documents
Ce script charge les documents PDF, les découpe en chunks, génère les embeddings et les stocke dans FAISS
"""
import sys
import os
from pathlib import Path
from loguru import logger

# Ajouter le répertoire src au path de manière robuste
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ingestion import DocumentLoader, DocumentChunker, BERTEmbedder
from retrieval import FAISSVectorStore
from utils import load_config, setup_logging, ensure_directories


def main():
    """Fonction principale pour l'ingestion des documents"""
    
    # 1. Charger la configuration
    logger.info("=" * 60)
    logger.info("RAG BENCHMARK - INGESTION DES DOCUMENTS")
    logger.info("=" * 60)
    
    config = load_config("config.yaml")
    setup_logging(
        log_level=config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("file")
    )
    
    # 2. Créer les répertoires nécessaires
    ensure_directories([
        config.get("vector_store", {}).get("persist_directory", "./data/vector_store"),
        "./logs"
    ])
    
    # 3. Charger les documents
    logger.info("\n[1/5] Chargement des documents PDF...")
    data_source = config.get("ingestion", {}).get("data_source", "./donnees rag")
    loader = DocumentLoader(data_source)
    documents = loader.load_all_pdfs()
    
    if not documents:
        logger.error("Aucun document chargé. Arrêt du processus.")
        return
    
    stats = loader.get_document_stats(documents)
    logger.info(f"Documents chargés: {stats}")
    
    # 4. Découper en chunks
    logger.info("\n[2/5] Découpage des documents en chunks...")
    chunk_size = config.get("ingestion", {}).get("chunk_size", 1000)
    chunk_overlap = config.get("ingestion", {}).get("chunk_overlap", 200)

    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)
    
    chunk_stats = chunker.get_chunk_stats(chunks)
    logger.info(f"Statistiques des chunks: {chunk_stats}")
    
    # 5. Générer les embeddings
    logger.info("\n[3/5] Génération des embeddings avec BERT...")
    embedding_config = config.get("embeddings", {})
    embedder = BERTEmbedder(
        model_name=embedding_config.get("model_name", "bert-base-multilingual-cased"),
        device=embedding_config.get("device", "cpu")
    )
    
    enriched_chunks = embedder.embed_chunks(chunks, batch_size=32)
    logger.success(f"Embeddings générés pour {len(enriched_chunks)} chunks")
    
    # 6. Créer et remplir le vector store
    logger.info("\n[4/5] Création du vector store FAISS...")
    vector_store_config = config.get("vector_store", {})
    vector_store = FAISSVectorStore(
        embedding_dim=embedder.get_embedding_dimension(),
        persist_directory=vector_store_config.get("persist_directory", "./data/vector_store")
    )
    
    vector_store.add_chunks(enriched_chunks)
    
    # 7. Sauvegarder le vector store
    logger.info("\n[5/5] Sauvegarde du vector store...")
    vector_store.save()
    
    # 8. Afficher le résumé
    logger.info("\n" + "=" * 60)
    logger.success("INGESTION COMPLÉTÉE AVEC SUCCÈS!")
    logger.info("=" * 60)
    
    final_stats = vector_store.get_stats()
    logger.info(f"""
Résumé:
  - Documents chargés: {len(documents)}
  - Chunks créés: {final_stats['total_chunks']}
  - Vecteurs indexés: {final_stats['total_vectors']}
  - Dimension des embeddings: {final_stats['embedding_dimension']}
  - Type d'index: {final_stats['index_type']}
  
Prochaines étapes:
  1. Démarrer le serveur Ollama si ce n'est pas déjà fait
  2. Lancer l'API: python api.py
  3. Tester avec Postman sur http://localhost:8000
    """)


def test_search():
    """Fonction de test pour vérifier la recherche"""
    logger.info("\n[TEST] Test de recherche...")
    
    config = load_config(os.path.join(PROJECT_DIR, "config.yaml"))
    
    # Charger l'embedder
    embedding_config = config.get("embeddings", {})
    embedder = BERTEmbedder(
        model_name=embedding_config.get("model_name", "bert-base-multilingual-cased"),
        device=embedding_config.get("device", "cpu")
    )
    
    # Charger le vector store
    vector_store_config = config.get("vector_store", {})
    vector_store = FAISSVectorStore(
        embedding_dim=embedder.get_embedding_dimension(),
        persist_directory=vector_store_config.get("persist_directory", "./data/vector_store")
    )
    
    try:
        vector_store.load()
        logger.success("Vector store chargé")
        
        # Test de recherche
        test_query = "Qu'est-ce que la TVA?"
        logger.info(f"Requête de test: {test_query}")
        
        query_embedding = embedder.embed_text(test_query)
        results = vector_store.search(query_embedding, top_k=3)
        
        logger.info(f"\nRésultats trouvés: {len(results)}")
        for i, result in enumerate(results):
            logger.info(f"\n[{i+1}] Score: {result['score']:.4f}")
            logger.info(f"Texte: {result['text'][:200]}...")
            
    except FileNotFoundError:
        logger.error("Vector store non trouvé. Exécutez d'abord l'ingestion.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Benchmark - Ingestion des documents")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Effectuer un test de recherche après l'ingestion"
    )
    
    args = parser.parse_args()
    
    # Exécuter l'ingestion
    main()
    
    # Test optionnel
    if args.test:
        test_search()
