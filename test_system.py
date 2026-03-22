"""
Script de Test Rapide
Teste rapidement toutes les fonctionnalités principales du système
"""
import sys
import os

# Ajouter le répertoire src au path de manière robuste
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from loguru import logger
from utils import setup_logging

setup_logging(log_level="INFO")

def test_ollama():
    """Test de connexion à Ollama"""
    logger.info("Test 1: Connexion à Ollama...")
    try:
        from generation import OllamaLLM
        llm = OllamaLLM()
        response = llm.generate("Dis bonjour en français", max_tokens=50)
        logger.success(f"✓ Ollama fonctionne. Réponse: {response[:100]}...")
        return True
    except Exception as e:
        logger.error(f"✗ Erreur Ollama: {e}")
        return False

def test_embeddings():
    """Test du modèle d'embeddings"""
    logger.info("Test 2: Modèle d'embeddings BERT...")
    try:
        from ingestion import BERTEmbedder
        embedder = BERTEmbedder()
        embedding = embedder.embed_text("Test de texte")
        logger.success(f"✓ Embeddings fonctionnent. Dimension: {len(embedding)}")
        return True
    except Exception as e:
        logger.error(f"✗ Erreur embeddings: {e}")
        return False

def test_document_loading():
    """Test du chargement de documents"""
    logger.info("Test 3: Chargement des documents...")
    try:
        from ingestion import DocumentLoader
        loader = DocumentLoader("./donnees rag")
        documents = loader.load_all_pdfs()
        logger.success(f"✓ Documents chargés: {len(documents)} fichiers")
        return True
    except Exception as e:
        logger.error(f"✗ Erreur chargement: {e}")
        return False

def test_chunking():
    """Test du découpage en chunks"""
    logger.info("Test 4: Découpage en chunks...")
    try:
        from ingestion import DocumentChunker
        chunker = DocumentChunker(chunk_size=256, chunk_overlap=50)
        chunks = chunker.chunk_text("Ceci est un texte de test. " * 100)
        logger.success(f"✓ Chunking fonctionne. Chunks créés: {len(chunks)}")
        return True
    except Exception as e:
        logger.error(f"✗ Erreur chunking: {e}")
        return False

def test_vector_store():
    """Test du vector store"""
    logger.info("Test 5: Vector store FAISS...")
    try:
        from retrieval import FAISSVectorStore
        from utils import load_config
        import numpy as np
        
        config = load_config(os.path.join(PROJECT_DIR, "config.yaml"))
        dim = config.get("embeddings", {}).get("dimension", 384)
        
        store = FAISSVectorStore(embedding_dim=dim)
        
        # Ajouter des vecteurs de test
        test_chunks = [
            {
                "text": f"Chunk {i}",
                "chunk_id": i,
                "embedding": np.random.rand(dim).astype('float32')
            }
            for i in range(5)
        ]
        store.add_chunks(test_chunks)
        
        # Tester la recherche
        query = np.random.rand(dim).astype('float32')
        results = store.search(query, top_k=3)
        
        logger.success(f"✓ Vector store fonctionne. Résultats: {len(results)}")
        return True
    except Exception as e:
        logger.error(f"✗ Erreur vector store: {e}")
        return False

def test_evaluation():
    """Test du module d'évaluation RAGAS"""
    logger.info("Test 6: Système d'évaluation RAGAS...")
    try:
        from evaluation import RAGEvaluator, ContextPrecision, ContextRecall, Faithfulness, RAGASScore

        # Test du score RAGAS (ne nécessite pas de LLM)
        ragas = RAGASScore.compute(
            context_precision=0.8,
            context_recall=0.6,
            faithfulness=0.9,
        )
        assert ragas["score"] is not None and ragas["score"] > 0
        logger.success(f"✓ Module évaluation RAGAS chargé. Score test = {ragas['score']:.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Erreur évaluation: {e}")
        return False

def test_end_to_end():
    """Test end-to-end si le vector store existe"""
    logger.info("Test 7: Test end-to-end (si index existe)...")
    try:
        from ingestion import BERTEmbedder
        from retrieval import FAISSVectorStore
        from generation import OllamaLLM
        from utils import load_config
        
        config = load_config(os.path.join(PROJECT_DIR, "config.yaml"))
        
        # Charger les composants
        embedder = BERTEmbedder()
        vector_store = FAISSVectorStore(
            embedding_dim=embedder.get_embedding_dimension(),
            persist_directory=config.get("vector_store", {}).get("persist_directory", "./data/vector_store")
        )
        
        try:
            vector_store.load()
            llm = OllamaLLM()
            
            # Test de requête
            query = "Qu'est-ce que la TVA?"
            query_embedding = embedder.embed_text(query)
            chunks = vector_store.search(query_embedding, top_k=2)
            
            if chunks:
                response = llm.generate_with_context(query, chunks, max_tokens=200)
                logger.success(f"✓ Test end-to-end réussi!")
                logger.info(f"Réponse: {response['answer'][:200]}...")
                return True
            else:
                logger.warning("⚠ Aucun chunk trouvé (normal si pas encore ingéré)")
                return True
                
        except FileNotFoundError:
            logger.warning("⚠ Vector store non trouvé (exécutez d'abord l'ingestion)")
            return True
            
    except Exception as e:
        logger.error(f"✗ Erreur test end-to-end: {e}")
        return False

def main():
    """Exécute tous les tests"""
    logger.info("=" * 60)
    logger.info("TESTS DU SYSTÈME RAG")
    logger.info("=" * 60)
    
    tests = [
        ("Ollama", test_ollama),
        ("Embeddings", test_embeddings),
        ("Document Loading", test_document_loading),
        ("Chunking", test_chunking),
        ("Vector Store", test_vector_store),
        ("Evaluation", test_evaluation),
        ("End-to-End", test_end_to_end)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Exception dans {name}: {e}")
            results.append((name, False))
        logger.info("")
    
    # Résumé
    logger.info("=" * 60)
    logger.info("RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nRésultat: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.success("\n🎉 Tous les tests sont passés!")
        logger.info("\nProchaines étapes:")
        logger.info("1. Exécuter: python ingest_documents.py")
        logger.info("2. Démarrer: python api.py")
        logger.info("3. Tester avec Postman sur http://localhost:8000")
    else:
        logger.warning("\n⚠ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")

if __name__ == "__main__":
    main()
