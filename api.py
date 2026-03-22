"""
RAG API - FastAPI application pour tester le système RAG avec Postman
"""
import time
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
from loguru import logger

# Ajouter le répertoire src au path de manière robuste
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ingestion import BERTEmbedder
from retrieval import FAISSVectorStore
from generation import OllamaLLM
from evaluation import RAGEvaluator
from utils import load_config, setup_logging

# Configuration du logging
setup_logging(log_level="INFO")

# Charger la configuration
try:
    config = load_config(os.path.join(PROJECT_DIR, "config.yaml"))
    logger.info("Configuration chargée")
except Exception as e:
    logger.error(f"Erreur lors du chargement de la configuration: {e}")
    config = {}

# Variables globales pour les composants
embedder: Optional["BERTEmbedder"] = None
vector_store: Optional["FAISSVectorStore"] = None
llm: Optional["OllamaLLM"] = None
similarity_threshold = 0.3  # seuil de similarité cosinus
config_top_k = 5  # top_k minimum depuis la config
search_mode = "hybrid"
hybrid_alpha = 0.6
hybrid_candidate_factor = 4


def _numpy_to_python(obj) -> Any:
    """Convertit récursivement les types numpy en types Python natifs pour la sérialisation JSON"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_python(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# Modèles Pydantic
class QueryRequest(BaseModel):
    query: str = Field(..., description="Question à poser au système RAG")
    top_k: int = Field(5, description="Nombre de chunks à récupérer", ge=1, le=20)
    temperature: float = Field(0.2, description="Température de génération", ge=0.0, le=2.0)
    max_tokens: int = Field(300, description="Nombre maximum de tokens", ge=50, le=2000)

class QueryResponse(BaseModel):
    query: str
    answer: str
    context_chunks: List[Dict]
    metrics: Dict
    latency_seconds: float

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]
    config: Dict

class StatsResponse(BaseModel):
    vector_store_stats: Dict


class EvalSingleRequest(BaseModel):
    query: str = Field(..., description="Question à évaluer")
    ground_truth: str = Field("", description="Réponse attendue (vérité terrain)")
    top_k: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(300, ge=50, le=2000)


class EvalDatasetRequest(BaseModel):
    top_k: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(300, ge=50, le=2000)
    max_questions: int = Field(0, description="Limiter le nombre de questions (0 = toutes)", ge=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère l'initialisation et la fermeture des composants"""
    global embedder, vector_store, llm, similarity_threshold, config_top_k, search_mode, hybrid_alpha, hybrid_candidate_factor

    try:
        logger.info("Initialisation des composants...")

        # Initialiser l'embedder
        embedding_config = config.get("embeddings", {})
        embedder = BERTEmbedder(
            model_name=embedding_config.get("model_name", "bert-base-multilingual-cased"),
            device=embedding_config.get("device", "cpu")
        )

        # Initialiser le vector store
        vector_store_config = config.get("vector_store", {})
        vector_store = FAISSVectorStore(
            embedding_dim=embedder.get_embedding_dimension(),
            persist_directory=vector_store_config.get("persist_directory", "./data/vector_store")
        )

        # Charger l'index s'il existe
        try:
            vector_store.load()
            logger.success("Vector store chargé depuis le disque")
        except FileNotFoundError:
            logger.warning("Aucun index FAISS trouvé. Veuillez d'abord ingérer des documents.")

        # Initialiser le LLM (ne pas bloquer si Ollama n'est pas accessible)
        llm_config = config.get("llm", {})
        try:
            llm = OllamaLLM(
                model=llm_config.get("model", "llama3.1:8b"),
                host=llm_config.get("host", "http://localhost:11434")
            )
        except Exception as e:
            logger.warning(f"Impossible d'initialiser le LLM: {e}. Les requêtes /query échoueront.")

        # Charger le seuil de similarité et top_k depuis la config 
        retrieval_config = config.get("retrieval", {})
        similarity_threshold = retrieval_config.get("similarity_threshold", 0.3)
        config_top_k = retrieval_config.get("top_k", 5)
        search_mode = retrieval_config.get("search_mode", "hybrid")
        hybrid_alpha = retrieval_config.get("hybrid_alpha", 0.6)
        hybrid_candidate_factor = retrieval_config.get("hybrid_candidate_factor", 4)
        logger.info(f"Seuil de similarité cosinus: {similarity_threshold}, top_k config: {config_top_k}, search_mode: {search_mode}")

        logger.success("Tous les composants sont initialisés")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        raise

    yield

    # Nettoyage à l'arrêt
    logger.info("Arrêt de l'application...")


# Initialiser l'application FastAPI avec lifespan
app = FastAPI(
    title="RAG Benchmark API",
    description="API pour tester le système RAG avec BERT, FAISS et Llama 3.1:8b",
    version="1.0.0",
    lifespan=lifespan,
)

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_component(name: str, component):
    """Vérifie qu'un composant est initialisé, lève une HTTPException sinon"""
    if component is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Le composant '{name}' n'est pas initialisé. Redémarrez le serveur."
        )


@app.get("/", tags=["General"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "RAG Benchmark API",
        "version": "1.0.0",
        "model": config.get("model_name", "bert_faiss_llama3.1_v1"),
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Vérifie l'état de santé de tous les composants"""
    vs_ready = (
        vector_store is not None
        and hasattr(vector_store, "index")
        and vector_store.index is not None
        and vector_store.index.ntotal > 0
    )

    components_status = {
        "embedder": embedder is not None,
        "vector_store": vs_ready,
        "llm": llm is not None
    }

    overall_status = "healthy" if all(components_status.values()) else "degraded"

    if not components_status["vector_store"]:
        logger.warning("Vector store vide - veuillez ingérer des documents")

    return {
        "status": overall_status,
        "components": components_status,
        "config": {
            "model_name": config.get("model_name", "N/A"),
            "chunk_size": config.get("ingestion", {}).get("chunk_size", "N/A"),
            "embedding_model": config.get("embeddings", {}).get("model_name", "N/A"),
            "llm_model": config.get("llm", {}).get("model", "N/A")
        }
    }
                                             

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Envoie une requête au système RAG

    - **query**: Question à poser
    - **top_k**: Nombre de chunks à récupérer (défaut: 5)
    - **temperature**: Température de génération (défaut: 0.2)
    - **max_tokens**: Nombre maximum de tokens (défaut: 512)
    """
    _check_component("embedder", embedder)
    _check_component("vector_store", vector_store)
    _check_component("llm", llm)
    assert embedder is not None
    assert vector_store is not None
    assert llm is not None

    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store vide. Veuillez d'abord ingérer des documents via le script d'ingestion."
        )

    try:
        start_time = time.time()

        # 1. Générer l'embedding de la requête
        logger.info(f"Traitement de la requête: {request.query[:50]}...")
        query_embedding = embedder.embed_text(request.query)

        # 2. Rechercher dans le vector store
        # Toujours récupérer au moins config_top_k chunks pour ne pas rater de résultats pertinents
        effective_top_k = max(request.top_k, config_top_k)

        if search_mode == "hybrid":
            retrieved_chunks = vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=request.query,
                top_k=effective_top_k,
                alpha=hybrid_alpha,
                candidate_factor=hybrid_candidate_factor,
            )
        else:
            retrieved_chunks = vector_store.search(
                query_embedding=query_embedding,
                top_k=effective_top_k,
            )

        # Filtrer par seuil de similarité cosinus
        before_filter = len(retrieved_chunks)
        retrieved_chunks = [c for c in retrieved_chunks if c.get("similarity", 0) >= similarity_threshold]
        if len(retrieved_chunks) < before_filter:
            logger.debug(f"Threshold {similarity_threshold}: {before_filter} → {len(retrieved_chunks)} chunks")

        if not retrieved_chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Aucun chunk avec similarité ≥ {similarity_threshold} trouvé"
            )

        # 3. Générer la réponse avec le LLM
        response_data = llm.generate_with_context(
            query=request.query,
            context_chunks=retrieved_chunks,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        end_time = time.time()

        # 4. Convertir les types numpy pour la sérialisation JSON
        metrics = {"num_chunks_retrieved": len(retrieved_chunks)}
        safe_chunks = _numpy_to_python(retrieved_chunks)

        # 5. Retourner la réponse
        return QueryResponse(
            query=request.query,
            answer=response_data["answer"],
            context_chunks=safe_chunks,
            metrics=metrics,
            latency_seconds=round(end_time - start_time, 3)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Récupère les statistiques du vector store"""
    _check_component("vector_store", vector_store)
    assert vector_store is not None

    return StatsResponse(
        vector_store_stats=_numpy_to_python(vector_store.get_stats())
    )


@app.get("/config", tags=["General"])
async def get_config():
    """Retourne la configuration actuelle"""
    return config


# -----------------------------------------------------------------------
# Endpoints d'évaluation RAGAS
# -----------------------------------------------------------------------

@app.post("/evaluate", tags=["Evaluation"])
async def evaluate_single(request: EvalSingleRequest):
    """
    Évalue une seule requête RAG avec les métriques RAGAS.

    - **query**: Question à évaluer
    - **ground_truth**: Réponse attendue (vérité terrain) — améliore Context Recall
    - **top_k / temperature / max_tokens**: Paramètres du pipeline RAG
    """
    _check_component("embedder", embedder)
    _check_component("vector_store", vector_store)
    _check_component("llm", llm)
    assert embedder is not None
    assert vector_store is not None
    assert llm is not None

    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store vide. Ingérez d'abord des documents."
        )

    try:
        evaluator = RAGEvaluator(llm=llm, embedder=embedder, vector_store=vector_store)

        result = evaluator.evaluate_query_end_to_end(
            question=request.query,
            ground_truth=request.ground_truth,
            top_k=max(request.top_k, config_top_k),
            similarity_threshold=similarity_threshold,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            search_mode=search_mode,
            hybrid_alpha=hybrid_alpha,
            hybrid_candidate_factor=hybrid_candidate_factor,
        )

        return _numpy_to_python(result)

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'évaluation : {str(e)}"
        )


@app.post("/evaluate/dataset", tags=["Evaluation"])
async def evaluate_dataset(request: EvalDatasetRequest):
    """
    Évalue le jeu de test complet (test_questions.yaml) avec les métriques RAGAS.

    Retourne un rapport avec les scores moyens et les détails par question.
    """
    import yaml as _yaml

    _check_component("embedder", embedder)
    _check_component("vector_store", vector_store)
    _check_component("llm", llm)
    assert embedder is not None
    assert vector_store is not None
    assert llm is not None

    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store vide. Ingérez d'abord des documents."
        )

    # Charger le jeu de test
    test_file = os.path.join(PROJECT_DIR, "test_questions.yaml")
    if not os.path.exists(test_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fichier test_questions.yaml introuvable."
        )

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = _yaml.safe_load(f)

    questions = test_data.get("questions", [])
    if not questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucune question trouvée dans test_questions.yaml."
        )

    if request.max_questions > 0:
        questions = questions[:request.max_questions]

    try:
        evaluator = RAGEvaluator(llm=llm, embedder=embedder, vector_store=vector_store)

        report = evaluator.evaluate_dataset(
            test_questions=questions,
            top_k=max(request.top_k, config_top_k),
            similarity_threshold=similarity_threshold,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            search_mode=search_mode,
            hybrid_alpha=hybrid_alpha,
            hybrid_candidate_factor=hybrid_candidate_factor,
        )

        # Sauvegarder le rapport
        RAGEvaluator.generate_report(
            evaluation_results=report,
            output_path="./logs/evaluation_report.json",
            config=config,
        )

        return _numpy_to_python(report)

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du dataset : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'évaluation : {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Démarrage de l'API RAG...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
