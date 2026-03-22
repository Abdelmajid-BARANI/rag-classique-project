"""
FAISS Vector Store Module + BM25 Hybrid Search
Gère le stockage et la recherche de vecteurs avec FAISS, combiné à BM25 pour la recherche hybride
"""
import os
import re
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from loguru import logger
from pathlib import Path


class FAISSVectorStore:
    """Classe pour gérer le vector store avec FAISS"""
    
    def __init__(self, embedding_dim: int, persist_directory: str = "./data/vector_store"):
        """
        Initialise le vector store FAISS
        
        Args:
            embedding_dim: Dimension des embeddings
            persist_directory: Répertoire pour sauvegarder l'index
        """
        self.embedding_dim = embedding_dim
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Créer un index FAISS (IndexFlatIP pour similarité cosinus)
        # Les vecteurs doivent être normalisés L2 avant l'ajout/la recherche
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks = []  # Stocker les chunks originaux
        self.bm25: Optional[BM25Okapi] = None  # Index BM25 (construit au chargement)
        self._tokenized_corpus: List[List[str]] = []  # Corpus tokenisé pour BM25
        
        logger.info(f"Vector store initialisé (dimension={embedding_dim}, index=IndexFlatIP/cosine)")
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Ajoute des chunks avec leurs embeddings au vector store
        
        Args:
            chunks: Liste de chunks contenant les embeddings
        """
        if not chunks:
            logger.warning("Aucun chunk à ajouter")
            return
        
        # Extraire les embeddings
        embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype('float32')
        
        # Vérifier la dimension
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Dimension des embeddings incorrecte: {embeddings.shape[1]} vs {self.embedding_dim}")
        
        # Normaliser L2 pour que le produit scalaire = similarité cosinus
        faiss.normalize_L2(embeddings)

        # Ajouter à l'index FAISS
        self.index.add(embeddings)
        
        # Stocker les chunks (sans les embeddings pour économiser la mémoire)
        for chunk in chunks:
            chunk_copy = chunk.copy()
            if "embedding" in chunk_copy:
                del chunk_copy["embedding"]
            self.chunks.append(chunk_copy)
        
        # Reconstruire l'index BM25 sur l'ensemble des chunks
        self._build_bm25_index()
        
        logger.success(f"{len(chunks)} chunks ajoutés au vector store (total: {self.index.ntotal})")
    
    # ------------------------------------------------------------------
    # BM25 Index
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenise un texte en mots significatifs (≥ 2 caractères, minuscules)."""
        return [w.lower() for w in re.findall(r'[A-Za-zÀ-ÿ0-9/]{2,}', text)]

    def _build_bm25_index(self):
        """Construit (ou reconstruit) l'index BM25 à partir de self.chunks."""
        if not self.chunks:
            self.bm25 = None
            self._tokenized_corpus = []
            return
        self._tokenized_corpus = [self._tokenize(c.get("text", "")) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"Index BM25 construit sur {len(self._tokenized_corpus)} documents")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Recherche les chunks les plus similaires à un embedding de requête
        
        Args:
            query_embedding: Embedding de la requête
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste des chunks les plus similaires avec leurs scores
        """
        if self.index.ntotal == 0:
            logger.warning("Le vector store est vide")
            return []
        
        # S'assurer que l'embedding est au bon format
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Normaliser L2 pour que le produit scalaire = similarité cosinus
        faiss.normalize_L2(query_vector)

        # Rechercher
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Construire les résultats
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            result = self.chunks[idx].copy()
            # score = similarité cosinus ∈ [-1, 1] ; 1 = identique
            result["score"] = float(score)
            result["similarity"] = float(score)  # cosine similarity directe
            result["rank"] = i + 1
            results.append(result)
        
        logger.debug(f"Recherche effectuée: {len(results)} résultats trouvés")
        return results

    # ------------------------------------------------------------------
    # Recherche BM25 pure
    # ------------------------------------------------------------------

    def bm25_search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Recherche BM25 (keyword / lexicale) sur les chunks stockés.

        Args:
            query_text: Texte de la requête
            top_k: Nombre de résultats

        Returns:
            Liste de chunks triés par score BM25 décroissant
        """
        if self.bm25 is None or not self.chunks:
            logger.warning("Index BM25 non disponible")
            return []

        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            result = self.chunks[idx].copy()
            result["bm25_score"] = float(scores[idx])
            result["rank"] = rank + 1
            results.append(result)

        logger.debug(f"BM25 search: {len(results)} résultats")
        return results

    # ------------------------------------------------------------------
    # Recherche hybride : FAISS sémantique + BM25 lexical
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.6,
        candidate_factor: int = 4,
    ) -> List[Dict]:
        """
        Recherche hybride combinant FAISS (similarité cosinus) et BM25 (lexicale).

        Méthode :
          1. Récupérer top_k * candidate_factor candidats depuis FAISS
          2. Calculer les scores BM25 sur TOUS les chunks
          3. Normaliser les deux distributions en [0, 1] (min-max)
          4. Score final = α * score_sémantique + (1-α) * score_bm25
          5. Re-trier et garder top_k

        Args:
            query_embedding: Embedding de la requête
            query_text: Texte brut de la requête (pour BM25)
            top_k: Nombre final de résultats
            alpha: Poids du score sémantique (0 = tout BM25, 1 = tout FAISS)
            candidate_factor: Facteur de sur-récupération FAISS

        Returns:
            Liste des top_k chunks triés par score hybride décroissant
        """
        if self.index.ntotal == 0:
            logger.warning("Le vector store est vide")
            return []

        # Préparer le vecteur de requête normalisé (réutilisé pour les lookups)
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)

        # 1. Scores sémantiques FAISS — sur-récupérer
        n_candidates = min(top_k * candidate_factor, self.index.ntotal)
        semantic_results = self.search(query_embedding, top_k=n_candidates)

        # Construire un dict idx → score_sémantique (via chunk_id ou position)
        # On s'appuie sur l'index du chunk dans self.chunks
        sem_scores_map: Dict[int, float] = {}
        for r in semantic_results:
            # Retrouver l'index du chunk dans self.chunks
            for ci, c in enumerate(self.chunks):
                if c.get("chunk_id") == r.get("chunk_id") and c.get("text") == r.get("text"):
                    sem_scores_map[ci] = r["score"]
                    break

        # 2. Scores BM25 sur TOUT le corpus
        bm25_scores_all = np.zeros(len(self.chunks))
        if self.bm25 is not None:
            tokenized_query = self._tokenize(query_text)
            bm25_scores_all = self.bm25.get_scores(tokenized_query)

        # Union des candidats : indices FAISS + top BM25
        bm25_top_indices = set(np.argsort(bm25_scores_all)[::-1][:n_candidates].tolist())
        all_candidate_indices = set(sem_scores_map.keys()) | bm25_top_indices

        # 2b. Calculer le vrai score sémantique pour les candidats BM25-only
        #     (ceux absents du top FAISS) via index.reconstruct() + dot product
        bm25_only = all_candidate_indices - set(sem_scores_map.keys())
        if bm25_only:
            for idx in bm25_only:
                vec = self.index.reconstruct(int(idx)).reshape(1, -1)
                sem_scores_map[idx] = float(np.dot(query_vector, vec.T)[0][0])

        # 3. Normaliser les scores en [0, 1]
        sem_values = [sem_scores_map.get(i, 0.0) for i in all_candidate_indices]
        bm25_values = [float(bm25_scores_all[i]) for i in all_candidate_indices]

        sem_min, sem_max = (min(sem_values), max(sem_values)) if sem_values else (0, 1)
        bm25_min, bm25_max = (min(bm25_values), max(bm25_values)) if bm25_values else (0, 1)

        sem_range = sem_max - sem_min if sem_max != sem_min else 1.0
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

        # 4. Calculer le score hybride
        scored_candidates = []
        for idx in all_candidate_indices:
            sem_score = sem_scores_map.get(idx, 0.0)
            sem_norm = (sem_score - sem_min) / sem_range
            bm25_norm = (float(bm25_scores_all[idx]) - bm25_min) / bm25_range
            hybrid_score = alpha * sem_norm + (1 - alpha) * bm25_norm

            chunk = self.chunks[idx].copy()
            chunk["score"] = hybrid_score
            chunk["semantic_score"] = sem_score
            chunk["bm25_score"] = float(bm25_scores_all[idx])
            chunk["similarity"] = hybrid_score  # pour compatibilité threshold
            scored_candidates.append(chunk)

        # 5. Trier par score hybride décroissant et garder top_k
        scored_candidates.sort(key=lambda r: r["score"], reverse=True)
        final = scored_candidates[:top_k]
        for i, r in enumerate(final):
            r["rank"] = i + 1

        logger.debug(
            f"Hybrid search: {len(all_candidate_indices)} candidats → {len(final)} résultats "
            f"(α={alpha})"
        )
        return final

    def save(self, filename: str = "faiss_index"):
        """
        Sauvegarde l'index FAISS et les chunks
        
        Args:
            filename: Nom de base des fichiers
        """
        # Sauvegarder l'index FAISS
        index_path = self.persist_directory / f"{filename}.index"
        faiss.write_index(self.index, str(index_path))
        
        # Sauvegarder les chunks
        chunks_path = self.persist_directory / f"{filename}.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.success(f"Vector store sauvegardé dans {self.persist_directory}")
    
    def load(self, filename: str = "faiss_index"):
        """
        Charge un index FAISS et les chunks depuis le disque
        
        Args:
            filename: Nom de base des fichiers
        """
        index_path = self.persist_directory / f"{filename}.index"
        chunks_path = self.persist_directory / f"{filename}.pkl"
        
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Fichiers d'index non trouvés dans {self.persist_directory}")
        
        # Charger l'index
        self.index = faiss.read_index(str(index_path))
        
        # Charger les chunks
        try:
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Fichier chunks corrompu: {e}")
            self.chunks = []
            raise
        
        # Vérification de cohérence
        if self.index.ntotal != len(self.chunks):
            logger.warning(
                f"Incohérence: {self.index.ntotal} vecteurs dans l'index mais {len(self.chunks)} chunks. "
                "Réingérez les documents."
            )
        
        # Reconstruire l'index BM25 à partir des chunks chargés
        self._build_bm25_index()
        
        logger.success(f"Vector store chargé: {self.index.ntotal} vecteurs, {len(self.chunks)} chunks")
    
    def get_stats(self) -> Dict:
        """
        Retourne des statistiques sur le vector store
        
        Returns:
            Dictionnaire avec les statistiques
        """
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "total_chunks": len(self.chunks),
            "index_type": type(self.index).__name__
        }
    
    def clear(self):
        """Vide le vector store"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks = []
        logger.info("Vector store vidé")


if __name__ == "__main__":
    # Test du vector store
    dim = 768
    store = FAISSVectorStore(embedding_dim=dim)
    
    # Créer des embeddings de test
    test_embeddings = np.random.rand(10, dim).astype('float32')
    test_chunks = [
        {"text": f"Chunk {i}", "chunk_id": i, "embedding": test_embeddings[i]}
        for i in range(10)
    ]
    
    store.add_chunks(test_chunks)
    
    # Recherche
    query = np.random.rand(dim).astype('float32')
    results = store.search(query, top_k=3)
    print(f"Résultats: {len(results)}")
    print(results[0])
