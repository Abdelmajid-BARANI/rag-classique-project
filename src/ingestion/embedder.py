"""
Embedder Module
Génère des embeddings à partir de textes en utilisant sentence-transformers (all-MiniLM-L6-v2)
"""
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
import torch


class BERTEmbedder:
    """Classe pour générer des embeddings avec sentence-transformers"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2", device: str = None):
        """
        Initialise le modèle sentence-transformers pour les embeddings
        
        Args:
            model_name: Nom du modèle à utiliser (défaut: paraphrase-multilingual-mpnet-base-v2, 768d, multilingue)
            device: Device à utiliser (cpu/cuda). Auto-détecté si None
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        logger.info(f"Chargement du modèle {model_name} sur {device}...")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.success(f"Modèle chargé. Dimension des embeddings: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Génère l'embedding d'un texte
        
        Args:
            text: Texte à encoder
            
        Returns:
            Vecteur d'embedding
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Génère les embeddings pour un batch de textes
        
        Args:
            texts: Liste de textes à encoder
            batch_size: Taille du batch
            show_progress: Afficher la barre de progression
            
        Returns:
            Matrice d'embeddings
        """
        try:
            logger.info(f"Génération des embeddings pour {len(texts)} textes...")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            logger.success(f"Embeddings générés: shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage batch: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Génère les embeddings pour une liste de chunks
        
        Args:
            chunks: Liste de chunks avec leur texte
            batch_size: Taille du batch
            
        Returns:
            Liste de chunks enrichie avec les embeddings
        """
        if not chunks:
            logger.warning("Aucun chunk à encoder")
            return []
        
        # Extraire les textes
        texts = [chunk["text"] for chunk in chunks]
        
        # Générer les embeddings
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        
        # Ajouter les embeddings aux chunks
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = chunk.copy()
            enriched_chunk["embedding"] = embeddings[i]
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def get_embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings"""
        return self.embedding_dim


if __name__ == "__main__":
    # Test de l'embedder avec all-MiniLM-L6-v2
    embedder = BERTEmbedder()  # all-MiniLM-L6-v2 par défaut
    test_texts = [
        "Ceci est un texte de test.",
        "Voici un autre exemple de texte."
    ]
    embeddings = embedder.embed_batch(test_texts)
    print(f"Forme des embeddings: {embeddings.shape}")  # Attendu: (2, 384)
