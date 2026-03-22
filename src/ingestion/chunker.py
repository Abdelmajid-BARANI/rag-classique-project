"""
Chunker Module
Découpe les documents en chunks via RecursiveCharacterTextSplitter (langchain)
  chunk_size    = 1000 caractères
  chunk_overlap = 200  caractères
"""
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class DocumentChunker:
    """Découpe les documents en chunks avec RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        # Séparateurs hiérarchiques : paragraphes → phrases → mots → caractères
        separators: List[str] = None,
    ):
        """
        Initialise le chunker.

        Args:
            chunk_size:    Taille maximale d'un chunk en caractères (défaut 1000).
            chunk_overlap: Chevauchement entre deux chunks consécutifs (défaut 200).
            separators:    Liste de séparateurs hiérarchiques.  Si None, utilise
                           les séparateurs par défaut de RecursiveCharacterTextSplitter.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        splitter_kwargs: Dict = dict(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        if separators is not None:
            splitter_kwargs["separators"] = separators

        self._splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)

        logger.info(
            f"DocumentChunker initialisé — RecursiveCharacterTextSplitter: "
            f"chunk_size={chunk_size} chars, overlap={chunk_overlap} chars"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Découpe un texte en chunks.

        Args:
            text:     Texte à découper.
            metadata: Métadonnées à ajouter à chaque chunk.

        Returns:
            Liste de dicts {text, chunk_id, start_char, end_char, metadata}.
        """
        text = text.strip()
        if not text:
            return []

        raw_chunks = self._splitter.create_documents([text])
        chunks = []
        for chunk_id, doc in enumerate(raw_chunks):
            chunks.append({
                "text": doc.page_content,
                "chunk_id": chunk_id,
                "n_chars": len(doc.page_content),
                "metadata": dict(metadata) if metadata else {},
            })
        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Découpe une liste de documents en chunks.

        Args:
            documents: Liste de dicts avec au moins les clés 'content' et 'filename'.

        Returns:
            Liste de tous les chunks.
        """
        all_chunks: List[Dict] = []

        for doc in documents:
            metadata = dict(doc.get("metadata", {}))
            metadata["filename"] = doc.get("filename", "unknown")

            chunks = self.chunk_text(doc["content"], metadata)
            all_chunks.extend(chunks)

            logger.debug(f"{doc.get('filename')}: {len(chunks)} chunks créés")

        logger.success(
            f"Total de {len(all_chunks)} chunks créés depuis {len(documents)} documents"
        )
        return all_chunks

    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        Calcule des statistiques sur les chunks.

        Args:
            chunks: Liste des chunks produits par chunk_documents / chunk_text.

        Returns:
            Dictionnaire de statistiques.
        """
        if not chunks:
            return {}

        lengths = [c.get("n_chars", len(c["text"])) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "chunk_mode": "chars (RecursiveCharacterTextSplitter)",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "avg_chunk_chars": round(sum(lengths) / len(lengths), 1),
            "min_chunk_chars": min(lengths),
            "max_chunk_chars": max(lengths),
            "total_chars": sum(lengths),
        }


if __name__ == "__main__":
    # Test rapide
    test_text = ("Ceci est un texte de test. " * 40).strip()
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_text(test_text)
    print(f"Nombre de chunks: {len(chunks)}")
    for c in chunks:
        print(f"  chunk {c['chunk_id']}: {c['n_chars']} chars — {c['text'][:80]}...")
