"""
Document Loader Module
Charge les documents PDF via MinerU (magic-pdf) — extraction avancée de tableaux complexes, images et structure
"""
import os
import subprocess
import tempfile
from typing import List, Dict
from pathlib import Path
from loguru import logger


class DocumentLoader:
    """Classe pour charger les documents PDF via MinerU (magic-pdf)"""

    def __init__(self, data_dir: str):
        """
        Initialise le document loader

        Args:
            data_dir: Chemin vers le répertoire contenant les documents
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Le répertoire {data_dir} n'existe pas")

    def load_pdf(self, pdf_path: Path, output_dir: str = None) -> Dict[str, str]:
        """
        Charge un fichier PDF via MinerU et extrait son contenu en Markdown
        (texte, tableaux complexes, images/graphes détectés)

        Args:
            pdf_path: Chemin vers le fichier PDF
            output_dir: Répertoire de sortie MinerU (temporaire si None)

        Returns:
            Dictionnaire contenant le nom du fichier et son contenu Markdown
        """
        try:
            use_temp = output_dir is None
            tmp_obj = tempfile.TemporaryDirectory() if use_temp else None
            work_dir = tmp_obj.name if use_temp else output_dir
            os.makedirs(work_dir, exist_ok=True)

            result = subprocess.run(
                ["magic-pdf", "-p", str(pdf_path), "-o", work_dir, "-m", "auto"],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                raise RuntimeError(f"MinerU a échoué : {result.stderr.strip()}")

            # MinerU produit : <output_dir>/<pdf_stem>/auto/<pdf_stem>.md
            pdf_stem = pdf_path.stem
            md_dir = Path(work_dir) / pdf_stem / "auto"
            md_files = list(md_dir.glob("*.md"))

            if not md_files:
                raise FileNotFoundError(
                    f"Aucun fichier Markdown généré pour {pdf_path.name}"
                )

            md_content = md_files[0].read_text(encoding="utf-8")

            # Nettoyage du temporaire
            if use_temp and tmp_obj:
                tmp_obj.cleanup()

            logger.info(
                f"Document chargé via MinerU : {pdf_path.name} "
                f"({len(md_content)} caractères)"
            )

            return {
                "filename": pdf_path.name,
                "content": md_content,
                "metadata": {
                    "source": str(pdf_path),
                    "format": "markdown",
                },
            }
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {pdf_path}: {e}")
            raise

    def load_all_pdfs(self, output_dir: str = None) -> List[Dict[str, str]]:
        """
        Charge tous les fichiers PDF du répertoire via MinerU

        Args:
            output_dir: Répertoire de sortie MinerU partagé (temporaire si None)

        Returns:
            Liste de dictionnaires contenant les documents
        """
        documents = []
        pdf_files = list(self.data_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"Aucun fichier PDF trouvé dans {self.data_dir}")
            return documents

        logger.info(f"Chargement de {len(pdf_files)} fichiers PDF via MinerU...")

        for pdf_path in pdf_files:
            try:
                doc = self.load_pdf(pdf_path, output_dir=output_dir)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Impossible de charger {pdf_path.name}: {e}")
                continue

        logger.success(f"{len(documents)} documents chargés avec succès")
        return documents

    def get_document_stats(self, documents: List[Dict[str, str]]) -> Dict:
        """
        Calcule des statistiques sur les documents chargés

        Args:
            documents: Liste des documents

        Returns:
            Dictionnaire avec les statistiques
        """
        total_chars = sum(len(doc["content"]) for doc in documents)
        total_words = sum(len(doc["content"].split()) for doc in documents)

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chars_per_doc": total_chars / len(documents) if documents else 0,
            "avg_words_per_doc": total_words / len(documents) if documents else 0,
        }


if __name__ == "__main__":
    # Test du loader
    loader = DocumentLoader("./donnees rag")
    docs = loader.load_all_pdfs()
    stats = loader.get_document_stats(docs)
    print(stats)
