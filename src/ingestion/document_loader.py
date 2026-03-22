"""
Document Loader Module
Charge les documents PDF via MinerU (magic-pdf) — extraction avancée de tableaux complexes, images et structure
Utilise l'API Python MinerU directement (plus fiable que le subprocess CLI)
"""
import os
import tempfile
from typing import List, Dict
from pathlib import Path
from loguru import logger


class DocumentLoader:
    """Classe pour charger les documents PDF via MinerU (magic-pdf)"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Le répertoire {data_dir} n'existe pas")

    def load_pdf(self, pdf_path: Path, output_dir: str = None) -> Dict[str, str]:
        """
        Charge un fichier PDF via l'API Python MinerU et extrait son contenu en Markdown.

        Args:
            pdf_path: Chemin vers le fichier PDF
            output_dir: Répertoire de sortie (temporaire si None)

        Returns:
            Dictionnaire contenant le nom du fichier et son contenu Markdown
        """
        try:
            from magic_pdf.data.data_reader_writer import (
                FileBasedDataWriter,
                FileBasedDataReader,
            )
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.config.enums import SupportedPdfParseMethod
        except ImportError as e:
            raise ImportError(
                f"MinerU (magic-pdf) n'est pas installé ou mal configuré : {e}\n"
                "Exécutez : pip install 'magic-pdf[full]'"
            ) from e

        try:
            use_temp = output_dir is None
            tmp_obj = tempfile.TemporaryDirectory() if use_temp else None
            work_dir = Path(tmp_obj.name if use_temp else output_dir)

            pdf_stem = pdf_path.stem
            md_output_dir = work_dir / pdf_stem / "auto"
            image_dir = md_output_dir / "images"
            md_output_dir.mkdir(parents=True, exist_ok=True)
            image_dir.mkdir(parents=True, exist_ok=True)

            image_writer = FileBasedDataWriter(str(image_dir))
            md_writer = FileBasedDataWriter(str(md_output_dir))

            # Lire le PDF
            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(str(pdf_path))
            ds = PymuDocDataset(pdf_bytes)

            # Choisir le mode : texte natif ou OCR
            parse_method = ds.classify()
            logger.info(f"{pdf_path.name} → mode détecté : {parse_method.value}")
            ocr_flag = (parse_method == SupportedPdfParseMethod.OCR)

            # Ne pas passer ocr= via ds.apply() : selon la version de magic-pdf,
            # ds.apply remaps 'ocr' en 'apply_ocr' ou pas, ce qui cause des conflits.
            # On laisse doc_analyze utiliser son défaut et on choisit le pipe selon parse_method.
            infer_result = ds.apply(doc_analyze)
            if ocr_flag:
                pipe = infer_result.pipe_ocr_mode(image_writer)
            else:
                pipe = infer_result.pipe_txt_mode(image_writer)

            # Générer le Markdown
            md_filename = f"{pdf_stem}.md"
            pipe.dump_md(md_writer, md_filename, "images")

            md_file = md_output_dir / md_filename
            if not md_file.exists():
                # Chercher n'importe quel .md généré en fallback
                candidates = list(md_output_dir.rglob("*.md"))
                if not candidates:
                    raise FileNotFoundError(
                        f"Aucun fichier Markdown généré pour {pdf_path.name}"
                    )
                md_file = max(candidates, key=lambda f: f.stat().st_size)

            md_content = md_file.read_text(encoding="utf-8")

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
                    "parse_method": parse_method.value,
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
