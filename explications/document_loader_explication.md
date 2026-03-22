# Document Loader — Explication

**Fichier :** `src/ingestion/document_loader.py`  
**Rôle :** Charger les fichiers PDF et en extraire le contenu textuel structuré.

---

## 1. Responsabilité

Le `DocumentLoader` est le premier maillon de la chaîne d'ingestion. Il prend un fichier PDF en entrée et retourne un dictionnaire contenant :
- le nom du fichier
- le texte extrait (avec une structure conservée au maximum)
- des métadonnées (nombre de pages, taille, chemin source)

---

## 2. Technologie : Unstructured

Le chargeur utilise la bibliothèque **Unstructured** (`partition_pdf`) avec la stratégie `"fast"`.

### Pourquoi Unstructured ?

| Critère | pdfplumber (ancienne solution) | Unstructured (solution actuelle) |
|---------|-------------------------------|----------------------------------|
| Texte simple | Correct | Correct |
| **Tableaux** | Texte brut mélangé | Détecte la structure, retourne du HTML |
| **Figures/titres** | Tout mélangé | Classe les éléments (Title, Table, Image, etc.) |
| Robustesse | Fragile sur PDFs complexes | Gère les PDFs scannés (OCR), les colonnes, etc. |

### Stratégie `"fast"`

La stratégie `"fast"` extrait le texte **sans OCR** (pas de Tesseract requis). Elle est parfaite quand les PDFs sont numériques (non scannés), ce qui est le cas des documents officiels du CGI et des directives fiscales.

---

## 3. Traitement des Tableaux

Unstructured retourne les tableaux sous forme de **HTML** (`<table>...</table>`). La méthode privée `_html_table_to_text()` convertit ce HTML en une version lisible :

```
Avant (HTML) :
<table><tr><td>Format</td><td>Norme</td></tr>...</table>

Après (texte) :
[TABLEAU]
Format | Norme
UBL | ISO/IEC ...
[/TABLEAU]
```

Les balises `[TABLEAU]...[/TABLEAU]` permettent :
1. Au LLM de comprendre qu'il s'agit d'un tableau
2. Aux chunks suivants de ne pas couper un tableau en deux

---

## 4. Traitement des Figures

Les éléments de type `Image` dans Unstructured sont encadrés par `[FIGURE]...[/FIGURE]`. Si la figure a une description ou un titre, il est conservé.

---

## 5. Interface Publique

### `load_pdf(pdf_path: str) -> Dict`

Charge un seul fichier PDF.

**Retourne :**
```python
{
    "filename": "nom_du_fichier.pdf",
    "content": "Texte extrait complet avec balises [TABLEAU]...[/TABLEAU]",
    "metadata": {
        "source": "/chemin/absolu/fichier.pdf",
        "num_pages": 42,
        "file_size": 123456
    }
}
```

### `load_all_pdfs(pdf_dir: str) -> List[Dict]`

Charge tous les PDFs d'un dossier et retourne une liste de dictionnaires (un par fichier).

---

## 6. Gestion des Erreurs

- Si un PDF ne peut pas être lu (fichier corrompu, protégé), le loader l'ignore avec un log d'erreur et continue avec les autres fichiers.
- Si Unstructured ne retourne aucun élément, le document est ignoré.

---

## 7. Exemple de Sortie

Pour un article du CGI sur la facturation électronique :

```
Art. 289 bis du CGI - Facturation électronique

I. Les assujettis établis en France sont tenus de...

[TABLEAU]
Plateforme | Type | Certifiée
PDP A | Partenaire | Oui
PPF | Portail Public | Oui
[/TABLEAU]

II. Les conditions sont définies par décret...
```

---

## 8. Position dans le Pipeline

```
PDFs  ──►  [DocumentLoader]  ──►  Chunker  ──►  Embedder  ──►  FAISS
```

Le DocumentLoader reçoit des PDFs bruts et produit du texte structuré prêt à être découpé.
