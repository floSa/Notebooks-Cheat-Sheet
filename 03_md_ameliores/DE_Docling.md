---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# 📄 Docling — Parsing PDF / Docs intelligent
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur **Docling** (IBM, 2024+) — librairie open-source qui transforme des **PDFs, docs Office, images** en représentations structurées (JSON, Markdown) prêtes pour le RAG ou les pipelines ML.

C'est devenu en 2025-2026 une référence pour le **document AI** open-source, face à **LlamaParse** (SaaS), **Unstructured**, ou **Marker**.

Couverture :

1. **Le problème** : pourquoi PyPDF / pdfplumber ne suffisent pas.
2. **Architecture Docling** : modèles de layout + OCR + tables.
3. **Installation et premier parse**.
4. **Conversion vers Markdown** (pour RAG).
5. **Extraction structurée** : tables, images, code blocks.
6. **Pipeline RAG complet** avec Docling + LlamaIndex.
7. **Comparatif** alternatives 2026.
8. **Bonnes pratiques** : OCR, pré-traitement, chunking par section.

Dataset : PDFs sample (cf `00_datasets.md` — `data/_shared/pdfs_sample/`).
<!-- #endregion -->

<!-- #region -->
## 1. Le problème — pourquoi PyPDF ne suffit pas
<!-- #endregion -->

<!-- #region -->
**PDF est un format d'impression**, pas de structure logique. Un PDF dit "tracer ce texte ici, cette image là", pas "ceci est un titre, cela un tableau".

**Résultat** avec PyPDF / pdfplumber brut :

- Texte extrait dans le mauvais **ordre de lecture** (multi-colonnes mélangées).
- **Tableaux** récupérés comme des suites de cellules sans structure.
- **Images** ignorées.
- Pas de notion de **sections / titres / hiérarchie**.
- Catastrophe sur les **PDFs scannés** (besoin OCR).

**Docling** résout ça avec un **pipeline de modèles ML** :

1. **Layout analyzer** (`DocLayNet` model) — détecte titres, paragraphes, tables, figures, captions.
2. **Reading order** — ordonne les blocs logiquement.
3. **Table structure** — extrait les tables en format structuré.
4. **OCR** automatique sur images / pages scannées (Tesseract / EasyOCR).
5. **Export** : JSON structuré, Markdown, HTML.
<!-- #endregion -->

<!-- #region -->
## 2. Installation
<!-- #endregion -->

<!-- #region -->
```bash
# Lib légère + modèles téléchargés à la 1ère utilisation
uv add docling

# Pour OCR ajout :
# Tesseract via apt/brew, ou Pillow + easyocr (pip install easyocr)
```

> Au premier `DocumentConverter().convert(...)`, Docling télécharge ~500 MB de modèles (DocLayNet + TableFormer). Persiste dans le cache HF.
<!-- #endregion -->

<!-- #region -->
## 3. Premier parse
<!-- #endregion -->

```python
# Pseudo-code (deps non installées par défaut dans cet env)
"""
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2206.01062"   # ou chemin local
converter = DocumentConverter()
result = converter.convert(source)

doc = result.document
print(doc.export_to_markdown()[:1000])
"""
```

<!-- #region -->
## 4. Export en Markdown pour RAG
<!-- #endregion -->

<!-- #region -->
**Markdown** est le format de chunking idéal pour le RAG : structure préservée (titres, listes, tables), lisible humainement, splittable par section.

```python
"""
md = doc.export_to_markdown()

# Split intelligent par section (vs split brut par chars)
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "section"), ("##", "subsection")]
)
chunks = splitter.split_text(md)

for c in chunks[:3]:
    print(c.metadata, c.page_content[:200], "...")
"""
```

Cette approche préserve le **contexte structurel** (chaque chunk sait qu'il appartient à telle section).
<!-- #endregion -->

<!-- #region -->
## 5. Extraction structurée
<!-- #endregion -->

<!-- #region -->
### 5.1 Tables
<!-- #endregion -->

<!-- #region -->
```python
"""
# Iterate sur les éléments de type table
for table in doc.tables:
    df = table.export_to_dataframe()   # pandas DataFrame !
    print(df.head())
"""
```

Docling utilise **TableFormer** (IBM, 2022) — modèle Transformer dédié à la reconstruction de structure tabulaire. Bien meilleur que pdfplumber / camelot sur les tables complexes (cellules fusionnées, headers multi-lignes).
<!-- #endregion -->

<!-- #region -->
### 5.2 Images et figures
<!-- #endregion -->

<!-- #region -->
```python
"""
for fig in doc.pictures:
    image_pil = fig.get_image()       # PIL Image
    caption = fig.caption_text         # caption automatique si détectée
    bbox = fig.bbox                    # position dans la page
    fig.save_to_disk("figs/fig.png")
"""
```
<!-- #endregion -->

<!-- #region -->
### 5.3 Listing du JSON structuré
<!-- #endregion -->

<!-- #region -->
```python
"""
import json
print(json.dumps(doc.export_to_dict(), indent=2)[:500])

# Structure :
# {
#   "pages": [...],
#   "main_text": [
#     {"label": "title", "text": "Document title", ...},
#     {"label": "section_header", "text": "Introduction", ...},
#     {"label": "paragraph", "text": "...", ...},
#     {"label": "table", "data": [[...]], ...},
#     ...
#   ],
#   "tables": [...],
#   "pictures": [...]
# }
"""
```
<!-- #endregion -->

<!-- #region -->
## 6. Pipeline RAG complet — Docling + LlamaIndex
<!-- #endregion -->

<!-- #region -->
```python
"""
from docling.document_converter import DocumentConverter
from llama_index.readers.docling import DoclingReader
from llama_index.core import VectorStoreIndex

# Loader Docling natif pour LlamaIndex (depuis fin 2024)
reader = DoclingReader()
docs = reader.load_data(file_path="paper.pdf")

# Index vectoriel
index = VectorStoreIndex.from_documents(docs)
qe = index.as_query_engine()
print(qe.query("Quel est le contribution principale ?"))
"""
```

L'intégration garantit que le chunking respecte la structure du document (sections, tables, captions) plutôt qu'un split brut par caractères.
<!-- #endregion -->

<!-- #region -->
## 7. Comparatif 2026
<!-- #endregion -->

<!-- #region -->
| Outil | Type | Forces | Faiblesses |
|---|---|---|---|
| **Docling** (IBM) | OSS | **Top OSS 2026**, modèles ML intégrés, tables, OCR, MD export, intégration LlamaIndex/LangChain | DL → CPU lent, GPU recommandé pour volume |
| **LlamaParse** (LlamaIndex) | SaaS | Très bonne qualité, API simple | Payant, lock-in |
| **Unstructured** | OSS + SaaS | Multi-format (HTML, email, ...) | Qualité tables PDF moyenne |
| **Marker** | OSS (Ramachandran) | Très bonne qualité PDF→MD, batch | Plus de friction setup |
| **GPT-4V / Claude Sonnet** | LLM | Tout-en-un, comprend layout | Cher, lent, hallucinations possibles sur tables |
| **PyMuPDF / pdfplumber** | OSS | Très rapide, simple | Pas de compréhension structurelle |
| **AWS Textract** / **GCP Document AI** | SaaS cloud | OCR + tables top qualité | $$$, lock-in cloud |

**Recommandation 2026** :

- **POC / docs simples** : Docling.
- **Volume + qualité critique** : LlamaParse ou Docling sur GPU.
- **Stack 100 % OSS / on-prem** : Docling (1er choix), Marker (alternatif).
- **Cas légal / médical avec OCR critique** : AWS Textract / GCP Document AI.
<!-- #endregion -->

<!-- #region -->
## 8. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### 8.1 OCR
<!-- #endregion -->

<!-- #region -->
- Docling détecte automatiquement les pages scannées vs natives.
- Pour OCR de qualité : **Tesseract 5** (default Docling) ou **EasyOCR**, ou **Surya** (le meilleur en 2025-2026, multi-langues).
- Pré-traiter : redresser scans, augmenter contraste, dé-skew via `opencv`.
<!-- #endregion -->

<!-- #region -->
### 8.2 Chunking
<!-- #endregion -->

<!-- #region -->
- **Toujours splitter par structure logique** (sections, paragraphes) via le Markdown export.
- **Overlap** de 100-200 tokens entre chunks pour éviter de couper du contexte au milieu.
- **Conserver les métadonnées** : `{source, page, section, doc_id}` — indispensable pour la citation en RAG.
<!-- #endregion -->

<!-- #region -->
### 8.3 Volume
<!-- #endregion -->

<!-- #region -->
- Docling sur **CPU** : 5-15s par page (lent sur 1000 pages).
- Sur **GPU** : 10-50× plus rapide.
- Pour batch : **paralléliser** par fichier (1 process par PDF) ou utiliser le mode batch natif.
- Cacher les résultats : un PDF n'a pas besoin d'être reparsé entre deux appels (hash sur le contenu).
<!-- #endregion -->

<!-- #region -->
### 8.4 Évaluation qualité
<!-- #endregion -->

<!-- #region -->
- Constituer un set de PDFs "vérité" avec sortie attendue.
- Mesurer : taux de tables correctement extraites, taux de chars OCRisés correctement, taux de sections détectées.
- Comparer périodiquement à de nouvelles versions Docling — les modèles évoluent.
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [Docling — GitHub](https://github.com/DS4SD/docling)
- [Docling paper (2024)](https://arxiv.org/abs/2408.09869)
- [DocLayNet — IBM dataset / model](https://github.com/DS4SD/DocLayNet)
- [TableFormer paper](https://arxiv.org/abs/2203.01017)
- [LlamaIndex DoclingReader docs](https://docs.llamaindex.ai/en/stable/api_reference/readers/docling/)
- [Comparatif RAG document parsers — Galileo blog 2025](https://www.galileo.ai/blog)
- Notebooks liés : `NLP_Recherche_d_informations` (RAG), `BDD_Vectorielles`.
<!-- #endregion -->
