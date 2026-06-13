---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python (notebooks-refonte)
    language: python
    name: notebooks-refonte
---

<!-- #region -->
# Docling — Parsing de documents pour l'IA
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel + Wiki** sur **Docling**, la bibliothèque open-source d'IBM qui convertit des documents (PDF, DOCX, HTML, images) en une représentation structurée exploitable par des pipelines d'IA.

**Ce que couvre ce notebook** :

**Partie A — Tutoriel** : conversion d'un PDF, exploration du modèle `DoclingDocument` (ordre de lecture, bounding boxes, tables, images, captions), exports (Markdown / HTML / JSON) et découpage en *chunks* pour le RAG.

**Partie B — Pipeline d'extraction structurée** : un composant **typé et modulaire** qui produit, à partir d'un `DoclingDocument`, une liste d'**éléments** enrichis et de **relations** (hiérarchie section → contenu, liaison caption ↔ table/image) destinés à une ingestion *downstream* (orchestrateur, base, *knowledge graph*). Les deux heuristiques (hiérarchie, caption) sont isolées pour être **améliorées** indépendamment.

**Dataset** : le papier arXiv de Docling lui-même (`2408.09869`), riche en sections numérotées, tables, figures, formules et références — téléchargé une fois dans `data/DE_Docling/`.

**Versions** : Docling 2.96, docling-core 2.78 (2026).
<!-- #endregion -->

<!-- #region -->
## 1. Docling : positionnement 2026
<!-- #endregion -->

<!-- #region -->
**Docling** (IBM, open-source) transforme un document en un objet **`DoclingDocument`** : un arbre d'éléments typés (titres, sections, paragraphes, tables, images, formules, captions) accompagnés de leur **provenance** (numéro de page + *bounding box*) et de leur **ordre de lecture**.

Contrairement à une simple extraction de texte, Docling reconstruit la **structure** du document :

- détection de la mise en page (layout) par des modèles de vision ;
- reconstruction de la structure des **tables** (TableFormer) ;
- ordre de lecture cohérent même en multi-colonnes ;
- enrichissements optionnels : OCR pour les scans, classification d'images, description d'images par VLM, reconnaissance de formules et de code.

Lancé en 2024, Docling est en développement très actif (releases quasi mensuelles) ; vérifier la version installée car l'API évolue.
<!-- #endregion -->

<!-- #region -->
## 2. Docling dans l'écosystème
<!-- #endregion -->

<!-- #region -->
Quel outil pour quel besoin ?

| Outil | Texte | Layout | Tables | OCR scans | Sortie structurée | Quand l'utiliser |
|---|---|---|---|---|---|---|
| **PyPDF / pypdfium** | oui | non | non | non | brut | extraire vite du texte d'un PDF numérique |
| **pdfplumber** | oui | partiel | oui (géométrie) | non | tableaux | tables simples, accès aux coordonnées |
| **unstructured** | oui | oui | oui | oui | éléments | ingestion multi-formats hétérogènes |
| **marker** | oui | oui | oui | oui | Markdown | conversion PDF vers Markdown propre |
| **Docling** | oui | oui | oui (TableFormer) | oui | `DoclingDocument` + chunking | pipelines RAG / IA avec structure riche et provenance |

Docling se distingue par un **modèle de document unifié** (`DoclingDocument`) qui conserve la provenance et alimente directement un *chunker* pensé pour le RAG.
<!-- #endregion -->

<!-- #region -->
## 3. Installation et imports
<!-- #endregion -->

<!-- #region -->
Installation via uv : `uv add docling docling-core pymupdf rich`. PyMuPDF (`fitz`) sert à rendre les pages en image pour visualiser les *bounding boxes*. Hors projet uv, on peut utiliser la cellule `%pip` commentée ci-dessous.

Au **premier run**, Docling télécharge ses modèles de layout et TableFormer (mis en cache ensuite). On centralise ici les imports et les dossiers de sortie.
<!-- #endregion -->

```python
from __future__ import annotations

# %pip install -q docling docling-core pymupdf rich  # hors environnement uv

import json
import hashlib
import re
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import fitz  # PyMuPDF

from rich.console import Console
from rich.table import Table
from IPython.display import Image as IPyImage, display

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.labels import DocItemLabel

# Dossiers de sortie (données + images extraites)
DATA_DIR = Path("data/DE_Docling")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

console = Console()
```

<!-- #region -->
## 4. Conversion d'un PDF
<!-- #endregion -->

<!-- #region -->
On télécharge le PDF de test **une seule fois** et on le réutilise ensuite (conversion Docling *et* rendu PyMuPDF), pour éviter des appels réseau répétés.
<!-- #endregion -->

```python
PDF_URL = "https://arxiv.org/pdf/2408.09869.pdf"  # le papier Docling lui-même


def ensure_sample_pdf(url: str = PDF_URL, dest_dir: Path = DATA_DIR) -> Path:
    """Télécharge le PDF de démo une seule fois et renvoie son chemin local.

    Args:
        url: URL du PDF à récupérer.
        dest_dir: dossier de destination.

    Returns:
        Chemin local du PDF (téléchargé si absent, réutilisé sinon).
    """
    dest = dest_dir / "docling_paper.pdf"
    if not dest.exists():
        print(f"Téléchargement du PDF de test -> {dest}")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"PDF de test déjà présent : {dest}")
    return dest


pdf_path = ensure_sample_pdf()
```

<!-- #region -->
On configure le pipeline PDF puis on convertit. Deux options méritent attention :

- `table_structure_options=TableStructureOptions(mode=TableFormerMode.FAST)` : reconstruit la structure des tables (mode rapide ; `ACCURATE` est plus lent mais plus précis).
- `generate_picture_images=True` + `images_scale=2.0` : **indispensable** pour récupérer ensuite les images via `picture.get_image(doc)` — sans cette option, Docling n'enregistre que les positions, pas les pixels.

La limitation de pages se fait sur `convert(..., page_range=(début, fin))` (1-indexé), ce qui borne le temps de calcul.
<!-- #endregion -->

```python
PAGE_RANGE = (1, 9)  # borne le temps de calcul

pipeline_options = PdfPipelineOptions(
    do_ocr=False,                  # PDF numérique : pas besoin d'OCR
    do_table_structure=True,       # reconstruction de la structure des tables
    table_structure_options=TableStructureOptions(mode=TableFormerMode.FAST),
    generate_picture_images=True,  # clé : sans ça, pic.get_image() renvoie None
    images_scale=2.0,              # 2.0 = ~144 DPI (1.0 = 72 DPI)
)

converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF, InputFormat.HTML, InputFormat.DOCX],
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    },
)

print(f"Conversion des pages {PAGE_RANGE} (1er run : téléchargement des modèles)...")
result = converter.convert(pdf_path, page_range=PAGE_RANGE)
doc = result.document
filename = pdf_path.stem
```

<!-- #region -->
Un coup d'œil aux statistiques globales du document converti.
<!-- #endregion -->

```python
print("Document converti.")
print(f"  Pages    : {doc.num_pages()}")
print(f"  Textes   : {len(doc.texts)}")
print(f"  Tables   : {len(doc.tables)}")
print(f"  Pictures : {len(doc.pictures)}")
```

<!-- #region -->
## 5. Explorer le DoclingDocument
<!-- #endregion -->

<!-- #region -->
### 5.1 Ordre de lecture
<!-- #endregion -->

<!-- #region -->
`doc.iterate_items()` parcourt les éléments dans l'**ordre de lecture** reconstruit. Chaque élément porte son type, son texte éventuel et sa provenance (page + bbox).
<!-- #endregion -->

```python
def reading_order_table(doc, limit: int = 15) -> Table:
    """Construit une table rich des premiers éléments dans l'ordre de lecture.

    Args:
        doc: DoclingDocument converti.
        limit: nombre maximal d'éléments affichés.

    Returns:
        Une rich.Table prête à être imprimée.
    """
    table = Table(title="Reading order (premiers éléments)")
    table.add_column("Ordre", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Texte (extrait)", style="yellow")
    table.add_column("Page", style="magenta")

    for idx, (item, _level) in enumerate(doc.iterate_items()):
        if idx >= limit:
            break
        text = getattr(item, "text", "") or ""
        preview = text[:50].replace("\n", " ") if text else f"<{type(item).__name__}>"
        page_no = item.prov[0].page_no if getattr(item, "prov", None) else "N/A"
        table.add_row(str(idx), type(item).__name__, preview, str(page_no))
    return table


console.print(reading_order_table(doc))
```

<!-- #region -->
### 5.2 Bounding boxes
<!-- #endregion -->

<!-- #region -->
On rend une page en image avec PyMuPDF, puis on superpose les *bounding boxes* des éléments Docling, colorées par type. Les coordonnées PDF (origine en bas à gauche) sont converties en coordonnées image (origine en haut à gauche). La figure est sauvegardée puis affichée.
<!-- #endregion -->

```python
# Palette par type d'élément (couleurs distinctes, contrastées)
BBOX_COLORS: dict[DocItemLabel, str] = {
    DocItemLabel.TEXT: "#00798c",
    DocItemLabel.TITLE: "#2e4057",
    DocItemLabel.SECTION_HEADER: "#66a182",
    DocItemLabel.LIST_ITEM: "#9d83b8",
    DocItemLabel.TABLE: "#edae49",
    DocItemLabel.PICTURE: "#d1495b",
    DocItemLabel.CODE: "#b8848e",
    DocItemLabel.FORMULA: "#c9b78b",
    DocItemLabel.CAPTION: "#2e4057",
}


def plot_bboxes(doc, pdf_path: Path, page_no: int, out_path: Path,
                max_items: int = 60) -> Path:
    """Dessine les bounding boxes Docling sur la page PDF rendue (PyMuPDF).

    Args:
        doc: DoclingDocument converti.
        pdf_path: chemin du PDF source (pour le rendu pixel).
        page_no: numéro de page (1-indexé).
        out_path: chemin de sauvegarde du PNG.
        max_items: nombre maximal de boîtes dessinées.

    Returns:
        Le chemin du PNG sauvegardé.
    """
    import io
    from PIL import Image

    pdf = fitz.open(pdf_path)
    page = pdf[page_no - 1]
    zoom = 2.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    height_pt = page.rect.height

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(img, origin="upper")
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.set_title(f"Bounding boxes — page {page_no}")
    ax.axis("off")

    items = list(doc.texts) + list(doc.tables) + list(doc.pictures)
    drawn = 0
    for item in items:
        if drawn >= max_items:
            break
        prov = getattr(item, "prov", None)
        if not prov or prov[0].page_no != page_no:
            continue
        bbox = prov[0].bbox
        label = getattr(item, "label", DocItemLabel.TEXT)
        color = BBOX_COLORS.get(label, "#00798c")

        # Coords PDF (origine bas-gauche) -> pixels image (origine haut-gauche)
        x = bbox.l * zoom
        y = (height_pt - bbox.t) * zoom
        w = (bbox.r - bbox.l) * zoom
        h = (bbox.t - bbox.b) * zoom
        ax.add_patch(
            patches.Rectangle((x, y), w, h, linewidth=1.8, edgecolor=color,
                              facecolor="none")
        )
        ax.text(x, y - 4, label.value.upper(), fontsize=6, color=color,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.4))
        drawn += 1

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    pdf.close()
    return out_path


bbox_png = plot_bboxes(doc, pdf_path, page_no=5, out_path=FIG_DIR / "bboxes_page5.png")
display(IPyImage(filename=str(bbox_png)))
```

<!-- #region -->
### 5.3 Éléments de page
<!-- #endregion -->

<!-- #region -->
Taille des pages et éléments de type *page header* / *page footer* détectés.
<!-- #endregion -->

```python
def show_page_elements(doc) -> None:
    """Affiche la taille des pages et les headers/footers détectés."""
    print("Pages :")
    for pno, page in doc.pages.items():
        print(f"  page {pno}: {page.size.width:.0f} x {page.size.height:.0f} pts")

    def _label(item) -> str:
        return str(getattr(item, "label", "")).lower()

    headers = [t for t in doc.texts if "page_header" in _label(t)]
    footers = [t for t in doc.texts if "page_footer" in _label(t)]
    print(f"Page headers : {len(headers)} | Page footers : {len(footers)}")
    for h in headers[:3]:
        print(f"  header: {getattr(h, 'text', '').strip()[:60]!r}")
    for f in footers[:3]:
        print(f"  footer: {getattr(f, 'text', '').strip()[:60]!r}")


show_page_elements(doc)
```

<!-- #region -->
### 5.4 Éléments de texte
<!-- #endregion -->

<!-- #region -->
On inspecte les labels **réellement** présents. Sur un PDF, Docling émet surtout `TEXT` et `SECTION_HEADER` : le label `PARAGRAPH` existe dans l'énumération mais n'est quasiment jamais produit par le parsing PDF (chercher `PARAGRAPH` directement renverrait donc des listes vides — un piège fréquent).
<!-- #endregion -->

```python
def inspect_text_labels(doc) -> dict[str, int]:
    """Renvoie la distribution des labels réellement présents dans doc.texts."""
    from collections import Counter

    counts = Counter(str(getattr(t, "label", "None")).split(".")[-1] for t in doc.texts)
    print("Distribution des labels dans doc.texts :")
    for lbl, n in counts.most_common():
        print(f"  {lbl}: {n}")
    return dict(counts)


text_label_counts = inspect_text_labels(doc)

print("\nExtrait de SECTION_HEADER :")
shown = 0
for t in doc.texts:
    if getattr(t, "label", None) == DocItemLabel.SECTION_HEADER and shown < 5:
        print(f"  - {getattr(t, 'text', '').strip()[:70]}")
        shown += 1
```

<!-- #region -->
### 5.5 Tables
<!-- #endregion -->

<!-- #region -->
Pour chaque table : sa bbox, ses dimensions (lignes × colonnes × cellules), un aperçu via `export_to_dataframe()` (pont direct vers pandas) et son caption.
<!-- #endregion -->

```python
def show_tables(doc, max_tables: int = 3) -> None:
    """Affiche structure et aperçu DataFrame des premières tables."""
    if not doc.tables:
        print("Aucune table détectée sur la plage de pages.")
        return
    for idx, tbl in enumerate(doc.tables[:max_tables]):
        print(f"\nTable #{idx + 1}")
        if tbl.prov:
            b = tbl.prov[0].bbox
            print(f"  bbox: [{b.l:.0f}, {b.t:.0f}, {b.r:.0f}, {b.b:.0f}]")
        data = getattr(tbl, "data", None)
        if data is not None:
            print(f"  dimensions: {data.num_rows}r x {data.num_cols}c "
                  f"({len(data.table_cells)} cells)")
        try:
            df = tbl.export_to_dataframe()
            print("  aperçu :")
            print(df.head(3).to_string(index=False)[:300])
        except Exception as exc:  # noqa: BLE001
            print(f"  export_to_dataframe indisponible : {exc}")
        cap = tbl.caption_text(doc)
        if cap and cap.strip():
            print(f"  caption: {cap[:70]!r}")


show_tables(doc)
```

<!-- #region -->
### 5.6 Images
<!-- #endregion -->

<!-- #region -->
Grâce à `generate_picture_images=True`, `picture.get_image(doc)` renvoie une `PIL.Image`. On les sauvegarde et on affiche les premières inline.
<!-- #endregion -->

```python
def export_pictures(doc, out_dir: Path, max_pictures: int = 5) -> list[Path]:
    """Sauvegarde les images extraites (nécessite generate_picture_images=True).

    Returns:
        Liste des chemins PNG écrits.
    """
    written: list[Path] = []
    if not doc.pictures:
        print("Aucune picture détectée sur la plage de pages.")
        return written
    for idx, pic in enumerate(doc.pictures[:max_pictures]):
        try:
            pil = pic.get_image(doc)
        except Exception as exc:  # noqa: BLE001
            print(f"Picture #{idx + 1} : get_image a échoué : {exc}")
            pil = None
        if pil is not None:
            dest = out_dir / f"picture_{idx + 1}.png"
            pil.save(dest)
            written.append(dest)
            print(f"Picture #{idx + 1} sauvegardée : {dest} ({pil.size[0]}x{pil.size[1]})")
        else:
            print(f"Picture #{idx + 1} : pas de données image")
    return written


picture_paths = export_pictures(doc, FIG_DIR)
for p in picture_paths[:3]:
    display(IPyImage(filename=str(p)))
```

<!-- #region -->
### 5.7 Captions
<!-- #endregion -->

<!-- #region -->
`caption_text(doc)` renvoie le texte du caption associé à une table ou une image (relation gérée nativement par Docling).
<!-- #endregion -->

```python
def show_captions(doc) -> None:
    """Liste les captions associés aux tables et images."""
    print("Captions de tables :")
    for t in doc.tables:
        cap = t.caption_text(doc)
        if cap and cap.strip():
            print(f"  - {cap[:80]}")
    print("Captions d'images :")
    for p in doc.pictures:
        cap = p.caption_text(doc)
        if cap and cap.strip():
            print(f"  - {cap[:80]}")


show_captions(doc)
```

<!-- #region -->
## 6. Exports et résumé par page
<!-- #endregion -->

<!-- #region -->
Docling exporte vers **Markdown**, **HTML** et un **dict JSON** structuré (le format pivot, qui conserve toute la structure et la provenance). On sauvegarde le JSON localement.
<!-- #endregion -->

```python
md_output = doc.export_to_markdown()
html_output = doc.export_to_html()
json_output = doc.export_to_dict()

print(f"Markdown : {len(md_output)} caractères")
print(f"HTML     : {len(html_output)} caractères")
print(f"JSON     : clés principales = {list(json_output.keys())[:8]}")

json_path = DATA_DIR / "docling_output.json"
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(json_output, fh, ensure_ascii=False, indent=2)
print(f"Export JSON sauvegardé : {json_path}")
```

<!-- #region -->
Un résumé du contenu d'une page (compte des éléments par type). Cette fonction statique remplace avantageusement un widget interactif : elle s'exécute partout (script, notebook, pipeline).
<!-- #endregion -->

```python
def page_summary(doc, page_no: int) -> str:
    """Renvoie un résumé texte des éléments d'une page (compte par type)."""
    from collections import Counter

    counts: Counter = Counter()
    for item, _ in doc.iterate_items():
        if item.prov and item.prov[0].page_no == page_no:
            counts[str(getattr(item, "label", "UNKNOWN")).split(".")[-1]] += 1
    lines = [f"Page {page_no} — {sum(counts.values())} éléments", ""]
    lines += [f"  {lbl}: {n}" for lbl, n in counts.most_common()]
    return "\n".join(lines)


print(page_summary(doc, 1))
```

<!-- #region -->
## 7. Chunking pour le RAG
<!-- #endregion -->

<!-- #region -->
Pour un index vectoriel, on découpe le document en *chunks* de taille maîtrisée. Le `HybridChunker` de Docling est **tokenization-aware** : il part du découpage hiérarchique du document puis fusionne / scinde les morceaux pour respecter une limite de tokens.

**API 2026** : le tokenizer se passe désormais comme un objet `HuggingFaceTokenizer` (et non plus une simple chaîne). L'avertissement éventuel sur la longueur de séquence est normal — c'est précisément ce que le chunker corrige en scindant.

Une fois les chunks obtenus, on les *embed* et on les indexe dans une base vectorielle — voir le notebook [[BDD_Vectorielles]] pour la suite (embeddings, FAISS / Qdrant / Chroma, recherche).
<!-- #endregion -->

```python
def build_chunks(doc, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens: int = 128) -> list:
    """Découpe le document en chunks tokenization-aware pour un index RAG."""
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name, max_tokens=max_tokens)
    chunker = HybridChunker(tokenizer=tokenizer)
    return list(chunker.chunk(dl_doc=doc))


chunks = build_chunks(doc)
print(f"{len(chunks)} chunks générés")
for i, ch in enumerate(chunks[:5]):
    pages = set()
    meta = getattr(ch, "meta", None)
    for di in getattr(meta, "doc_items", []) or []:
        for pr in getattr(di, "prov", []) or []:
            if getattr(pr, "page_no", None) is not None:
                pages.add(pr.page_no)
    preview = ch.text[:120].replace("\n", " ")
    print(f"  chunk #{i + 1} | pages {sorted(pages) or 'N/A'} | {preview}...")
```

<!-- #region -->
# Partie B — Pipeline d'extraction structurée
<!-- #endregion -->

<!-- #region -->
## 8. Objectif et schéma de sortie
<!-- #endregion -->

<!-- #region -->
Au-delà du texte, on veut produire une représentation **structurée** prête pour une ingestion *downstream* (orchestrateur type Dagster, base, *knowledge graph*, stockage objet) :

- une liste d'**éléments** : chacun porte un identifiant stable, sa page, son ordre global, son type/label, son texte, sa bbox, et des **références** (document, header de section parent, caption) ;
- une liste de **relations** typées : par exemple `HAS_CAPTION` entre une figure/table et son caption.

Deux heuristiques portent l'essentiel de la valeur, et sont donc **isolées** pour être améliorées indépendamment :

1. la **hiérarchie section → contenu** (quel header gouverne quel élément) ;
2. la **liaison caption ↔ table/image** (quel texte légende quelle figure).
<!-- #endregion -->

<!-- #region -->
## 9. Modèle de données typé
<!-- #endregion -->

<!-- #region -->
On remplace les `dict` ad hoc par des **dataclasses** typées : le contrat de sortie est explicite, vérifiable et facile à étendre (ajouter un champ = ajouter une ligne). `to_dict()` assure une sérialisation JSON propre.
<!-- #endregion -->

```python
@dataclass
class StructuredElement:
    """Élément de document enrichi pour ingestion downstream."""

    id: str
    filename: str
    page_no: int
    global_order: int
    type: str
    label: str
    text: Optional[str] = None
    bbox: Optional[dict[str, float]] = None
    document_ref: Optional[str] = None
    section_header_ref: Optional[str] = None
    parent_header_ref: Optional[str] = None
    caption_ref: Optional[str] = None
    minio_key: Optional[str] = None


@dataclass
class Relationship:
    """Relation typée entre deux éléments (graphe documentaire)."""

    source_id: str
    target_id: str
    relation_type: str  # ex: HAS_CAPTION, IN_SECTION
    source_label: str = ""


@dataclass
class ExtractionResult:
    """Sortie complète : metadata + éléments + relations + état header."""

    metadata: dict[str, Any]
    elements: list[StructuredElement] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    next_header_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Sérialise en dict JSON-compatible."""
        return {
            "metadata": self.metadata,
            "elements": [asdict(e) for e in self.elements],
            "relationships": [asdict(r) for r in self.relationships],
            "next_header_state": self.next_header_state,
        }
```

<!-- #region -->
## 10. Helpers de base
<!-- #endregion -->

<!-- #region -->
Trois utilitaires :

- `compute_element_id` : identifiant **stable et reproductible** (hash) — deux exécutions donnent le même id pour le même élément, condition nécessaire pour un *upsert* idempotent en base.
- `normalize_bbox` : bbox vers dict JSON-serializable.
- `parse_header_numbering` : extrait la numérotation d'un header (`3.2`, chiffres romains, lettres). Le **fix acronymes** évite de prendre « OCR » ou « C » pour un niveau de section.
<!-- #endregion -->

```python
def compute_element_id(filename: str, page_no: int, order: int, snippet: str) -> str:
    """ID stable et reproductible (hash) pour un élément."""
    payload = f"{filename}|{page_no}|{order}|{snippet[:50]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def normalize_bbox(bbox) -> dict[str, float]:
    """Normalise une bbox Docling en dict JSON-serializable."""
    return {"l": round(bbox.l, 2), "t": round(bbox.t, 2),
            "r": round(bbox.r, 2), "b": round(bbox.b, 2)}


_NUM_RE = re.compile(r"^([A-Z]{1,2}|[IVXLCDM]{1,4}|[0-9]+(?:\.[0-9]+)*)\s*[.\)]?\s*")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def parse_header_numbering(text: str) -> tuple[Optional[tuple], str]:
    """Parse la numérotation d'un header. '3.2 AI models' -> ((3, 2), 'AI models').

    Renvoie (None, text) si pas de numérotation (ex: 'OCR' — acronyme).
    """
    clean = text.strip()
    m = _NUM_RE.match(clean)
    if not m:
        return None, text
    num_str = m.group(1)
    remainder = text[m.end():].strip()
    parts: list[int] = []
    for p in num_str.split("."):
        if p.isdigit():
            parts.append(int(p))
        elif re.fullmatch(r"[IVXLCDM]+", p):
            total, prev = 0, 0
            for ch in reversed(p):
                v = _ROMAN[ch]
                total += v if v >= prev else -v
                prev = v
            parts.append(total)
        elif p.isalpha():
            if len(p) > 1:  # acronyme (OCR, APP...) -> pas une numérotation
                return None, text
            parts.append(ord(p.upper()) - ord("A") + 1)
        else:
            return None, text
    return tuple(parts), remainder
```

<!-- #region -->
## 11. Stratégie 1 — hiérarchie section vers contenu
<!-- #endregion -->

<!-- #region -->
**Point d'amélioration n°1.** On suit la hiérarchie via une **pile de headers numérotés** : un header `3.2` est enfant du header dont le tuple de numérotation est un préfixe strict (`3`). Tout contenu hérite du header au sommet de la pile.

Pourquoi la **numérotation** plutôt que les niveaux renvoyés par Docling ? Sur beaucoup de PDF, les niveaux de header sont mal détectés, alors que la numérotation `1`, `2.1`, `2.2`… est explicite et fiable. La pile est **sérialisable** (`export_state` / `load_state`) pour le traitement par fenêtres des longs documents.
<!-- #endregion -->

```python
class SectionHierarchyTracker:
    """Suit la hiérarchie des sections via une pile de headers numérotés.

    Stratégie isolée = point d'amélioration n°1. Un header '3.2' est enfant du
    header dont le tuple est un préfixe ('3'). Le contenu courant hérite du
    header au sommet de la pile. La pile est sérialisable pour le fenêtrage.
    """

    def __init__(self) -> None:
        self.stack: list[dict[str, Any]] = []  # [{id, tuple}]

    def update_with_header(self, elem_id: str, numbering: Optional[tuple]
                           ) -> Optional[str]:
        """Enregistre un header et renvoie l'id de son parent (ou None)."""
        parent_id: Optional[str] = None
        if numbering:
            for h in reversed(self.stack):
                if (numbering[: len(h["tuple"])] == h["tuple"]
                        and len(h["tuple"]) < len(numbering)):
                    parent_id = h["id"]
                    break
            # dépiler frères/descendants, empiler le nouveau header
            self.stack = [h for h in self.stack
                          if numbering[: len(h["tuple"])] == h["tuple"]
                          and len(h["tuple"]) < len(numbering)]
            self.stack.append({"id": elem_id, "tuple": numbering})
        return parent_id

    def current(self) -> Optional[str]:
        """Renvoie l'id du header actif (sommet de pile)."""
        return self.stack[-1]["id"] if self.stack else None

    def export_state(self) -> dict[str, Any]:
        """Exporte l'état pour la fenêtre suivante."""
        return {"stack": [dict(h) for h in self.stack]}

    def load_state(self, state: dict[str, Any]) -> None:
        """Restaure l'état depuis la fenêtre précédente."""
        self.stack = [dict(h) for h in state.get("stack", [])]
```

<!-- #region -->
## 12. Stratégie 2 — liaison caption vers table/image
<!-- #endregion -->

<!-- #region -->
**Point d'amélioration n°2.** Heuristique **géométrique** d'appariement caption vers figure/table :

- **alignement horizontal** : $|c_x^{item} - c_x^{caption}| <$ `max_center_dx`, où $c_x = (l + r)/2$ ;
- **proximité verticale** : caption juste *au-dessus* (typique des tables) ou juste *en dessous* (typique des figures), à moins de `max_gap` points.

Un caption consommé n'est plus réutilisé (`used`). Cette fonction pure est le levier d'amélioration : on pourrait y ajouter une validation sémantique (« Figure N » / « Table N »), gérer le multi-colonnes, etc.
<!-- #endregion -->

```python
def link_caption(item_bbox, page_captions: list[dict], used: set,
                 max_center_dx: float = 150.0, max_gap: float = 120.0
                 ) -> Optional[dict]:
    """Apparie un élément (table/image) à un caption par proximité géométrique.

    Args:
        item_bbox: bbox de la table/image (objet Docling).
        page_captions: captions de la même page [{id, bbox, text}].
        used: set d'ids de captions déjà consommés (1 caption -> 1 élément).

    Returns:
        Le caption apparié {id, text} ou None.
    """
    item_cx = (item_bbox.l + item_bbox.r) / 2
    for cap in page_captions:
        if cap["id"] in used:
            continue
        cb = cap["bbox"]
        cap_cx = (cb.l + cb.r) / 2
        if abs(cap_cx - item_cx) > max_center_dx:
            continue
        dist_below = item_bbox.b - cb.t   # caption sous l'élément (figures)
        dist_above = cb.b - item_bbox.t   # caption au-dessus (tables)
        if 0 < dist_below < max_gap or 0 < dist_above < max_gap:
            used.add(cap["id"])
            return {"id": cap["id"], "text": cap["text"][:100]}
    return None
```

<!-- #region -->
## 13. Fonction d'extraction
<!-- #endregion -->

<!-- #region -->
La fonction d'orchestration procède en **deux passes** :

1. attribuer à chaque élément un ordre global et un **id stable**, et indexer les captions par page (ainsi la relation `HAS_CAPTION` pointe vers l'id **réel** du caption — corrige une incohérence des versions précédentes) ;
2. construire les éléments et relations en appliquant les deux stratégies isolées ; les tables/images reçoivent en plus une `minio_key` (clé de stockage objet).

L'argument `header_state` permet de reprendre la hiérarchie d'une fenêtre précédente (longs documents).
<!-- #endregion -->

```python
_TEXT_LIKE = {"text", "paragraph", "code", "formula", "list_item"}
_VISUAL = {"table", "picture"}


def extract_structured_elements(doc, filename: str,
                                header_state: Optional[dict] = None
                                ) -> ExtractionResult:
    """Extrait éléments + relations structurés depuis un DoclingDocument.

    Args:
        doc: DoclingDocument converti.
        filename: identifiant du document (sert aux ids et minio_key).
        header_state: état de hiérarchie d'une fenêtre précédente (longs docs).

    Returns:
        ExtractionResult (metadata, elements, relationships, next_header_state).
    """
    # Passe 1 : assigner un ordre + un id stable + indexer les captions
    indexed: list[dict[str, Any]] = []
    captions_by_page: dict[int, list[dict]] = {}
    for order, (item, _level) in enumerate(doc.iterate_items()):
        prov = item.prov[0] if getattr(item, "prov", None) else None
        page_no = prov.page_no if prov else 1
        bbox = prov.bbox if prov else None
        text = (getattr(item, "text", "") or "").strip()
        label = str(getattr(item, "label", "text")).split(".")[-1].lower()
        eid = compute_element_id(filename, page_no, order, text)
        indexed.append({"item": item, "order": order, "id": eid,
                        "page_no": page_no, "bbox": bbox, "text": text,
                        "label": label})
        if label == "caption" and bbox is not None:
            captions_by_page.setdefault(page_no, []).append(
                {"id": eid, "bbox": bbox, "text": text}
            )

    # Passe 2 : éléments + relations
    tracker = SectionHierarchyTracker()
    if header_state:
        tracker.load_state(header_state)
    elements: list[StructuredElement] = []
    relationships: list[Relationship] = []
    used_captions: set = set()

    for rec in indexed:
        item, label, bbox = rec["item"], rec["label"], rec["bbox"]
        elem = StructuredElement(
            id=rec["id"], filename=filename, page_no=rec["page_no"],
            global_order=rec["order"], type=type(item).__name__, label=label,
            text=rec["text"] or None,
            bbox=normalize_bbox(bbox) if bbox is not None else None,
        )

        if label == "title":
            elem.document_ref = filename
        elif label == "section_header":
            numbering, _ = parse_header_numbering(rec["text"])
            parent = tracker.update_with_header(rec["id"], numbering)
            if parent:
                elem.parent_header_ref = parent
            else:
                elem.section_header_ref = tracker.current()
        elif label in _TEXT_LIKE:
            elem.section_header_ref = tracker.current()
        elif label in _VISUAL:
            elem.section_header_ref = tracker.current()
            elem.minio_key = f"{filename}/{label}s/{rec['id']}.png"
            if bbox is not None:
                cap = link_caption(bbox, captions_by_page.get(rec["page_no"], []),
                                   used_captions)
                if cap:
                    elem.caption_ref = cap["id"]
                    relationships.append(Relationship(
                        source_id=rec["id"], target_id=cap["id"],
                        relation_type="HAS_CAPTION", source_label=label,
                    ))

        elements.append(elem)

    return ExtractionResult(
        metadata={"filename": filename, "total_pages": doc.num_pages(),
                  "elements_count": len(elements),
                  "relationships_count": len(relationships)},
        elements=elements, relationships=relationships,
        next_header_state=tracker.export_state(),
    )
```

<!-- #region -->
## 14. Exécution et contrôle de cohérence
<!-- #endregion -->

<!-- #region -->
On exécute l'extraction et on contrôle : distribution des labels, aperçu page par page (chaque élément avec son header de rattachement), et liste des relations créées.
<!-- #endregion -->

```python
def print_coherence(res: ExtractionResult, max_per_page: int = 12) -> None:
    """Affiche un contrôle de cohérence : distribution, relations, par page."""
    from collections import Counter

    print(f"Extraction : {res.metadata['elements_count']} éléments | "
          f"{res.metadata['relationships_count']} relations")
    counts = Counter(e.label for e in res.elements)
    print("Distribution des labels :")
    for lbl, n in counts.most_common():
        print(f"  {lbl}: {n}")

    by_page: dict[int, list[StructuredElement]] = {}
    for e in res.elements:
        by_page.setdefault(e.page_no, []).append(e)
    for pn in sorted(by_page):
        elems = by_page[pn]
        print(f"\nPage {pn} ({len(elems)} éléments)")
        for e in elems[:max_per_page]:
            ref = e.section_header_ref or e.parent_header_ref or e.document_ref or "-"
            ref_disp = ref[:8] if isinstance(ref, str) and ref != "-" else ref
            print(f"  {e.id}  {e.label:<16} ref={ref_disp}")
        if len(elems) > max_per_page:
            print(f"  ... +{len(elems) - max_per_page} autres")

    if res.relationships:
        print("\nRelations :")
        for r in res.relationships:
            print(f"  {r.source_id} --[{r.relation_type}]--> {r.target_id} "
                  f"({r.source_label})")


structured = extract_structured_elements(doc, filename)
print_coherence(structured)
```

<!-- #region -->
On sauvegarde le résultat structuré en JSON : c'est l'artefact consommé par l'étape d'ingestion suivante.
<!-- #endregion -->

```python
struct_path = DATA_DIR / "structured_elements.json"
with open(struct_path, "w", encoding="utf-8") as fh:
    json.dump(structured.to_dict(), fh, ensure_ascii=False, indent=2)
print(f"Résultat structuré sauvegardé : {struct_path}")
```

<!-- #region -->
## 15. Passage à l'échelle : fenêtrage
<!-- #endregion -->

<!-- #region -->
Pour un long document (ou un orchestrateur comme Dagster), on convertit **fenêtre par fenêtre** et on **propage l'état header** (`next_header_state`) d'une fenêtre à la suivante, pour que la hiérarchie reste cohérente à la jointure des fenêtres.
<!-- #endregion -->

```python
def iter_page_windows(total_pages: int, window_size: int = 50) -> list[dict]:
    """Génère les paramètres de fenêtres de pages pour un long document.

    Dans un orchestrateur, on convertit fenêtre par fenêtre et on propage
    next_header_state d'une fenêtre à la suivante.
    """
    windows: list[dict] = []
    for start in range(1, total_pages + 1, window_size):
        end = min(start + window_size - 1, total_pages)
        windows.append({"page_range": (start, end), "header_state": None})
    return windows


demo_windows = iter_page_windows(total_pages=120, window_size=50)
print("Fenêtres générées (doc de 120 pages, fenêtre=50) :")
for i, w in enumerate(demo_windows):
    print(f"  fenêtre {i + 1}: pages {w['page_range']}")
```

<!-- #region -->
## 16. Limites connues et axes d'amélioration
<!-- #endregion -->

<!-- #region -->
Le pipeline est volontairement modulaire pour itérer. Limites actuelles et pistes :

- **Hiérarchie (stratégie 1)** : échoue sur les headers **non numérotés**. Piste : combiner numérotation + niveaux Docling + indices typographiques (taille de police, casse) ; fallback sur le dernier header vu.
- **Caption (stratégie 2)** : heuristique purement géométrique, sans vérification sémantique. Piste : valider par regex (`^(Figure|Table)\s+\d+`), gérer le multi-colonnes, pondérer distance et alignement.
- **Ordre de lecture** : `iterate_items()` suit l'ordre Docling ; sur des layouts complexes multi-colonnes, un post-tri par (page, colonne, y) peut aider.
- **Déduplication** : headers/footers répétés ne sont pas dédupliqués — utile pour un index propre.
- **Validation** : valider chaque `StructuredElement` contre un schéma **Pydantic** avant ingestion ; exporter en Parquet ou vers un graphe (Neo4j) plutôt qu'en JSON.
<!-- #endregion -->

<!-- #region -->
## 17. PDF scannés : OCR et VLM
<!-- #endregion -->

<!-- #region -->
Le pipeline ci-dessus suppose un **PDF numérique** (texte sélectionnable). Pour des **scans** (images de pages), le texte n'est pas extractible directement :

- **OCR** : activer `PdfPipelineOptions(do_ocr=True)`. Docling supporte plusieurs moteurs (EasyOCR, RapidOCR, Tesseract). Plus lent, mais indispensable sur du scanné.
- **Pipeline VLM** : pour des documents très complexes (mises en page exotiques, manuscrits), Docling propose un pipeline basé sur des **modèles vision-langage** (`docling[vlm]`), qui « lit » la page comme un modèle multimodal. Plus coûteux, à réserver aux cas où le pipeline standard échoue.
- **Description d'images** : `do_picture_description=True` génère une légende automatique des figures (utile pour un RAG multimodal).

Règle pratique : commencer par le pipeline standard ; n'activer OCR/VLM que si l'extraction est vide ou de mauvaise qualité.
<!-- #endregion -->

<!-- #region -->
## 18. Sources
<!-- #endregion -->

<!-- #region -->
- Documentation Docling — [DocumentConverter](https://docling-project.github.io/docling/reference/document_converter/), [Pipeline options](https://docling-project.github.io/docling/reference/pipeline_options/), [Chunking](https://docling-project.github.io/docling/concepts/chunking/), [Figure export](https://docling-project.github.io/docling/examples/export_figures/).
- Papier : *Docling Technical Report* — [arXiv:2408.09869](https://arxiv.org/abs/2408.09869).
- Dépôts : [docling](https://github.com/docling-project/docling), [docling-core](https://github.com/docling-project/docling-core).
<!-- #endregion -->
