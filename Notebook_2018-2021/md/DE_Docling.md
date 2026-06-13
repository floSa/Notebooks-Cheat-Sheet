---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="r1l2H88hAdII" -->
# Docling Tuto
<!-- #endregion -->

<!-- #region id="eNFqwNB7AaSW" -->
## Étape 1 : Installation des dépendances
<!-- #endregion -->

```python id="FlMmHdZq9B23" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1776019133002, "user_tz": -120, "elapsed": 25921, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b9bb0cca-7ad8-43a5-f730-80163a50792f"
# 📦 Installation de Docling avec support VLM pour l'analyse d'images
!pip install -q docling[vlm] docling-core pillow matplotlib rich PyMuPDF ipywidgets ipywidgets plotly

# 🎨 Bibliothèques pour la visualisation
!pip install -q

# 📦 INSTALLATION - Cellule 1 (CORRIGÉE)
!pip install -q docling[vlm] docling-core pillow matplotlib rich

# ✅ IMPORTS CORRIGÉS
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
# 👇 CORRECTION IMPORT LABELS 👇
from docling_core.types.doc.labels import DocItemLabel, TableCellLabel
from docling.datamodel.base_models import InputFormat

import fitz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import requests, json, io


from rich import print as rprint
from rich.table import Table
from rich.console import Console
from IPython.display import display, HTML, Image as IPyImage, Markdown

print("✅ Installations terminées !")
```

<!-- #region id="aYo6l_Z6AivQ" -->
## Étape 2 : Préparation d'un document de test
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 153, "referenced_widgets": ["7e8a57be251d4032b0977a1a9f2cda85", "1adf838b16424b89b4d663b36448f40b", "6b9880bb0a7143cb81fec1ed06ac1487", "4233b312b0d24af68cdcdd6b1d46f4e2", "f7399d61870b4b8883772d9e8804f6fa", "32116a0c97ac449ea8998f462b5aaade", "f105b15e19984dfd9fcb6a88c3bc83d6", "3b699298cba845849c99d35890d44069", "21c5469ff1dc4cd2a5735e25a2874182", "1ae277c8d44b4787927627798493ea4b", "7bd7a701a5a44e288dfe92a343c17f2c"]} id="3_Ob4OmBAYP5" executionInfo={"status": "ok", "timestamp": 1776019048655, "user_tz": -120, "elapsed": 18571, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8eb9a0bc-8f5b-49cd-a0fc-5252df0fe0e5"
# 🔄 CONVERSION - Cellule 2 (CORRIGÉE & STABLE)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# 🎯 PARAMÈTRE PAGES : Liste des pages à traiter (1-indexé)
# Modifiez ici pour limiter le temps de calcul
PAGES_TO_CONVERT = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# ✅ Configuration table : utiliser l'objet dédié (pas un dict !)
table_opts = TableStructureOptions(mode=TableFormerMode.FAST)

# ✅ Pipeline options
pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=True,
    table_structure_options=table_opts,  # ← OBJET CORRIGÉ
    do_code_enrichment=False,
    do_formula_enrichment=False,
    do_picture_classification=False,
    do_picture_description=False,
)

# ✅ Converter avec limite de pages
converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(
            pipeline_options=pipeline_options,
            pages_to_convert=PAGES_TO_CONVERT  # ← 🎯 ICI
        )
    }
)

test_pdf_url = "https://arxiv.org/pdf/2408.09869.pdf"

print(f"⏳ Conversion pages {PAGES_TO_CONVERT}...")
result = converter.convert(test_pdf_url)
doc = result.document

print(f"✅ Conversion terminée !")
print(f"📊 Pages: {doc.num_pages()}")
print(f"📝 Textes: {len(doc.texts)}")
print(f"📊 Tables: {len(doc.tables)}")
print(f"🖼️ Pictures: {len(doc.pictures)}")
```

<!-- #region id="bheU9bfVAoZw" -->
## Étape 3 : Conversion du document avec Docling
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 486, "referenced_widgets": ["a8d20f6392934aacae5d068baf8150e8", "0485f68c4b3949839f1d7278124e55a4", "e2a807bad4334ab3a6fb7387bfc40987", "8b44bf34d64b48b7ab641b772c5e34b3", "1da8d1d687494004a08b661c99736013", "ff670d5c782848a4af3d04c6bea32c2f", "88cbac0ff6b44d089a7738997d63ca0d", "fe1b61ea74fe40cea6c3d314e56dc3cf", "a77ddb0221f9497492ee0c1e1a660e5e", "653ecc94eb4e426e976155abfe0e86cf", "5b06594ce0854d569b8658a290aa5bab"]} id="Pwsotnx7AqG8" executionInfo={"status": "ok", "timestamp": 1776019071854, "user_tz": -120, "elapsed": 23192, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b1be8671-dade-48f7-b959-86744515d765"
# 🔄 Initialisation du converter
converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF, InputFormat.HTML, InputFormat.DOCX]
)

# 🎯 Conversion du document
print("⏳ Conversion en cours...")
result = converter.convert(test_pdf_url)
doc = result.document

print(f"✅ Document converti !")
print(f"📊 Pages détectées : {doc.num_pages()}")
print(f"📝 Éléments texte : {len(doc.texts)}")
print(f"📊 Tables : {len(doc.tables)}")
print(f"🖼️ Images : {len(doc.pictures)}")
```

<!-- #region id="uIfpX30VArtf" -->
## Étape 4 : Afficher TOUTES les fonctionnalités de layout
<!-- #endregion -->

<!-- #region id="1rxE84IuAwCQ" -->
### 4.1 Reading Order & Structure hiérarchique
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="IyiBNbmUAsnS" executionInfo={"status": "ok", "timestamp": 1776019071888, "user_tz": -120, "elapsed": 29, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f9f94def-a9ee-4b52-bac7-53870fe2267a"
# 📚 Parcours des éléments dans l'ordre de lecture
console = Console()
table = Table(title="🧭 Reading Order - Premier page")
table.add_column("Ordre", style="cyan")
table.add_column("Type", style="green")
table.add_column("Texte (extrait)", style="yellow")
table.add_column("Page", style="magenta")

for idx, (item, level) in enumerate(doc.iterate_items()):
    if idx >= 15:  # Limite pour la lisibilité
        break

    # Extraction du texte selon le type
    if hasattr(item, 'text') and item.text:
        text_preview = item.text[:50].replace('\n', ' ') + "..."
    else:
        text_preview = f"<{type(item).__name__}>"

    # Provenance (page + bbox)
    page_no = item.prov[0].page_no if item.prov else "N/A"

    table.add_row(
        str(idx),
        type(item).__name__,
        text_preview,
        str(page_no)
    )

console.print(table)
```

<!-- #region id="T8cikHekA2cS" -->
## 4.2 Bounding Boxes - Visualisation graphique
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="quSyVWrHAzc4" executionInfo={"status": "ok", "timestamp": 1776023445114, "user_tz": -120, "elapsed": 1120, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="982f2630-3c33-41b0-a8e6-a8803cf56701"
import fitz # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from docling_core.types.doc.labels import DocItemLabel
import requests, io
from PIL import Image

def plot_bboxes_on_rendered_page(doc, page_no=1, pdf_url="https://arxiv.org/pdf/2408.09869.pdf"):
    print(f"📥 Rendu de la page {page_no}...")

    # 1. Récupérer le PDF & le rendre en image
    try:
        res = requests.get(pdf_url, stream=True, timeout=30)
        pdf_fitz = fitz.open(stream=io.BytesIO(res.content), filetype="pdf")
        page_fitz = pdf_fitz[page_no - 1]
        zoom = 2.0  # x2 pour plus de netteté
        pix = page_fitz.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception as e:
        print(f"Erreur lors du chargement du PDF : {e}")
        return

    # 2. Configurer le graphique
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(pil_img, origin='upper')
    ax.set_xlim(0, pil_img.width)
    ax.set_ylim(pil_img.height, 0)
    ax.set_title(f"📦 Bounding Boxes (Vivid) - Page {page_no}")
    ax.grid(alpha=0.1) # Grille plus légère

    # 3. Palette de couleurs VIVES (High Contrast)
    # On remplace les couleurs pastelles par des codes hexa francs
    VIVID_COLORS = {
        DocItemLabel.TEXT: "#FF0000",          # Rouge vif
        DocItemLabel.TITLE: "#0000FF",         # Bleu pur
        DocItemLabel.SECTION_HEADER: "#008000",# Vert foncé
        DocItemLabel.LIST_ITEM: "#FF00FF",     # Magenta
        DocItemLabel.TABLE: "#FFA500",         # Orange
        DocItemLabel.PICTURE: "#800080",       # Violet
        DocItemLabel.CODE: "#00CED1",          # Dark Turquoise
        DocItemLabel.FORMULA: "#FFD700",       # Or
        DocItemLabel.PAGE_HEADER: "#8B4513",   # Saddle Brown
        DocItemLabel.PAGE_FOOTER: "#708090",   # Slate Gray
        DocItemLabel.CAPTION: "#4B0082",       # Indigo
    }

    H_pt = page_fitz.rect.height
    count = 0

    # On combine les listes d'éléments
    items = doc.texts + doc.tables + doc.pictures

    for item in items:
        if count >= 50: # Augmenté un peu pour voir plus
            break
        if not item.prov or item.prov[0].page_no != page_no:
            continue

        bbox = item.prov[0].bbox
        label = getattr(item, 'label', DocItemLabel.TEXT)

        # Récupération de la couleur vive, ou Rouge par défaut si le type n'est pas dans la liste
        color_hex = VIVID_COLORS.get(label, "#FF0000")

        # Conversion Coords PDF -> Pixels Image
        x = bbox.l * zoom
        y = (H_pt - bbox.t) * zoom
        w = (bbox.r - bbox.l) * zoom
        h = (bbox.t - bbox.b) * zoom

        # Dessin du rectangle
        # linewidth=2.5 pour bien voir, alpha=1.0 pour pas de transparence
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2.5,
            edgecolor=color_hex,
            facecolor='none',
            alpha=1.0
        )
        ax.add_patch(rect)

        # Ajout du label
        # bbox props ajoute un petit fond blanc derrière le texte pour le rendre lisible
        ax.text(
            x, y - 5,
            label.value.upper(),
            fontsize=7,
            color=color_hex,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5)
        )
        count += 1

    plt.tight_layout()
    plt.show()

# Exécution
plot_bboxes_on_rendered_page(doc, page_no=5)
```

<!-- #region id="xQz1q6PaA5dJ" -->
### 4.3 Éléments PAGE : Image, Number, Header, Footer
<!-- #endregion -->

```python id="jClfMC_PA6V2"
# 4.3 Éléments PAGE : Number, Header, Footer (CORRIGÉ)
print("📄 === ÉLÉMENTS DE PAGE ===\n")
for pno, page in doc.pages.items():
    print(f"📑 Page {pno}: {page.size.width} × {page.size.height} pts")

# Vérification sécurisée (évite les erreurs d'enum manquants)
get_lbl = lambda x: str(getattr(x, 'label', '')).lower()
get_txt = lambda x: getattr(x, 'text', '').strip()

print("\n🔢 PAGE_NUMBERS :")
for item in doc.texts:
    if 'page_number' in get_lbl(item):
        print(f"   • Page {item.prov[0].page_no}: '{get_txt(item)}'")

print("\n📰 HEADERS :")
for item in doc.texts:
    if 'page_header' in get_lbl(item):
        print(f"   • '{get_txt(item)}'")

print("\n🦶 FOOTERS :")
for item in doc.texts:
    if 'page_footer' in get_lbl(item):
        print(f"   • '{get_txt(item)}'")
```

<!-- #region id="opKF9zZFA8-Q" -->
### 4.4 Éléments TEXT : Header, Paragraph, List item, Code, Formula
<!-- #endregion -->

```python id="BTrYbjMlA9xB"
# 4.4 Éléments TEXT : Header, Paragraph, List item, Code, Formula
print("📝 === ÉLÉMENTS DE TEXTE ===\n")
def show_text(label, title, n=1):
    print(f"🔷 {title} :")
    c = 0
    for item in doc.texts:
        if getattr(item, 'label', None) == label and c < n:
            t = getattr(item, 'text', '').strip()[:80].replace('\n', ' ')
            if t: print(f"   • {t}..."); c += 1
    if c == 0: print("   (aucun)")
    print()
show_text(DocItemLabel.SECTION_HEADER, "SECTION HEADERS")
show_text(DocItemLabel.PARAGRAPH, "PARAGRAPHES")
show_text(DocItemLabel.LIST_ITEM, "LIST ITEMS")
show_text(DocItemLabel.CODE, "CODE BLOCKS")
show_text(DocItemLabel.FORMULA, "FORMULAS")
```

<!-- #region id="_MoDClxHBAvu" -->
### 4.5 Éléments TABLE : Cell, Structure
<!-- #endregion -->

```python id="6eK12GK4BBpm"
# 4.5 Éléments TABLE : Cell, Structure
print("📊 === STRUCTURE DES TABLES ===\n")
for idx, tbl in enumerate(doc.tables):
    print(f"🔹 Table #{idx+1}")
    if tbl.prov:
        b = tbl.prov[0].bbox
        print(f"   📦 BBox: [{b.l:.0f}, {b.t:.0f}, {b.r:.0f}, {b.b:.0f}]")
    if tbl.data:
        d = tbl.data
        print(f"   📏 {d.num_rows}r × {d.num_cols}c | {len(d.table_cells)} cells")
        print("   🧱 Cellules (extrait) :")
        for cell in d.table_cells[:6]:
            r = getattr(cell, 'row_idx', 0)
            c = getattr(cell, 'col_idx', 0)
            txt = getattr(cell, 'text', '')[:30].replace('\n', ' ')
            print(f"      • R{r+1}C{c+1}: '{txt}...'")
        try:
            df = tbl.export_to_dataframe()
            print(f"\n   📤 CSV:\n{df.head(2).to_markdown(index=False)[:150]}")
        except: pass
    cap = tbl.caption_text(doc)
    if cap and cap.strip(): print(f"\n   🏷️ Caption: '{cap[:70]}...'")
    print("─"*40 + "\n")
```

<!-- #region id="f8vD8yO_BD2x" -->
### 4.6 Éléments PICTURE : Image, Class, Description
<!-- #endregion -->

```python id="vWZ7NbgfBF3j"
# 4.6 Éléments PICTURE : Image, Class, Description
print("🖼️ === ÉLÉMENTS IMAGE ===\n")
from IPython.display import Image as IPyImage, display
for idx, pic in enumerate(doc.pictures):
    print(f"🔹 Picture #{idx+1}")
    if pic.prov:
        b = pic.prov[0].bbox
        print(f"   📦 BBox: [{b.l:.0f}, {b.t:.0f}, {b.r:.0f}, {b.b:.0f}]")
    if pic.image:
        print(f"   🎨 Format: {pic.image.mimetype}")
        try:
            pil = pic.get_image(doc)
            if pil: display(IPyImage(pil))
        except: pass
    if hasattr(pic, 'annotations'):
        for ann in pic.annotations:
            if getattr(ann, 'kind', '') == 'classification':
                print(f"   🏷️ Class: {ann.predicted_classes}")
    cap = pic.caption_text(doc)
    if cap and cap.strip(): print(f"   🏷️ Caption: '{cap[:70]}...'")
    print("─"*40 + "\n")
```

<!-- #region id="6FgJJgk6BJSx" -->
### 4.7 Captions (liens avec tables/images)
<!-- #endregion -->

```python id="rmzwCXcuBLD3"
# 4.7 Captions
print("🏷️ === CAPTIONS ASSOCIÉS ===\n")
print("📊 Tables:")
for t in doc.tables:
    cap = t.caption_text(doc)
    if cap and cap.strip(): print(f"   • '{cap[:80]}...'")
print("\n🖼️ Pictures:")
for p in doc.pictures:
    cap = p.caption_text(doc)
    if cap and cap.strip(): print(f"   • '{cap[:80]}...'")
```

<!-- #region id="vkuWmdgfBNfY" -->
### 4.8 Chunks - Découpage pour RAG
<!-- #endregion -->

```python id="ZQrkHJHkBOkb"
# 4.8 Chunks RAG (CORRIGÉ & ROBUSTE)
print("🧩 === CHUNKING POUR RAG ===\n")
from docling.chunking import HybridChunker
import warnings
warnings.filterwarnings("ignore", message=".*longer than the specified maximum.*")

# Le chunker
chunker = HybridChunker(tokenizer="bert-base-uncased", max_tokens=100)
chunks = list(chunker.chunk(dl_doc=doc))

print(f"✅ {len(chunks)} chunks générés\n")

for i, chunk in enumerate(chunks[:7]):
    # ✅ Seul attribut garanti : chunk.text
    text = chunk.text

    # Compter les tokens manuellement si besoin (fallback simple)
    approx_tokens = len(text.split()) // 3  # estimation grossière

    # Pages : extraire depuis les provenances si disponibles
    pages = set()
    if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'prov'):
        for p in chunk.meta.prov:
            if hasattr(p, 'page_no'):
                pages.add(p.page_no)
    pages_str = sorted(pages) if pages else "N/A"

    print(f"🔹 Chunk #{i+1} | ~{approx_tokens} tokens | Pages: {pages_str}")
    print(f"   📝 {text[:150].replace(chr(10), ' ')}...\n")
```

<!-- #region id="Q055aAW8BQ9I" -->
### 4.9 Bounding Boxes - Données brutes
<!-- #endregion -->

```python id="kyVZA3nEBRxG"
print("📦 === DONNÉES BRUTES BOUNDING BOXES ===\n")

# Export complet en dict pour inspection
doc_dict = doc.export_to_dict()

# Extraire les bbox par type
bbox_summary = {}
for item_type in ['texts', 'tables', 'pictures']:
    items = doc_dict.get(item_type, [])
    bbox_summary[item_type] = []

    for item in items:
        if 'prov' in item and item['prov']:
            for prov in item['prov']:
                if 'bbox' in prov:
                    bbox = prov['bbox']
                    bbox_summary[item_type].append({
                        'page': prov.get('page_no'),
                        'bbox': bbox,
                        'label': item.get('label'),
                        'text_preview': item.get('text', '')[:30] if 'text' in item else ''
                    })

# Afficher un résumé
for item_type, bboxes in bbox_summary.items():
    print(f"🔹 {item_type.upper()} : {len(bboxes)} éléments avec bbox")
    if bboxes:
        sample = bboxes[0]
        print(f"   Exemple: Page {sample['page']}, "
              f"BBox[{sample['bbox']['l']:.1f}, {sample['bbox']['t']:.1f}, "
              f"{sample['bbox']['r']:.1f}, {sample['bbox']['b']:.1f}]")
        if sample['text_preview']:
            print(f"   Texte: '{sample['text_preview']}...'")
    print()
```

<!-- #region id="OUdshTfLBUpx" -->
## Étape 5 : Export dans différents formats
<!-- #endregion -->

```python id="Vr4Ra27rBVaQ"
print("📤 === EXPORTS DISPONIBLES ===\n")

# 📄 Markdown
print("🔹 Export Markdown (extrait):")
md_output = doc.export_to_markdown()
print(md_output[:500] + "...\n")

# 🌐 HTML
print("🔹 Export HTML (extrait):")
html_output = doc.export_to_html()
print(html_output[:300] + "...\n")

# 🗂️ JSON structuré
print("🔹 Export JSON (structure):")
json_output = doc.export_to_dict()
print(f"Clés principales: {list(json_output.keys())}\n")

# 💾 Sauvegarder localement dans Colab
import json
with open('docling_output.json', 'w', encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)
print("✅ Fichier 'docling_output.json' sauvegardé !")

# 📥 Télécharger le fichier
from google.colab import files
files.download('docling_output.json')
```

<!-- #region id="YDKEVwpHBX96" -->
### Bonus : Dashboard de visualisation interactive
<!-- #endregion -->

```python id="XZZEa0JGBZ2w"
# 🎛️ Widget interactif pour explorer le document
import ipywidgets as widgets
from IPython.display import display

def show_page_details(page_num):
    """Affiche les détails d'une page spécifique"""
    output = []

    output.append(f"### 📑 Page {page_num}\n")

    # Compter les éléments par type
    type_counts = {}
    for item, _ in doc.iterate_items():
        if item.prov and item.prov[0].page_no == page_num:
            label = str(getattr(item, 'label', 'UNKNOWN'))
            type_counts[label] = type_counts.get(label, 0) + 1

    # Tableau récapitulatif
    output.append("| Type | Count |")
    output.append("|------|-------|")
    for t, c in sorted(type_counts.items()):
        output.append(f"| {t} | {c} |")
    output.append("")

    # Bounding boxes de la page
    output.append("📦 **Bounding Boxes** :")
    for item, _ in doc.iterate_items():
        if item.prov and item.prov[0].page_no == page_num:
            bbox = item.prov[0].bbox
            label = getattr(item, 'label', 'UNKNOWN')
            text = getattr(item, 'text', '')[:40].replace('\n', ' ') if hasattr(item, 'text') else ''
            output.append(f"- `{label}`: [{bbox.l:.0f},{bbox.t:.0f},{bbox.r:.0f},{bbox.b:.0f}] → `{text}...`")

    return "\n".join(output)

# Créer le widget
page_slider = widgets.IntSlider(
    value=1,
    min=1,
    max=doc.num_pages(),
    step=1,
    description='📑 Page:',
    continuous_update=False
)

output_area = widgets.Output()

def on_page_change(change):
    with output_area:
        output_area.clear_output()
        display(Markdown(show_page_details(change['new'])))

page_slider.observe(on_page_change, names='value')

# Afficher le dashboard
print("🎛️ Dashboard interactif - Sélectionnez une page :")
display(page_slider)
display(output_area)
# Déclencher l'affichage initial
display(Markdown(show_page_details(1)))
```

<!-- #region id="mUpcc6I7Fidn" -->
## Étape 6 : Tester avec votre propre PDF/HTML
<!-- #endregion -->

```python id="Tz1vXzVJFle8"
# 📁 Upload de fichier dans Colab
from google.colab import files

print("📤 Uploadez votre fichier PDF, DOCX ou HTML :")
uploaded = files.upload()

if uploaded:
    filename = list(uploaded.keys())[0]
    print(f"✅ Fichier chargé : {filename}")

    # Re-convertir avec votre fichier
    result_custom = converter.convert(filename)
    doc_custom = result_custom.document

    print(f"\n📊 Statistiques de votre document :")
    print(f"• Pages: {doc_custom.num_pages()}")
    print(f"• Textes: {len(doc_custom.texts)}")
    print(f"• Tables: {len(doc_custom.tables)}")
    print(f"• Images: {len(doc_custom.pictures)}")

    # Ré-exécuter les visualisations avec doc_custom si souhaité
    # visualize_bounding_boxes(doc_custom, page_no=1)
```

<!-- #region id="tn19QBBJCJSc" -->
# test information
<!-- #endregion -->

<!-- #region id="UgZbibMuCNIx" -->
## Helpers (IDs, bbox, normalisation)
<!-- #endregion -->

```python id="oZUwsF0ECLCl"
# A. Helpers de base
import hashlib, json, re
from typing import Optional, Dict, Any, List
from docling_core.types.doc.labels import DocItemLabel

def compute_element_id(filename: str, page_no: int, order: int, text_snippet: str) -> str:
    """ID stable et reproductible pour un élément"""
    payload = f"{filename}|{page_no}|{order}|{text_snippet[:50]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def extract_bbox_dict(bbox) -> Dict[str, float]:
    """Normalise les coords PDF en dict JSON-serializable"""
    return {"l": round(bbox.l, 2), "t": round(bbox.t, 2), "r": round(bbox.r, 2), "b": round(bbox.b, 2)}

def clean_filename(filepath_or_url: str) -> str:
    """Extrait le nom sans extension depuis un chemin ou URL"""
    name = filepath_or_url.split("/")[-1].split("\\")[-1]
    return re.sub(r'\.[^.]+$', '', name)  # retire l'extension
```

<!-- #region id="mwDnZB3HCQpy" -->
## HeaderContextTracker (liaison header→contenu)
<!-- #endregion -->

```python id="WMGC6joICO9B"
# B. Tracker de contexte hiérarchique (persistant entre fenêtres)
class HeaderContextTracker:
    """Maintient les headers actifs par niveau pendant le parcours"""
    def __init__(self):
        self.current_headings: Dict[int, Dict] = {}  # {level: header_dict}
        self.last_header_ref: Optional[Dict] = None

    def update_with_header(self, header_item, filename: str, global_order: int):
        """Appelé quand on rencontre un SECTION_HEADER"""
        level = getattr(header_item, 'level', 1)
        hdr = {
            "id": compute_element_id(filename, header_item.prov[0].page_no if header_item.prov else 1,
                                    global_order, getattr(header_item, 'text', '')[:30]),
            "text": getattr(header_item, 'text', '').strip(),
            "level": level,
            "page_no": header_item.prov[0].page_no if header_item.prov else 1
        }
        # Mettre à jour le niveau courant et nettoyer les niveaux plus profonds
        self.current_headings[level] = hdr
        for lvl in list(self.current_headings.keys()):
            if lvl > level:
                del self.current_headings[lvl]
        # Le header le plus spécifique est celui du niveau max présent
        if self.current_headings:
            self.last_header_ref = self.current_headings[max(self.current_headings.keys())]

    def get_active_header_ref(self) -> Optional[Dict]:
        """Retourne le header actif le plus spécifique pour un item contenu"""
        return self.last_header_ref

    def export_state(self) -> Dict:
        """Exporte l'état pour le passer à la fenêtre suivante (Dagster)"""
        return {"current_headings": self.current_headings, "last_header_ref": self.last_header_ref}

    def load_state(self, state: Dict):
        """Restaure l'état depuis la fenêtre précédente"""
        self.current_headings = state.get("current_headings", {})
        self.last_header_ref = state.get("last_header_ref", None)
```

<!-- #region id="3VLTKgLMCXYn" -->
## Fonction principale d'extraction structurée
<!-- #endregion -->

```python id="oWJROg3sCTVy"
# C. Extraction structurée CORRIGÉE v3 (BOTTOMLEFT + Caption linking robuste)
def extract_structured_elements(doc, filename: str,
                                header_state: Optional[Dict] = None,
                                start_global_order: int = 0) -> Dict[str, Any]:
    tracker = HeaderContextTracker()
    if header_state: tracker.load_state(header_state)

    elements = []
    relationships = []
    global_order = start_global_order

    # Pré-indexer les captions par page
    captions_by_page = {}
    used_captions = set()
    for item, _ in doc.iterate_items():
        lbl = str(getattr(item, 'label', '')).lower()
        if 'caption' in lbl and item.prov:
            page = item.prov[0].page_no
            captions_by_page.setdefault(page, []).append(item)

    # Parcours dans l'ordre de lecture
    for item, level in doc.iterate_items():
        lbl_str = str(getattr(item, 'label', 'text')).lower()
        prov = item.prov[0] if item.prov else None
        page_no = prov.page_no if prov else 1
        bbox = prov.bbox if prov else None
        text = getattr(item, 'text', '').strip() if hasattr(item, 'text') else None

        elem_id = compute_element_id(filename, page_no, global_order, text or "")

        elem = {
            "id": elem_id, "filename": filename, "page_no": page_no,
            "order_in_page": global_order, "global_order": global_order,
            "type": type(item).__name__, "label": lbl_str,
            "text": text if text else None,
            "bbox": extract_bbox_dict(bbox) if bbox else None
        }

        # 🔗 Liens hiérarchiques
        if lbl_str == "title":
            elem["document_ref"] = filename
        elif lbl_str == "section_header":
            parent = tracker.get_active_header_ref()
            if parent and parent.get("level", 0) < getattr(item, 'level', 1):
                elem["parent_header_ref"] = parent
            tracker.update_with_header(item, filename, global_order)
        elif lbl_str in ["text", "paragraph", "code", "formula", "list_item"]:
            hdr = tracker.get_active_header_ref()
            if hdr: elem["section_header_ref"] = hdr
        elif lbl_str in ["table", "picture"]:
            hdr = tracker.get_active_header_ref()
            if hdr: elem["section_header_ref"] = hdr

            # 🎯 LIEN CAPTION (Géométrie BOTTOMLEFT corrigée)
            if bbox:
                item_cx = (bbox.l + bbox.r) / 2
                item_b = bbox.b  # bas de l'élément
                item_t = bbox.t  # haut de l'élément

                for cap in captions_by_page.get(page_no, []):
                    if id(cap) in used_captions or not cap.prov: continue
                    cap_bbox = cap.prov[0].bbox
                    cap_cx = (cap_bbox.l + cap_bbox.r) / 2

                    # Alignement horizontal (>50% de recouvrement ou centrage proche)
                    if abs(cap_cx - item_cx) > 150: continue

                    cap_t = cap_bbox.t
                    cap_b = cap_bbox.b
                    dist_below = item_b - cap_t   # caption en dessous (typique image)
                    dist_above = cap_b - item_t   # caption au-dessus (typique tableau)

                    cap_text = getattr(cap, 'text', '').strip()
                    if cap_text and (0 < dist_below < 120 or 0 < dist_above < 120):
                        cap_id = compute_element_id(filename, page_no, global_order, cap_text[:30])
                        elem["caption_ref"] = {"id": cap_id, "text": cap_text[:100]}
                        relationships.append({
                            "source_id": elem_id, "target_id": cap_id,
                            "relation_type": "HAS_CAPTION", "source_label": lbl_str
                        })
                        used_captions.add(id(cap))
                        break

        if lbl_str in ["picture", "table"]:
            elem["minio_key"] = f"{filename}/{lbl_str}s/{elem_id}.png"

        elements.append(elem)
        global_order += 1

    return {
        "metadata": {"filename": filename, "total_pages": doc.num_pages(),
                     "elements_count": len(elements), "relationships_count": len(relationships)},
        "elements": elements, "relationships": relationships,
        "next_header_state": tracker.export_state()
    }
```

<!-- #region id="K_ztlGuyCatC" -->
## Exécution sur le document de test
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dDf-Ub8eCcKb" executionInfo={"status": "ok", "timestamp": 1776021044717, "user_tz": -120, "elapsed": 15, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="5612e1bd-02d3-4709-f35c-701e54fac3de"
# D. Exécution & Affichage complet
print(f"🎯 Extraction structurée pour : {filename}")
result = extract_structured_elements(doc, filename)

print(f"✅ {result['metadata']['elements_count']} éléments | 🔗 {result['metadata']['relationships_count']} relations\n")

# Affichage par page
for pn in sorted(set(e['page_no'] for e in result['elements'])):
    page_elems = [e for e in result['elements'] if e['page_no'] == pn]
    print(f"\n📄 PAGE {pn} ({len(page_elems)} éléments)")
    print(f"{'ID':<10} {'LABEL':<16} {'RÉFÉRENCE':<30} {'ORDRE'}")
    print(f"{'-'*10} {'-'*16} {'-'*30} {'-'*6}")
    for e in page_elems[:15]:  # max 15/page pour lisibilité
        ref = "-"
        if e.get('section_header_ref'): ref = f"§{e['section_header_ref']['text'][:20]}..."
        elif e.get('caption_ref'): ref = f"🏷️{e['caption_ref']['text'][:20]}..."
        elif e.get('document_ref'): ref = f"📄{e['document_ref']}"
        print(f"{e['id'][:8]:<10} {e['label']:<16} {ref:<30} {e['global_order']}")
    if len(page_elems) > 15: print(f"   ... +{len(page_elems)-15} autres")

# Validation des relations
print(f"\n🔗 RELATIONS CRÉÉES ({len(result['relationships'])}):")
for r in result['relationships']:
    print(f"  • {r['source_id'][:8]} --[{r['relation_type']}]--> {r['target_id'][:8]} ({r['source_label']})")
```

<!-- #region id="Rj9SeGS0Cepi" -->
## Gestion des fenêtres de pages (pour longs docs)
<!-- #endregion -->

```python id="xAyzAWfPCfz7"
# E. Exemple de traitement par fenêtres (à adapter dans Dagster)
WINDOW_SIZE = 50  # pages par fenêtre

def process_document_in_windows(pdf_path: str, window_size: int = WINDOW_SIZE):
    """Génère les paramètres pour chaque fenêtre de traitement"""
    # Ici on suppose qu'on connaît le nombre total de pages (à obtenir via PyMuPDF ou premier passage léger)
    # Pour l'exemple, on simule avec une estimation
    total_pages = 100  # À remplacer par détection réelle

    windows = []
    for start in range(1, total_pages + 1, window_size):
        end = min(start + window_size - 1, total_pages)
        windows.append({
            "pages_to_convert": list(range(start, end + 1)),
            "header_state": None,  # Sera rempli par la fenêtre précédente
            "start_global_order": 0  # Sera calculé cumulativement
        })
    return windows

# Pour ton pipeline Dagster :
# 1. Premier appel : header_state=None, start_global_order=0
# 2. Appels suivants : header_state=result_prev["next_header_state"], start_global_order=cumulative_count
# 3. Fusionner tous les résultats JSON en un seul fichier final

print("🪟 Fenêtres générées (exemple) :")
for i, win in enumerate(process_document_in_windows("dummy.pdf")[:3]):
    print(f"  Fenêtre {i+1}: pages {win['pages_to_convert'][0]}-{win['pages_to_convert'][-1]}")
```

<!-- #region id="-R5wIQaaEJ9w" -->

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1xi0xrFCEKOz" executionInfo={"status": "ok", "timestamp": 1776020403433, "user_tz": -120, "elapsed": 10, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="6a460fe6-1e80-433b-a215-8e219d9ff490"
# Debug : Pourquoi 0 relations ?
print("🔍 Inspection des items TABLE/PICTURE/CAPTION\n")

for item, level in doc.iterate_items():
    label = getattr(item, 'label', None)
    label_str = label.value if hasattr(label, 'value') else str(label)

    if label_str in ["TABLE", "PICTURE", "CAPTION"]:
        print(f"📦 {label_str} | Page {item.prov[0].page_no if item.prov else '?'}")
        print(f"   • Type: {type(item).__name__}")
        print(f"   • Text: {getattr(item, 'text', '')[:50] if hasattr(item, 'text') else 'N/A'}")

        # Inspection caption
        if hasattr(item, 'caption'):
            print(f"   • .caption exists: {item.caption}")
            if item.caption:
                print(f"     - Type caption: {type(item.caption)}")
                if hasattr(item.caption, 'text'):
                    print(f"     - Caption text: {item.caption.text[:50]}")
                if hasattr(item.caption, 'target'):
                    print(f"     - Caption target: {item.caption.target}")
        else:
            print(f"   • .caption: NOT FOUND")

        # Inspection annotations (autre source possible de caption)
        if hasattr(item, 'annotations') and item.annotations:
            print(f"   • Annotations: {len(item.annotations)}")
            for ann in item.annotations:
                print(f"     - {ann.kind}: {getattr(ann, 'predicted_classes', getattr(ann, 'text', ''))}")

        print("   " + "-"*40)
```

```python id="ZRQQcnfDELQE"
# C. Extraction structurée CORRIGÉE v3 (BOTTOMLEFT + Caption linking robuste)
def extract_structured_elements(doc, filename: str,
                                header_state: Optional[Dict] = None,
                                start_global_order: int = 0) -> Dict[str, Any]:
    tracker = HeaderContextTracker()
    if header_state: tracker.load_state(header_state)

    elements = []
    relationships = []
    global_order = start_global_order

    # Pré-indexer les captions par page
    captions_by_page = {}
    used_captions = set()
    for item, _ in doc.iterate_items():
        lbl = str(getattr(item, 'label', '')).lower()
        if 'caption' in lbl and item.prov:
            page = item.prov[0].page_no
            captions_by_page.setdefault(page, []).append(item)

    # Parcours dans l'ordre de lecture
    for item, level in doc.iterate_items():
        lbl_str = str(getattr(item, 'label', 'text')).lower()
        prov = item.prov[0] if item.prov else None
        page_no = prov.page_no if prov else 1
        bbox = prov.bbox if prov else None
        text = getattr(item, 'text', '').strip() if hasattr(item, 'text') else None

        elem_id = compute_element_id(filename, page_no, global_order, text or "")

        elem = {
            "id": elem_id, "filename": filename, "page_no": page_no,
            "order_in_page": global_order, "global_order": global_order,
            "type": type(item).__name__, "label": lbl_str,
            "text": text if text else None,
            "bbox": extract_bbox_dict(bbox) if bbox else None
        }

        # 🔗 Liens hiérarchiques
        if lbl_str == "title":
            elem["document_ref"] = filename
        elif lbl_str == "section_header":
            parent = tracker.get_active_header_ref()
            if parent and parent.get("level", 0) < getattr(item, 'level', 1):
                elem["parent_header_ref"] = parent
            tracker.update_with_header(item, filename, global_order)
        elif lbl_str in ["text", "paragraph", "code", "formula", "list_item"]:
            hdr = tracker.get_active_header_ref()
            if hdr: elem["section_header_ref"] = hdr
        elif lbl_str in ["table", "picture"]:
            hdr = tracker.get_active_header_ref()
            if hdr: elem["section_header_ref"] = hdr

            # 🎯 LIEN CAPTION (Géométrie BOTTOMLEFT corrigée)
            if bbox:
                item_cx = (bbox.l + bbox.r) / 2
                item_b = bbox.b  # bas de l'élément
                item_t = bbox.t  # haut de l'élément

                for cap in captions_by_page.get(page_no, []):
                    if id(cap) in used_captions or not cap.prov: continue
                    cap_bbox = cap.prov[0].bbox
                    cap_cx = (cap_bbox.l + cap_bbox.r) / 2

                    # Alignement horizontal (>50% de recouvrement ou centrage proche)
                    if abs(cap_cx - item_cx) > 150: continue

                    cap_t = cap_bbox.t
                    cap_b = cap_bbox.b
                    dist_below = item_b - cap_t   # caption en dessous (typique image)
                    dist_above = cap_b - item_t   # caption au-dessus (typique tableau)

                    cap_text = getattr(cap, 'text', '').strip()
                    if cap_text and (0 < dist_below < 120 or 0 < dist_above < 120):
                        cap_id = compute_element_id(filename, page_no, global_order, cap_text[:30])
                        elem["caption_ref"] = {"id": cap_id, "text": cap_text[:100]}
                        relationships.append({
                            "source_id": elem_id, "target_id": cap_id,
                            "relation_type": "HAS_CAPTION", "source_label": lbl_str
                        })
                        used_captions.add(id(cap))
                        break

        if lbl_str in ["picture", "table"]:
            elem["minio_key"] = f"{filename}/{lbl_str}s/{elem_id}.png"

        elements.append(elem)
        global_order += 1

    return {
        "metadata": {"filename": filename, "total_pages": doc.num_pages(),
                     "elements_count": len(elements), "relationships_count": len(relationships)},
        "elements": elements, "relationships": relationships,
        "next_header_state": tracker.export_state()
    }
```

```python colab={"base_uri": "https://localhost:8080/"} id="Jx8sxFwCEN6b" executionInfo={"status": "ok", "timestamp": 1776020422719, "user_tz": -120, "elapsed": 42, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="727dbac8-8854-40c4-c3f9-67c3e0f65b13"
# Affichage complet des éléments (9 pages, ~12 éléments/page = lisible)
def print_elements_summary(result, max_per_page=15):
    print(f"📋 RÉCAPITULATIF - {result['metadata']['filename']}")
    print(f"   Pages: 1-{result['metadata']['total_pages']} | Éléments: {result['metadata']['elements_count']} | Relations: {result['metadata']['relationships_count']}\n")

    # Grouper par page
    by_page = {}
    for elem in result['elements']:
        pn = elem['page_no']
        if pn not in by_page:
            by_page[pn] = []
        by_page[pn].append(elem)

    for page_no in sorted(by_page.keys()):
        print(f"\n📄 PAGE {page_no} ({len(by_page[page_no])} éléments)")
        print(f"   {'ID':<12} {'LABEL':<18} {'REF':<25} {'ORDRE'}")
        print(f"   {'-'*12} {'-'*18} {'-'*25} {'-'*10}")

        for i, elem in enumerate(by_page[page_no][:max_per_page]):
            eid = elem['id'][:8]
            lbl = elem['label']

            # Référence la plus pertinente
            ref = "-"
            if 'section_header_ref' in elem:
                ref = f"§{elem['section_header_ref']['text'][:15]}..."
            elif 'caption_ref' in elem:
                ref = f"🖼️{elem['caption_ref']['text'][:15]}..."
            elif 'document_ref' in elem:
                ref = f"📄{elem['document_ref']}"
            elif 'parent_header_ref' in elem:
                ref = f"↑{elem['parent_header_ref']['text'][:15]}..."

            # Ordre relatif à la référence
            order_info = f"{elem['global_order']}"
            if 'section_header_ref' in elem:
                order_info = f"{elem['global_order']} (après §)"

            print(f"   {eid:<12} {lbl:<18} {ref:<25} {order_info}")

        if len(by_page[page_no]) > max_per_page:
            print(f"   ... et {len(by_page[page_no]) - max_per_page} autres")

# Exécution
result = extract_structured_elements(doc, filename)
print_elements_summary(result)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EGUvUlnFEP8v" executionInfo={"status": "ok", "timestamp": 1776020453137, "user_tz": -120, "elapsed": 18, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="0b08e338-f77d-4557-e72c-9936f1befd44"
# Vérification des relations
print("🔗 RELATIONS CRÉÉES:\n")
if result['relationships']:
    for rel in result['relationships']:
        print(f"• {rel['source_id'][:8]} --[{rel['relation_type']}]--> {rel['target_id'][:8]}")
        print(f"  Source: {rel.get('source_label', 'N/A')} | Target: {rel.get('target_label', 'N/A')}")
else:
    print("⚠️ Aucune relation créée. Causes possibles:")
    print("   1. Le PDF n'a pas de captions détectées")
    print("   2. Les captions sont trop éloignés géométriquement des tables/images")
    print("   3. Les items TABLE/PICTURE n'ont pas de bbox valide")

    # Debug rapide
    print("\n🔍 Debug rapide:")
    caps = [e for e in result['elements'] if e['label'] == 'CAPTION']
    pics = [e for e in result['elements'] if e['label'] == 'PICTURE']
    tabs = [e for e in result['elements'] if e['label'] == 'TABLE']
    print(f"   CAPTION items: {len(caps)}")
    print(f"   PICTURE items: {len(pics)}")
    print(f"   TABLE items: {len(tabs)}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="pdeEraLkEXYg" executionInfo={"status": "ok", "timestamp": 1776020645499, "user_tz": -120, "elapsed": 54, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2faad76b-558e-4d26-9cb4-bbfe4050b3cf"
# Debug : Inspection DIRECTE des attributs du document
print("🔍 STRUCTURE RÉELLE DU DOCUMENT\n")

print(f"📦 doc.texts : {len(doc.texts)} éléments")
print(f"📊 doc.tables : {len(doc.tables)} éléments")
print(f"🖼️ doc.pictures : {len(doc.pictures)} éléments")

# 🔎 Inspection des labels réels dans doc.texts
print("\n📋 Labels uniques dans doc.texts :")
labels_in_texts = set()
for item in doc.texts:
    lbl = getattr(item, 'label', None)
    lbl_str = str(lbl).split('.')[-1] if lbl else 'None'
    labels_in_texts.add(lbl_str)
for lbl in sorted(labels_in_texts):
    print(f"   • {lbl}")

# 🔎 Inspection DES TABLES
print(f"\n📊 TABLES ({len(doc.tables)}):")
for i, tbl in enumerate(doc.tables):
    print(f"   Table #{i+1}:")
    print(f"     • Type: {type(tbl).__name__}")
    print(f"     • Label: {getattr(tbl, 'label', 'N/A')}")
    print(f"     • Prov: {tbl.prov[0].bbox if tbl.prov else 'None'}")
    print(f"     • Caption attr: {hasattr(tbl, 'caption')}")
    if hasattr(tbl, 'caption'):
        print(f"     • Caption value: {tbl.caption}")
        if tbl.caption and hasattr(tbl.caption, 'text'):
            print(f"     • Caption text: {tbl.caption.text[:50]}")

# 🔎 Inspection DES PICTURES
print(f"\n🖼️ PICTURES ({len(doc.pictures)}):")
for i, pic in enumerate(doc.pictures):
    print(f"   Picture #{i+1}:")
    print(f"     • Type: {type(pic).__name__}")
    print(f"     • Label: {getattr(pic, 'label', 'N/A')}")
    print(f"     • Prov: {pic.prov[0].bbox if pic.prov else 'None'}")
    print(f"     • Caption attr: {hasattr(pic, 'caption')}")
    if hasattr(pic, 'caption'):
        print(f"     • Caption value: {pic.caption}")
        if pic.caption and hasattr(pic.caption, 'text'):
            print(f"     • Caption text: {pic.caption.text[:50]}")

# 🔎 Vérifier si iterate_items() inclut tables/pictures
print(f"\n🔄 Ce que yield doc.iterate_items() :")
types_seen = set()
for item, level in doc.iterate_items():
    types_seen.add(type(item).__name__)
print(f"   Types vus: {sorted(types_seen)}")
if 'TableItem' not in types_seen or 'PictureItem' not in types_seen:
    print("   ⚠️ iterate_items() n'inclut PAS tables/pictures ! Il faut les itérer séparément.")
```

```python id="4GsX8YA3FGVi"
# C. Extraction structurée CORRIGÉE v2 (parcours complet)
def extract_structured_elements(doc, filename: str,
                                header_state: Optional[Dict] = None,
                                start_global_order: int = 0) -> Dict[str, Any]:
    tracker = HeaderContextTracker()
    if header_state:
        tracker.load_state(header_state)

    elements = []
    relationships = []
    global_order = start_global_order

    # 🎯 PARCOURS COMBINÉ : texts + tables + pictures, triés par (page, bbox top)
    all_items = []

    # Ajouter les textes
    for item in doc.texts:
        all_items.append((item, getattr(item, 'prov', [None])[0]))

    # Ajouter les tables
    for item in doc.tables:
        all_items.append((item, getattr(item, 'prov', [None])[0]))

    # Ajouter les pictures
    for item in doc.pictures:
        all_items.append((item, getattr(item, 'prov', [None])[0]))

    # Trier par page puis par position verticale (top) pour l'ordre de lecture
    def sort_key(x):
        item, prov = x
        if not prov:
            return (9999, 9999)
        return (prov.page_no, prov.bbox.t)

    all_items.sort(key=sort_key)

    # Index des captions pour appariement rapide
    caption_items = [item for item, _ in all_items
                     if str(getattr(item, 'label', '')).endswith('CAPTION')]

    for item, prov in all_items:
        label = getattr(item, 'label', None)
        label_str = str(label).split('.')[-1] if label else 'TEXT'
        page_no = prov.page_no if prov else 1
        text = getattr(item, 'text', '').strip() if hasattr(item, 'text') else None
        bbox = extract_bbox_dict(prov.bbox) if prov and hasattr(prov, 'bbox') else None

        elem_id = compute_element_id(filename, page_no, global_order, text or "")

        element = {
            "id": elem_id,
            "filename": filename,
            "page_no": page_no,
            "order_in_page": global_order,
            "global_order": global_order,
            "type": type(item).__name__,
            "label": label_str,
            "text": text if text else None,
            "bbox": bbox
        }

        # Gestion références
        if label_str == "TITLE":
            element["document_ref"] = filename

        elif label_str == "SECTION_HEADER":
            parent_hdr = tracker.get_active_header_ref()
            if parent_hdr and parent_hdr.get("level", 0) < getattr(item, 'level', 1):
                element["parent_header_ref"] = parent_hdr
            tracker.update_with_header(item, filename, global_order)

        elif label_str in ["PARAGRAPH", "CODE", "FORMULA", "LIST_ITEM"]:
            hdr_ref = tracker.get_active_header_ref()
            if hdr_ref:
                element["section_header_ref"] = hdr_ref

        elif label_str in ["TABLE", "PICTURE"]:
            hdr_ref = tracker.get_active_header_ref()
            if hdr_ref:
                element["section_header_ref"] = hdr_ref

            # 🔗 Recherche caption associée (proximité géométrique)
            for cap_item in caption_items:
                cap_prov = getattr(cap_item, 'prov', [None])[0]
                if not cap_prov or cap_prov.page_no != page_no:
                    continue
                cap_bbox = cap_prov.bbox
                item_bbox = prov.bbox
                # Caption généralement dans les 150 pts verticaux
                if abs(cap_bbox.l - item_bbox.l) < 100 and 0 < (cap_bbox.t - item_bbox.b) < 150:
                    cap_text = getattr(cap_item, 'text', '').strip()
                    if cap_text:
                        cap_id = compute_element_id(filename, page_no, global_order+1, cap_text[:30])
                        element["caption_ref"] = {"id": cap_id, "text": cap_text[:100]}
                        relationships.append({
                            "source_id": elem_id,
                            "target_id": cap_id,
                            "relation_type": "HAS_CAPTION",
                            "source_label": label_str
                        })
                    break

        elif label_str == "CAPTION":
            # Lier au TABLE/PICTURE le plus proche en dessous
            for tgt_item, tgt_prov in all_items:
                tgt_label = str(getattr(tgt_item, 'label', '')).split('.')[-1]
                if tgt_label not in ["TABLE", "PICTURE"]:
                    continue
                if not tgt_prov or tgt_prov.page_no != page_no:
                    continue
                tgt_bbox = tgt_prov.bbox
                cap_bbox = prov.bbox
                # Caption au-dessus ou juste en-dessous de l'élément cible
                if abs(cap_bbox.l - tgt_bbox.l) < 100 and abs(cap_bbox.t - tgt_bbox.t) < 150:
                    tgt_id = compute_element_id(filename, page_no, global_order-1, getattr(tgt_item, 'text', '')[:30])
                    element["target_ref"] = {"id": tgt_id, "type": type(tgt_item).__name__}
                    relationships.append({
                        "source_id": elem_id,
                        "target_id": tgt_id,
                        "relation_type": "CAPTION_OF",
                        "target_label": tgt_label
                    })
                    break

        # MinIO key
        if label_str in ["PICTURE", "TABLE"]:
            element["minio_key"] = f"{filename}/{label_str.lower()}s/{elem_id}.png"

        elements.append(element)
        global_order += 1

    return {
        "metadata": {
            "filename": filename,
            "total_pages": doc.num_pages(),
            "elements_count": len(elements),
            "relationships_count": len(relationships)
        },
        "elements": elements,
        "relationships": relationships,
        "next_header_state": tracker.export_state()
    }
```

```python colab={"base_uri": "https://localhost:8080/"} id="H3C4_DdrFQ3G" executionInfo={"status": "ok", "timestamp": 1776020701464, "user_tz": -120, "elapsed": 16, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="7beef505-7947-4eca-ed68-1ac14c3bdb0e"
# Relancer l'extraction avec la version corrigée
print(f"🎯 Extraction structurée pour : {filename}")
result = extract_structured_elements(doc, filename)

print(f"✅ {result['metadata']['elements_count']} éléments")
print(f"🔗 {result['metadata']['relationships_count']} relations")

# Compter par label
from collections import Counter
label_counts = Counter(e['label'] for e in result['elements'])
print(f"\n📊 Distribution labels:")
for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"   {lbl}: {cnt}")

# Afficher les relations si présentes
if result['relationships']:
    print(f"\n🔗 Relations:")
    for rel in result['relationships']:
        print(f"   {rel['source_id'][:8]} --[{rel['relation_type']}]--> {rel['target_id'][:8]}")
```

```python colab={"base_uri": "https://localhost:8080/"} id="5R_6EtlwFUAP" executionInfo={"status": "ok", "timestamp": 1776021811886, "user_tz": -120, "elapsed": 33, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="637f8053-080b-4023-9811-63fed8c637cd"
# 🔍 VÉRIFICATION COMPLÈTE (TOUTES LES PAGES, AUCUNE TRONCATURE)
def print_cohérence_check(result):
    print(f"📋 VÉRIFICATION COHÉRENCE - {result['metadata']['filename']}")
    print(f"   Éléments: {len(result['elements'])} | Relations: {len(result['relationships'])}\n")

    by_page = {}
    for e in result['elements']:
        by_page.setdefault(e['page_no'], []).append(e)

    # Compteur GLOBAL par référence (continue de page en page)
    global_ref_order = {}

    for pn in sorted(by_page.keys()):
        elems = by_page[pn]
        print(f"\n📄 PAGE {pn} ({len(elems)} éléments)")
        print(f"{'ID':<12} {'LABEL':<16} {'RÉFÉRENCE':<10} {'ORDRE'}")
        print(f"{'-'*12} {'-'*16} {'-'*10} {'-'*6}")

        for e in elems:  # ← PARCOURS COMPLET, AUCUNE LIMITE
            ref_id = '-'
            if e.get('document_ref'):
                ref_id = 'DOC'
            elif e.get('section_header_ref') and e['section_header_ref'].get('id'):
                ref_id = e['section_header_ref']['id'][:8]
            elif e.get('caption_ref') and e['caption_ref'].get('id'):
                ref_id = e['caption_ref']['id'][:8]

            if ref_id == '-':
                order_val = '-'
            else:
                global_ref_order[ref_id] = global_ref_order.get(ref_id, 0) + 1
                order_val = global_ref_order[ref_id]

            print(f"{e['id'][:10]:<12} {e['label']:<16} {ref_id:<10} {order_val}")

# Exécution directe
print_cohérence_check(result)
```

<!-- #region id="oUaVl-txLCN7" -->
# Nouveaux test
<!-- #endregion -->

```python id="c0PC6vY5IZGd"
# Extraction structurée v4 (Hiérarchie headers corrigée + Niveaux visibles)
def extract_structured_elements(doc, filename: str, header_state: dict | None = None):
    tracker = {}  # {level: header_dict}
    if header_state: tracker.update(header_state)

    elements = []
    captions_by_page = {}

    # Indexer les captions
    for item, _ in doc.iterate_items():
        if str(getattr(item, 'label', '')).lower() == 'caption' and item.prov:
            captions_by_page.setdefault(item.prov[0].page_no, []).append(item)

    used_captions = set()

    for item, _ in doc.iterate_items():
        lbl_str = str(getattr(item, 'label', 'text')).lower()
        prov = item.prov[0] if item.prov else None
        page_no = prov.page_no if prov else 1
        bbox = prov.bbox if prov else None
        text = getattr(item, 'text', '').strip() if hasattr(item, 'text') else None

        elem = {
            "id": compute_element_id(filename, page_no, len(elements), text or ""),
            "filename": filename, "page_no": page_no,
            "type": type(item).__name__, "label": lbl_str,
            "text": text if text else None,
            "level": getattr(item, 'level', None), # ← Niveau détecté par Docling
            "bbox": extract_bbox_dict(bbox) if bbox else None
        }

        if lbl_str == "title":
            elem["document_ref"] = "DOC"

        elif lbl_str == "section_header":
            level = elem["level"] or 1
            # Recherche du parent strict (niveau inférieur)
            parent = None
            for l in sorted(tracker.keys(), reverse=True):
                if l < level:
                    parent = tracker[l]
                    break

            # Fallback si niveaux mal détectés : lier au header précédent
            if parent is None and level == 1 and tracker:
                parent = tracker.get(max(tracker.keys()))

            elem["section_header_ref"] = parent["id"] if parent else "DOC"
            tracker[level] = elem
            # Nettoyer les niveaux supérieurs
            for l in list(tracker.keys()):
                if l > level: del tracker[l]

        elif lbl_str in ["text", "paragraph", "code", "formula", "list_item"]:
            active = tracker.get(max(tracker.keys()), None) if tracker else None
            if active: elem["section_header_ref"] = active["id"]

        elif lbl_str in ["table", "picture"]:
            active = tracker.get(max(tracker.keys()), None) if tracker else None
            if active: elem["section_header_ref"] = active["id"]

            if bbox and page_no in captions_by_page:
                cx = (bbox.l + bbox.r) / 2
                for cap in captions_by_page[page_no]:
                    if id(cap) in used_captions or not cap.prov: continue
                    cb = cap.prov[0].bbox
                    cap_text = getattr(cap, 'text', '').strip()
                    if abs((cb.l+cb.r)/2 - cx) < 150 and cap_text:
                        elem["caption_ref"] = compute_element_id(filename, page_no, len(elements)+1, cap_text[:30])
                        used_captions.add(id(cap))
                        break

        elements.append(elem)

    return {"elements": elements, "metadata": {"filename": filename, "total_pages": doc.num_pages()}, "tracker_state": tracker.copy()}
```

```python colab={"base_uri": "https://localhost:8080/"} id="md6OUu8qLGDr" executionInfo={"status": "ok", "timestamp": 1776022480692, "user_tz": -120, "elapsed": 18, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="13f9ed75-0d3a-4054-91d1-452d314fc043"
# Vérification COMPLÈTE (TOUT afficher + Niveau + Compteur continu)
def print_cohérence_check(result):
    elements = result["elements"]
    print(f"📋 VÉRIFICATION - {result['metadata']['filename']} | Éléments: {len(elements)}\n")

    by_page = {}
    for e in elements: by_page.setdefault(e["page_no"], []).append(e)

    ref_counter = {}

    for pn in sorted(by_page.keys()):
        print(f"\n📄 PAGE {pn} ({len(by_page[pn])} éléments)")
        print(f"{'ID':<12} {'LVL':<4} {'LABEL':<16} {'RÉFÉRENCE':<12} {'ORDRE'}")
        print(f"{'-'*12} {'-'*4} {'-'*16} {'-'*12} {'-'*6}")

        for e in by_page[pn]:  # ← PARCOURS COMPLET
            ref_id = e.get("section_header_ref") or e.get("document_ref") or "-"

            if ref_id == "DOC" or ref_id == "-" or ref_id is None:
                order_val = "-"
            else:
                ref_counter[ref_id] = ref_counter.get(ref_id, 0) + 1
                order_val = ref_counter[ref_id]

            lvl = e.get("level") or "-"
            ref_display = str(ref_id)[:10] + "..." if isinstance(ref_id, str) and len(ref_id) > 10 else str(ref_id)
            print(f"{e['id'][:10]:<12} {lvl:<4} {e['label']:<16} {ref_display:<12} {order_val}")

# Exécution
result = extract_structured_elements(doc, filename)
print_cohérence_check(result)
```

```python colab={"base_uri": "https://localhost:8080/"} id="2JHfUOhALHsG" executionInfo={"status": "ok", "timestamp": 1776022483907, "user_tz": -120, "elapsed": 13, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8823d2ed-ebc7-46b3-9dbd-e5cc067d93dd"
# Vérification COMPLÈTE (TOUT afficher + compteur par référence)
def print_cohérence_check(result):
    elements = result["elements"]
    print(f"📋 VÉRIFICATION - {result['metadata']['filename']} | Éléments: {len(elements)}\n")

    by_page = {}
    for e in elements: by_page.setdefault(e["page_no"], []).append(e)

    # Compteur GLOBAL par référence
    ref_counter = {}

    for pn in sorted(by_page.keys()):
        print(f"\n📄 PAGE {pn} ({len(by_page[pn])} éléments)")
        print(f"{'ID':<12} {'LABEL':<16} {'RÉFÉRENCE':<12} {'ORDRE'}")
        print(f"{'-'*12} {'-'*16} {'-'*12} {'-'*6}")

        for e in by_page[pn]:  # ← PARCOURS COMPLET, AUCUNE LIMITE
            ref_id = e.get("section_header_ref") or e.get("document_ref") or "-"

            if ref_id == "DOC":
                order_val = "-"
            elif ref_id == "-" or ref_id is None:
                order_val = "-"
            else:
                ref_counter[ref_id] = ref_counter.get(ref_id, 0) + 1
                order_val = ref_counter[ref_id]

            ref_display = ref_id[:10] + "..." if isinstance(ref_id, str) and len(ref_id) > 10 else str(ref_id)
            print(f"{e['id'][:10]:<12} {e['label']:<16} {ref_display:<12} {order_val}")

# Exécution
result = extract_structured_elements(doc, filename)
print_cohérence_check(result)
```

<!-- #region id="29PvuyQtUj9W" -->
# NEEEEEEEW
<!-- #endregion -->

```python id="TiaHUohGUjT8"
# Helpers + Extraction v7 (Fix acronymes + Hiérarchie stricte)
import re, hashlib
from typing import Dict, Optional, Tuple

def compute_element_id(filename: str, page_no: int, order: int, text_snippet: str) -> str:
    return hashlib.sha256(f"{filename}|{page_no}|{order}|{text_snippet[:50]}".encode()).hexdigest()[:10]

def extract_bbox_dict(bbox) -> Dict[str, float]:
    return {"l": round(bbox.l, 2), "t": round(bbox.t, 2), "r": round(bbox.r, 2), "b": round(bbox.b, 2)}

def parse_header_numbering(text: str) -> Tuple[Optional[tuple], str]:
    """Parse '3.2 AI models' -> ((3,2), 'AI models') | 'OCR' -> (None, 'OCR')"""
    clean = text.strip()
    match = re.match(r'^([A-Z]{1,2}|[IVXLCDM]{1,4}|[0-9]+(?:\.[0-9]+)*)\s*[.\)]?\s*', clean)
    if not match: return None, text

    num_str = match.group(1)
    remainder = text[match.end():].strip()
    parts = []
    for p in num_str.split('.'):
        if p.isdigit():
            parts.append(int(p))
        elif re.match(r'^[IVXLCDM]+$', p):
            vals = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
            total, prev = 0, 0
            for c in reversed(p):
                v = vals[c]
                total += v if v >= prev else -v
                prev = v
            parts.append(total)
        elif p.isalpha():
            # ✅ FIX: Ignorer les acronymes >1 lettre (OCR, APP, INT...)
            if len(p) > 1: return None, text
            parts.append(ord(p.upper()) - ord('A') + 1)
        else:
            return None, text
    return tuple(parts), remainder

def extract_structured_elements(doc, filename: str, header_state: dict | None = None):
    stack = header_state.get("stack", []) if header_state else []
    elements = []
    captions_by_page = {}

    for item, _ in doc.iterate_items():
        if str(getattr(item, 'label', '')).lower() == 'caption' and item.prov:
            captions_by_page.setdefault(item.prov[0].page_no, []).append(item)

    used_captions = set()
    for item, _ in doc.iterate_items():
        lbl_str = str(getattr(item, 'label', 'text')).lower()
        prov = item.prov[0] if item.prov else None
        page_no = prov.page_no if prov else 1
        bbox = prov.bbox if prov else None
        text = getattr(item, 'text', '').strip() if hasattr(item, 'text') else None

        elem = {
            "id": compute_element_id(filename, page_no, len(elements), text or ""),
            "filename": filename, "page_no": page_no,
            "type": type(item).__name__, "label": lbl_str,
            "text": text if text else None, "bbox": extract_bbox_dict(bbox) if bbox else None
        }

        if lbl_str == "title":
            elem["document_ref"] = "DOC"

        elif lbl_str == "section_header":
            numbering_tuple, _ = parse_header_numbering(text)
            parent = None
            if numbering_tuple:
                for h in reversed(stack):
                    if numbering_tuple[:len(h["tuple"])] == h["tuple"]:
                        parent = h
                        break
            elem["section_header_ref"] = parent["id"] if parent else "DOC"

            # ✅ Seuls les headers numérotés entrent dans la pile
            if numbering_tuple:
                stack = [h for h in stack if numbering_tuple[:len(h["tuple"])] == h["tuple"]]
                stack.append({"id": elem["id"], "tuple": numbering_tuple})

        elif lbl_str in ["text", "paragraph", "code", "formula", "list_item"]:
            active = stack[-1] if stack else None
            if active: elem["section_header_ref"] = active["id"]

        elif lbl_str in ["table", "picture"]:
            active = stack[-1] if stack else None
            if active: elem["section_header_ref"] = active["id"]
            if bbox and page_no in captions_by_page:
                cx = (bbox.l + bbox.r) / 2
                for cap in captions_by_page[page_no]:
                    if id(cap) in used_captions or not cap.prov: continue
                    cb = cap.prov[0].bbox
                    cap_text = getattr(cap, 'text', '').strip()
                    if abs((cb.l+cb.r)/2 - cx) < 150 and cap_text:
                        elem["caption_ref"] = compute_element_id(filename, page_no, len(elements)+1, cap_text[:30])
                        used_captions.add(id(cap))
                        break

        if lbl_str in ["picture", "table"]:
            elem["minio_key"] = f"{filename}/{lbl_str}s/{elem['id']}.png"
        elements.append(elem)

    return {"elements": elements, "metadata": {"filename": filename, "total_pages": doc.num_pages()}, "stack": stack}
```

```python colab={"base_uri": "https://localhost:8080/"} id="T_IM1ELzUmdU" executionInfo={"status": "ok", "timestamp": 1776026141683, "user_tz": -120, "elapsed": 54, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e3298a0c-270b-4093-af84-1d8f8caa17a7"
# Vérification COMPLÈTE (Format exact demandé + Compteur par référence)
def print_cohérence_check(result):
    elems = result["elements"]
    print(f"📋 VÉRIFICATION - {result['metadata']['filename']} | Éléments: {len(elems)}\n")

    by_page = {}
    for e in elems: by_page.setdefault(e["page_no"], []).append(e)
    ref_counter = {}

    for pn in sorted(by_page.keys()):
        print(f"\n📄 PAGE {pn} ({len(by_page[pn])} éléments)")
        print(f"{'ID':<12} {'LABEL':<16} {'RÉFÉRENCE':<12} {'ORDRE'}")
        print(f"{'-'*12} {'-'*16} {'-'*12} {'-'*6}")

        for e in by_page[pn]:  # ← PARCOURS COMPLET, AUCUNE TRONCATURE
            ref_id = e.get("section_header_ref") or e.get("document_ref") or "-"

            if ref_id in ["DOC", "-", None]:
                order_val = "-"
            else:
                ref_counter[ref_id] = ref_counter.get(ref_id, 0) + 1
                order_val = ref_counter[ref_id]

            ref_disp = str(ref_id)[:10] + "..." if isinstance(ref_id, str) and len(ref_id) > 10 else str(ref_id)
            print(f"{e['id']:<12} {e['label']:<16} {ref_disp:<12} {order_val}")

# Exécution
result = extract_structured_elements(doc, filename)
print_cohérence_check(result)
```

```python id="mQI2VQtSUnyl"

```
