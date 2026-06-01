"""Audit format des notebooks — étape 8 du workflow contrat.

Usage :
    # Pre-conversion (sur .md) :
    python scripts/check_format.py --pre 2_NOUVEAUX/md/NOM.md

    # Post-conversion (sur .ipynb) :
    python scripts/check_format.py --post 2_NOUVEAUX/ipynb/NOM.ipynb

    # Les deux (vérification croisée pre→post) :
    python scripts/check_format.py --both 2_NOUVEAUX/md/NOM.md 2_NOUVEAUX/ipynb/NOM.ipynb

Codes de sortie :
    0  : tous les checks verts
    1  : au moins un check rouge (détails en stdout)

Détecte :
    - ```python blocs à l'intérieur de <!-- #region --> ... <!-- #endregion --> → cellule code perdue en text
    - ````python (4 backticks) → bloc qui ne sera PAS une cellule code
    - Cellules markdown du .ipynb qui contiennent ```python (code leaké)
    - Cellules code avec syntax error (ast.parse échoue)
    - Mismatch nombre cellules code attendu vs réel
    - Cellules markdown avec titre + contenu (viole "titre seul")
    - Cellules code uniquement composées de commentaires
    - Cellules code en triple-string (pseudo-code masqué)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

# Force UTF-8 stdout sur Windows pour les caractères accentués
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ─── Pre-conversion : audit du .md ─────────────────────────────────────────────


def parse_md_structure(md_path: Path) -> dict:
    """Parse le .md et compte ce qui DEVRAIT devenir cellule code vs cellule md.

    Renvoie :
        {
            'expected_code_cells': int,
            'code_in_region': [(line_no, snippet), ...],   # ```python à l'intérieur d'une région → perdu
            'four_backtick_blocks': [(line_no, snippet), ...],  # ````python → reste en text
            'top_level_code_blocks': [(line_no_start, line_no_end), ...],
        }
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    in_region = False
    in_code_3 = False        # ```python ou ```... ouvert
    in_code_4 = False        # ````python ou ````... ouvert
    fence_3_lang = None      # langage de la fence courante (python, bash, '', ...)

    expected_code_cells = 0
    code_in_region: list[tuple[int, str]] = []
    four_backtick_blocks: list[tuple[int, str]] = []
    top_level_code_blocks: list[tuple[int, int]] = []
    current_block_start: int | None = None

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Détection des marqueurs de région
        if stripped == "<!-- #region -->":
            in_region = True
            continue
        if stripped == "<!-- #endregion -->":
            in_region = False
            continue

        # 4-backtick fences (priorité car commence par 4, pas 3)
        if stripped.startswith("````"):
            if not in_code_4:
                in_code_4 = True
                four_backtick_blocks.append((i, line[:80]))
            else:
                in_code_4 = False
            continue

        # 3-backtick fences
        if stripped.startswith("```") and not in_code_4:
            if not in_code_3:
                # Ouverture d'une fence
                in_code_3 = True
                fence_3_lang = stripped[3:].strip().split()[0] if len(stripped) > 3 else ""
                if fence_3_lang == "python":
                    if in_region:
                        code_in_region.append((i, line[:80]))
                    else:
                        expected_code_cells += 1
                        current_block_start = i
            else:
                # Fermeture
                in_code_3 = False
                if fence_3_lang == "python" and not in_region and current_block_start is not None:
                    top_level_code_blocks.append((current_block_start, i))
                    current_block_start = None
                fence_3_lang = None
            continue

    return {
        "expected_code_cells": expected_code_cells,
        "code_in_region": code_in_region,
        "four_backtick_blocks": four_backtick_blocks,
        "top_level_code_blocks": top_level_code_blocks,
    }


def check_pre(md_path: Path) -> tuple[bool, dict]:
    """Pré-check sur le .md. Renvoie (ok, info)."""
    info = parse_md_structure(md_path)
    ok = True
    print(f"\n=== PRE-CHECK : {md_path.name} ===")
    print(f"  Cellules code attendues (```python top-level) : {info['expected_code_cells']}")

    if info["code_in_region"]:
        ok = False
        print(f"  ❌ {len(info['code_in_region'])} bloc(s) ```python À L'INTÉRIEUR de <!-- #region --> :")
        for line_no, snippet in info["code_in_region"]:
            print(f"     ligne {line_no}: {snippet}")
        print("     → ces blocs deviendront du texte markdown, pas des cellules code.")

    if info["four_backtick_blocks"]:
        ok = False
        print(f"  ❌ {len(info['four_backtick_blocks'])} bloc(s) ````python (4 backticks) :")
        for line_no, snippet in info["four_backtick_blocks"]:
            print(f"     ligne {line_no}: {snippet}")
        print("     → ces blocs resteront en text. Utiliser 3 backticks + sortir des régions.")

    if ok:
        print("  ✅ Pré-check OK")
    return ok, info


# ─── Post-conversion : audit du .ipynb ──────────────────────────────────────────


def check_post(ipynb_path: Path, expected_code_cells: int | None = None) -> bool:
    """Post-check sur le .ipynb. Si expected_code_cells fourni, vérifie le mismatch."""
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]

    ok = True
    print(f"\n=== POST-CHECK : {ipynb_path.name} ===")
    print(f"  Total cellules : {len(cells)} (md={len(md_cells)}, code={len(code_cells)})")

    # 1. Mismatch nombre cellules code
    if expected_code_cells is not None:
        if len(code_cells) != expected_code_cells:
            ok = False
            print(f"  ❌ Mismatch cellules code : attendues={expected_code_cells}, réelles={len(code_cells)}")
        else:
            print(f"  ✅ Nombre cellules code conforme ({len(code_cells)})")

    # 2. Cellules code avec syntax error
    syntax_errors = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if not src.strip():
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            syntax_errors.append((i, e))
    if syntax_errors:
        ok = False
        print(f"  ❌ {len(syntax_errors)} cellule(s) code avec syntax error :")
        for i, e in syntax_errors:
            print(f"     cell {i}: {e}")
    else:
        print(f"  ✅ Toutes les cellules code parsent (ast.parse OK)")

    # 3. Code Python leaké dans des cellules markdown
    leaked = []
    code_fence_pat = re.compile(r"^```(?:python|py)\b", re.MULTILINE)
    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c["source"])
        if code_fence_pat.search(src):
            leaked.append(i)
    if leaked:
        ok = False
        print(f"  ❌ {len(leaked)} cellule(s) markdown contenant un bloc ```python (code leaké) :")
        for i in leaked[:10]:
            print(f"     cell {i}")
    else:
        print(f"  ✅ Zéro code Python leaké dans les cellules markdown")

    # 4. Cellules markdown avec titre + contenu
    title_content = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c["source"]).strip()
        if not src:
            continue
        first_line = src.split("\n")[0]
        if first_line.startswith("#"):
            rest = "\n".join(src.split("\n")[1:]).strip()
            if rest:
                title_content.append((i, first_line[:60]))
    if title_content:
        ok = False
        print(f"  ❌ {len(title_content)} cellule(s) markdown avec titre + contenu (viole consignes) :")
        for i, t in title_content[:10]:
            print(f"     cell {i}: {t}")
    else:
        print(f"  ✅ Tous les titres sont seuls dans leur cellule")

    # 5. Cellules code uniquement commentaires
    only_comments = []
    only_pseudo = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]).strip()
        if not src:
            continue
        if src.startswith('"""') and src.endswith('"""'):
            only_pseudo.append(i)
            continue
        lines = [l for l in src.split("\n") if l.strip()]
        if lines and all(l.strip().startswith("#") for l in lines):
            only_comments.append(i)
    if only_pseudo:
        ok = False
        print(f"  ❌ {len(only_pseudo)} cellule(s) code en triple-string (pseudo-code masqué) : {only_pseudo[:5]}")
    if only_comments:
        ok = False
        print(f"  ❌ {len(only_comments)} cellule(s) code uniquement commentées : {only_comments[:5]}")
    if not only_pseudo and not only_comments:
        print(f"  ✅ Aucune cellule code en pseudo-code / uniquement commentée")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit format notebook — étape 8 du workflow contrat.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pre", metavar="MD", help="Pré-check sur le .md")
    grp.add_argument("--post", metavar="IPYNB", help="Post-check sur le .ipynb")
    grp.add_argument("--both", nargs=2, metavar=("MD", "IPYNB"), help="Pre + post + vérification croisée")
    args = parser.parse_args()

    if args.pre:
        ok, _ = check_pre(Path(args.pre))
        return 0 if ok else 1

    if args.post:
        ok = check_post(Path(args.post))
        return 0 if ok else 1

    md_path, ipynb_path = Path(args.both[0]), Path(args.both[1])
    ok_pre, info = check_pre(md_path)
    ok_post = check_post(ipynb_path, expected_code_cells=info["expected_code_cells"])
    overall = ok_pre and ok_post
    print(f"\n=== RESUME : {'[OK] TOUT VERT' if overall else '[FAIL] AU MOINS UN CHECK ROUGE'} ===")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
