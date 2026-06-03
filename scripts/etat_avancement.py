#!/usr/bin/env python3
"""
etat_avancement.py — régénère 0_Documentation/00_FAIT_A_FAIRE.md à partir de l'état RÉEL du dépôt.

Logique (data-based, pas de message à interpréter) :
- Pour chaque notebook refait (2_New_Notebooks/ipynb/*.ipynb), on regarde la DATE du dernier
  commit qui l'a touché.
- A-t-il un original (1_Old_Notebooks/ipynb, via zip) ? → c'est une REFONTE ; sinon SUJET NEUF.
- REFONTE touchée le >= SEUIL (date où on a eu les vrais originaux) → ✅ FAIT.
  REFONTE plus ancienne → 🔧 À REFAIRE (faite avant, sur du vidé).
- check_format : 🐞 si le notebook a un bug de format (code dans markdown, etc.).

Usage : uv run python scripts/etat_avancement.py   (régénère le .md)
"""
import json, re, subprocess, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "1_Old_Notebooks" / "ipynb"
REFAIT = ROOT / "2_New_Notebooks" / "ipynb"
PLANS = ROOT / "2_New_Notebooks" / "plans"
OUT = ROOT / "0_Documentation" / "00_FAIT_A_FAIRE.md"
SEUIL = "2026-06-01"  # à partir d'ici on travaillait sur les VRAIS originaux

def norm(name: str) -> str:
    """ASCII, minuscule, alphanumérique seulement (accents/&/_/espaces supprimés)."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"\.ipynb$", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Renommages refait -> original (en espace `norm`). Seuls ceux qui ne matchent pas tout seuls.
ALIAS = {
    "detectionoutliers": "detectiondoutliers",                 # Détection D_outliers
    "structurepython": "structurepyhton",                      # typo Pyhton
    "structuregenererdonneesclassification": "structuregenererdonneespourclassification",
    "nlpclassificationsupervisee": "nlpclassificationspervise", # typo Spervisé
    "tsarima": "tsarimasrevoir",
    "tdsintroductiontraitementsignal": "tdsintroductionautraitementdusignal",
    "tsmaintenancepredictive": "tsmaintenancepredictivegood",
    "dlkankolmogorovarnold": "kankolmogorovarnoldnetworks",
    "dlkeras": "dltensorflowkeras",                            # split DL
    "dltensorflow": "dltensorflowkeras",                       # split DL
}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


REFONTE_RE = re.compile(r"refont|refait|refondu|5 crit|conforme contrat|selon workflow|end-to-end|blueprint|gabarit", re.I)
EXCLUDE_RE = re.compile(r"^(docs|chore|style|fix\(doc|refactor\(structure)", re.I)


def last_date(path: Path) -> str:
    """Date du VRAI commit de refonte le plus récent qui nomme ce notebook.
    Exclut les commits de doc / fix-doc / renommage (faux positifs)."""
    stem = path.stem.lower()
    log = git("log", "--format=%ad|%s", "--date=short")
    dates = []
    for ln in log.splitlines():
        date, subj = ln.split("|", 1)
        pos = subj.lower().find(stem)
        # le nom doit apparaître AU DÉBUT (vraie refonte "Nom : refonte..."),
        # pas mentionné en fin de message (ex. un commit de suivi qui cite le nom).
        if 0 <= pos <= 30 and REFONTE_RE.search(subj) and not EXCLUDE_RE.search(subj):
            dates.append(date)
    return max(dates) if dates else ""


def has_format_bug(path: Path) -> bool:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown" and "```python" in "".join(c.get("source", [])):
            return True
    return False


# index des originaux (clés normalisées ASCII)
orig_keys = {norm(p.name) for p in ORIG.glob("*.ipynb")}

plan_names = {p.stem for p in PLANS.glob("*.md") if p.stem != "README"}

faits, a_refaire, neufs = [], [], []
for p in sorted(REFAIT.glob("*.ipynb")):
    name = p.stem
    key = norm(p.name)
    target = ALIAS.get(key, key)
    has_orig = target in orig_keys
    date = last_date(p)
    bug = " 🐞" if has_format_bug(p) else ""
    if not has_orig:
        neufs.append(f"{name}{bug}")
    elif date >= SEUIL:
        faits.append(f"{name} ({date}){bug}")
    else:
        a_refaire.append(f"{name} ({date}){bug}")

# sujets neufs sans même un ipynb
for n in sorted(plan_names):
    if not (REFAIT / f"{n}.ipynb").exists():
        neufs.append(f"{n} (plan seul)")


def cols(items, n=3):
    rows = []
    for i in range(0, len(items), n):
        rows.append("| " + " | ".join(items[i:i+n] + [""] * (n - len(items[i:i+n]))) + " |")
    return "\n".join(rows)


md = f"""# ✅ Fait / 🔧 À faire — mises à jour 2026 (auto-généré)

> Régénéré par `uv run python scripts/etat_avancement.py`. Ne pas éditer à la main.
> Règle : refonte d'un de mes notebooks faite **après {SEUIL}** (= depuis qu'on a mes vrais
> originaux) → ✅ FAIT. Avant → 🔧 à refaire (faite sur du vidé). 🐞 = bug de format détecté.

## ✅ FAIT ({len(faits)})

| | | |
|---|---|---|
{cols(faits)}

## 🔧 À REFAIRE ({len(a_refaire)})

| | | |
|---|---|---|
{cols(a_refaire)}

## 🆕 À CRÉER — sujets neufs ({len(neufs)})

| | | |
|---|---|---|
{cols(neufs)}

---

**Compteur : {len(faits)} faits · {len(a_refaire)} à refaire · {len(neufs)} à créer.**
"""

OUT.write_text(md, encoding="utf-8")
print(f"✅ {OUT.relative_to(ROOT)} régénéré : {len(faits)} faits / {len(a_refaire)} à refaire / {len(neufs)} à créer")
