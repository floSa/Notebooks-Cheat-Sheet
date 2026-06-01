#!/usr/bin/env python3
"""
verifier_base.py — Détermine, pour chaque notebook REFAIT, s'il s'est basé sur
le VRAI original ou seulement sur la version GUTÉE.

Principe : on extrait des "signatures" (def/class + titres markdown) de l'original
et du guté. Les signatures présentes dans l'original MAIS PAS dans le guté sont le
"signal original-only". Si le refait contient beaucoup de ce signal, il s'est basé
sur le vrai original ; sinon, seulement sur le guté.

Usage : uv run python scripts/verifier_base.py
"""
import json, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "1_MES_NOTEBOOKS" / "ipynb"
GUTE = ROOT / "3_SESSIONS_RATEES" / "ipynb"
REFAIT = ROOT / "2_NOUVEAUX" / "ipynb"


def norm(name: str) -> str:
    """Normalise un nom de fichier pour le matching (sans accents/casse/séparateurs)."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"\.ipynb$", "", s)
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return s


def signatures(path: Path) -> set[str]:
    """Extrait def/class names + titres markdown d'un notebook."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    sigs = set()
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code":
            for m in re.findall(r"^\s*(?:def|class)\s+([A-Za-z_]\w+)", src, re.M):
                sigs.add("fn:" + m.lower())
        else:
            for m in re.findall(r"^#{1,6}\s+(.{4,60})", src, re.M):
                sigs.add("h:" + re.sub(r"[^a-z0-9]", "", m.lower())[:40])
    return sigs


def index(folder: Path) -> dict[str, Path]:
    return {norm(p.name): p for p in folder.glob("*.ipynb")}


# Alias : refait (clé normalisée) -> nom de fichier original (renommages/typos/splits)
ALIAS = {
    "detectionoutliers": "Détection D_outliers.ipynb",
    "structurepython": "Structure_Pyhton.ipynb",
    "structuregenererdonneesclassification": "Structure_Generer_Données_pour_Classification.ipynb",
    "nlpclassificationsupervisee": "NLP_Classification_Spervisé.ipynb",
    "tsarima": "TS_ARIMAs_Revoir.ipynb",
    "tdsintroductiontraitementsignal": "TdS_Introduction au Traitement du Signal.ipynb",
    "dlkeras": "DL_Tensorflow_Keras.ipynb",
    "dltensorflow": "DL_Tensorflow_Keras.ipynb",
    "dlkankolmogorovarnold": "KAN (Kolmogorov-Arnold Networks).ipynb",
    "tsmaintenancepredictive": "TS_Maintenance_Predictive_GOOD.ipynb",
    "mlregressionclassificationmultiple": "ML_Régression_&_Classification_Multiple.ipynb",
}

orig_idx, gute_idx, refait_idx = index(ORIG), index(GUTE), index(REFAIT)
# applique les alias
for k, fname in ALIAS.items():
    p = ORIG / fname
    if p.exists():
        orig_idx[k] = p

print(f"{'Refait':<42}{'orig-only':>10}{'repris':>8}{'%':>6}  verdict")
print("-" * 84)

rows = []
for key, rpath in sorted(refait_idx.items()):
    o = orig_idx.get(key)
    g = gute_idx.get(key)
    if not o:
        rows.append((rpath.stem, None, None, None, "pas d'original (sujet neuf)"))
        continue
    so = signatures(o)
    sg = signatures(g) if g else set()
    sr = signatures(rpath)
    orig_only = so - sg                       # signal présent SEULEMENT dans le vrai original
    repris = orig_only & sr                   # combien le refait en reprend
    n_only = len(orig_only)
    n_rep = len(repris)
    pct = (100 * n_rep / n_only) if n_only else 0
    if n_only < 3:
        verdict = "indéterminé (original ≈ guté)"
    elif pct >= 25:
        verdict = "✅ basé sur TON ORIGINAL"
    elif pct >= 8:
        verdict = "🟠 partiellement"
    else:
        verdict = "❌ basé sur le GUTÉ seulement"
    rows.append((rpath.stem, n_only, n_rep, pct, verdict))

for stem, n_only, n_rep, pct, verdict in rows:
    if n_only is None:
        print(f"{stem:<42}{'—':>10}{'—':>8}{'—':>6}  {verdict}")
    else:
        print(f"{stem:<42}{n_only:>10}{n_rep:>8}{pct:>5.0f}%  {verdict}")
