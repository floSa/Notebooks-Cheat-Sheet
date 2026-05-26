# ⚖️ Contrat de travail — Notebooks_convertion

Ce contrat est **binding**. Toute session de travail sur ce repo (humaine ou assistée) doit le respecter.

Il existe parce qu'une session antérieure a livré 44 notebooks annoncés "✅ fait" qui étaient en réalité défaillants :
- Générés depuis des connaissances générales **sans lire les notebooks d'origine** (juste un `Grep` des titres).
- Avec des "smoke tests" qui vérifiaient `import` mais **pas l'exécution des cellules dans l'ordre**.
- Contenant du code qui référence des variables jamais construites (`model.fit(X, y, ...)` sans `model`, `X`, `y`).
- Avec du contenu original supprimé (ex: TS_Maintenance : 1397 lignes de code original → 93 dans la refonte).
- Avec ~30 % de cellules code qui finissaient en cellule texte dans l'`.ipynb` (fences mal placées dans des régions).

---

## 1. Workflow obligatoire — 9 étapes dans l'ordre (zéro raccourci)

Pour CHAQUE notebook, dans l'ordre exact :

1. **`ipynb` → `md`** : convertir le notebook d'origine en markdown via `scripts/02_ipynb_to_md.sh` (ou jupytext direct).
2. **Lecture du `md` original** : `Read` (pas `Grep`) le `02_md_originaux/<nom>.md` **en entier**. Si > 25k tokens, en plusieurs passes avec `offset`/`limit`. Lire tout le code, pas seulement les titres.
3. **Plan d'amélioration** dans `scripts/_sandbox/plan_<nom>.md` suivant les règles ci-dessous :
   - **Préserver le contenu original** : table sections × décision (garde / refactore / supprime / ajoute), justifier chaque suppression.
   - **Format `00_consignes.md`** : titre seul dans sa cellule, description avant ET après chaque cellule code, marqueurs jupytext `<!-- #region -->` autour de chaque bloc markdown.
   - **Datasets** mutualisés (`00_datasets.md`).
   - **Recherche web 2026** si sujet fast-moving (LLMs, vector DBs, MLOps, agents, frameworks ML modernes).
   - **Refactor code brut** en fonctions typées + docstrings + commentaires utiles.
4. **Plan détaillé de la structure du nouveau notebook en `.md`** (`scripts/_sandbox/structure_<nom>.md`) :
   - Sections numérotées avec intention de chaque section.
   - Estimation nombre cellules code / markdown par section.
   - **AUCUN code** à cette étape.
5. **Code Python** : écrire `scripts/_sandbox/notebook_<nom>.py` qui contient **toutes les cellules code prévues, dans l'ordre exact**, comme si c'était un seul script. Données chargées, variables définies, modèles construits, tout dans l'ordre.
6. **Test du code** : `uv run python scripts/_sandbox/notebook_<nom>.py`. Itérer jusqu'à **exit code 0** ET résultats cohérents (pas seulement imports OK).
7. **Intégration du code dans le `.md`** : écrire `03_md_ameliores/<nom>.md` en **copiant-collant** le code testé du sandbox + enrobant des cellules markdown selon les règles de format.
8. **Audit format** (pre + post conversion) :
   ```bash
   uv run python scripts/check_format.py --both 03_md_ameliores/<nom>.md 04_notebooks_finaux/<nom>.ipynb
   ```
   Doit afficher `[OK] TOUT VERT`. Si **un seul** check rouge → corriger le `.md` et relancer. **Pas de commit tant que ce script n'est pas vert.**
9. **Conversion `md` → `ipynb`** : `uv run jupytext --to ipynb --output 04_notebooks_finaux/<nom>.ipynb 03_md_ameliores/<nom>.md`.

Puis : mettre à jour `00_status_notebooks.md` avec le statut réel + commit + push avec le diff content listé dans le message.

---

## 2. Définition de "fait" (5 critères binding)

Un notebook est marqué `✅ fait` **SI ET SEULEMENT SI les 5 critères suivants sont vérifiés** :

| # | Critère | Vérification |
|---|---|---|
| (a) | Le `.md` original a été **lu en entier** (pas juste les titres) | Plan `scripts/_sandbox/plan_<nom>.md` existe : table sections + décisions + justifications |
| (b) | `scripts/_sandbox/notebook_<nom>.py` s'exécute end-to-end | `uv run python ...` → exit code 0 |
| (c) | Aucune cellule code ne référence de variable / fonction non définie en amont | Sous-cas de (b) |
| (d) | `check_format.py --both` est **vert** (0 issue) | Script s'exécute sans `[FAIL]` |
| (e) | Le commit liste explicitement : sections conservées / supprimées (+ pourquoi) / ajoutées | Lisible dans `git log` |

**Si un seul critère manque, le notebook n'est PAS fait.** Statut à utiliser à la place :

- `🟡 v0` : squelette posé, code non vérifié end-to-end.
- `🟠 partiel` : sections X, Y manquent vs original.
- `🔴 cassé` : erreur connue.

---

## 3. Vocabulaire honnête (anti-mensonge)

### Interdit tant que les 5 critères ne sont pas vérifiés

- ❌ "✅ fait"
- ❌ "smoke test OK" (sauf si = exécution end-to-end des cellules du notebook, pas un simple `import`)
- ❌ "vérifié"
- ❌ "tout marche"

### À utiliser à la place

- 🟡 "v0 — squelette posé, code non exécuté end-to-end"
- 🟠 "partiel — sections X, Y manquent vs original"
- 🔴 "cassé — erreur ligne N"
- 📋 "audité format uniquement (titres seuls) — pas testé code"

### Toujours mentionner

- Si du contenu original a été supprimé → **lister ce qui a été supprimé** dans le commit.
- Si une cellule de code n'a pas été exécutée → le dire.
- Si une variable est référencée sans être construite → refuser de committer.

---

## 4. Le script `check_format.py` (étape 8, obligatoire)

Le script `scripts/check_format.py` détecte les 6 pathologies les plus courantes :

| # | Pathologie | Cause typique |
|---|---|---|
| 1 | ` ```python ` à l'intérieur d'une `<!-- #region -->` | Cellule code perdue en cellule texte après conversion |
| 2 | ` ````python ` (4 backticks) | Bloc reste en text dans le `.ipynb` |
| 3 | Cellule code avec syntax error (`ast.parse` échoue) | Code copié-collé incomplet |
| 4 | Cellule markdown contenant ` ```python ` | Code Python leaké dans du texte |
| 5 | Titre avec contenu en dessous dans la même cellule markdown | Viole "titre seul dans sa cellule" |
| 6 | Cellule code uniquement composée de commentaires ou de triple-string `"""..."""` | Pseudo-code masqué |

Usage :
```bash
# Pre-conversion : audit du .md
uv run python scripts/check_format.py --pre 03_md_ameliores/<nom>.md

# Post-conversion : audit du .ipynb
uv run python scripts/check_format.py --post 04_notebooks_finaux/<nom>.ipynb

# Les deux + vérification croisée (nb cellules code attendues == réelles)
uv run python scripts/check_format.py --both 03_md_ameliores/<nom>.md 04_notebooks_finaux/<nom>.ipynb
```

**Exit code 0 obligatoire avant tout commit.**

---

## 5. Périmètre — refus du "tout-en-un"

Si on demande N notebooks et que N × workflow_complet dépasse le budget contextuel raisonnable :

- **Première chose à dire** (avant de commencer) : "À workflow complet, je peux faire ~K notebooks par session. Tu en veux N. Soit on cible K bien faits, soit on dégrade explicitement le périmètre."
- **Jamais** de "je fais N en accélérant".
- **Si le contexte se charge** : sortir une livraison honnête de ce qui passe les 5 critères, marquer le reste avec son vrai statut.

Le temps n'est pas le problème. Le respect du workflow et l'honnêteté du rapporting le sont.

---

## 6. État actuel du projet

- **0 notebook** ne passe les 5 critères au moment où ce contrat est écrit.
- Les 44 fichiers dans `04_notebooks_finaux/` sont au mieux des 🟡 v0.
- Le `00_status_notebooks.md` est aligné sur cette réalité.
- Les `.md` originaux sont dans `02_md_originaux/` (gitignored, générables via `bash scripts/01_unzip.sh && bash scripts/02_ipynb_to_md.sh`).
- Env UV fonctionnel, scripts de conversion OK.
- Script `scripts/check_format.py` opérationnel (testé et trouvant des issues réelles).

## 7. Pour reprendre proprement

1. Lire ce contrat (tu le lis).
2. Lire `00_consignes.md` (règles de format) et `00_status_notebooks.md` (état).
3. Choisir **UN** notebook. Appliquer le workflow 1-9 ci-dessus en entier.
4. Lancer `check_format.py --both`. Doit être vert.
5. Quand les 5 critères sont vrais, commit + push + passer au suivant.

Pas de batch. Pas de "je fais la vague NLP en parallèle". **Un. À. La. Fois.**
