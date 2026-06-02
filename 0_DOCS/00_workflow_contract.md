# ⚖️ Contrat de travail — Notebooks-Cheat-Sheet

Ce contrat est **binding**. Toute session de travail sur ce repo (humaine ou assistée) doit le respecter.

Il existe parce que des sessions antérieures ont livré des notebooks annoncés "✅ fait" défaillants :
- Générés **sans lire les vrais notebooks d'origine** (les originaux n'étaient pas sur la machine ;
  les sessions lisaient des versions **gutées** — réduites de 200+ à ~20 cellules — venues de `main`).
- "Smoke tests" qui vérifiaient `import` mais **pas l'exécution des cellules dans l'ordre**.
- Code référençant des variables jamais construites ; contenu original supprimé ;
  ~30 % de cellules code finissant en cellule texte (fences mal placées).

Cf. `00_VERIFICATION_BASE.md` (quels refait ont vraiment utilisé l'original) et `00_FAIT_A_FAIRE.md`.

---

## 0. 🎯 RÈGLE D'OR (priorité absolue)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. TOUJOURS se baser sur MES VRAIS ORIGINAUX  →  1_MES_NOTEBOOKS/ipynb/   │
│    (régénérés via `bash scripts/restore_originaux.sh`)                    │
│    JAMAIS sur `main` ni sur 3_SESSIONS_RATEES/ (versions gutées à jeter). │
│                                                                           │
│ 2. S'INSPIRER de 2_NOUVEAUX/ (travail récent) UNIQUEMENT pour les 24      │
│    sujets NEUFS (DS_/DE_/AI_/MLE_/MLOps_ — ils n'ont pas d'original).     │
│    Pour tout notebook qui A un original, 2_NOUVEAUX/ n'est PAS une base.  │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Refonte d'un de MES notebooks** → source = `1_MES_NOTEBOOKS/ipynb/<nom>.ipynb`. On préserve
  mes sections, on modernise, on teste. `2_NOUVEAUX/` peut servir de simple référence d'idées,
  pas de point de départ.
- **Sujet NEUF (un des 24)** → pas d'original ; on part du plan (`2_NOUVEAUX/plans/`) et on
  s'inspire du refait existant (`2_NOUVEAUX/ipynb/`), à valider/tester intégralement.

---

## 1. Workflow obligatoire — 9 étapes dans l'ordre (zéro raccourci)

Pour CHAQUE notebook, dans l'ordre exact :

1. **S'assurer d'avoir les vrais originaux** : `bash scripts/restore_originaux.sh` (dézippe `1_MES_NOTEBOOKS/Notebooks.zip` → `1_MES_NOTEBOOKS/ipynb/`). Le `.md` correspondant est dans `1_MES_NOTEBOOKS/md/<nom>.md`.
2. **Lecture du `md` original** : `Read` (pas `Grep`) le `1_MES_NOTEBOOKS/md/<nom>.md` **en entier** (le VRAI original, pas un guté). Si > 25k tokens, en plusieurs passes avec `offset`/`limit`. Lire tout le code, pas seulement les titres.
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
7. **Intégration du code dans le `.md`** : écrire `2_NOUVEAUX/md/<nom>.md` en **copiant-collant** le code testé du sandbox + enrobant des cellules markdown selon les règles de format.
8. **Audit format** (pre + post conversion) :
   ```bash
   uv run python scripts/check_format.py --both 2_NOUVEAUX/md/<nom>.md 2_NOUVEAUX/ipynb/<nom>.ipynb
   ```
   Doit afficher `[OK] TOUT VERT`. Si **un seul** check rouge → corriger le `.md` et relancer. **Pas de commit tant que ce script n'est pas vert.**
9. **Conversion `md` → `ipynb`** : `uv run jupytext --to ipynb --output 2_NOUVEAUX/ipynb/<nom>.ipynb 2_NOUVEAUX/md/<nom>.md`.

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
uv run python scripts/check_format.py --pre 2_NOUVEAUX/md/<nom>.md

# Post-conversion : audit du .ipynb
uv run python scripts/check_format.py --post 2_NOUVEAUX/ipynb/<nom>.ipynb

# Les deux + vérification croisée (nb cellules code attendues == réelles)
uv run python scripts/check_format.py --both 2_NOUVEAUX/md/<nom>.md 2_NOUVEAUX/ipynb/<nom>.ipynb
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

## 6. UV / venv : **utiliser exclusivement UV côté WSL** sur ce projet

Le projet vit dans WSL (`~/Projets/Notebooks/Notebooks-Cheat-Sheet`).

**Piège** : créer le `.venv` avec **UV Windows** (depuis MSYS, PowerShell, ou via `\\wsl.localhost\...`) produit un venv avec des binaires Windows + symlink Linux `lib64` qui :

- N'est **pas utilisable depuis WSL** (pas de Python ELF natif).
- Casse le kernel Jupyter quand VSCode est en mode Remote-WSL.
- Casse `uv sync` au run suivant (UV Windows ne sait pas supprimer le symlink Linux et plante avec `error: failed to remove file ...lib64`).

### Setup one-time pour un nouveau poste / une nouvelle session WSL

Installer UV dans WSL (au choix) :

```bash
# Officiel
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ou via apt + pipx (sans curl|sh) :
sudo apt update && sudo apt install -y pipx && pipx ensurepath && pipx install uv
```

Puis créer le venv Linux + enregistrer le kernel côté Linux :

```bash
cd ~/Projets/Notebooks/Notebooks-Cheat-Sheet
rm -rf .venv
uv sync
uv run python -m ipykernel install --user --name=notebooks-refonte --display-name="Python (notebooks-refonte)"
```

Le kernel se retrouve à `~/.local/share/jupyter/kernels/notebooks-refonte/` et pointe sur `.venv/bin/python3` (Linux ELF). VSCode Remote-WSL le voit ; le notebook ouvre sans erreur.

### Règle pour l'assistant (Claude / autre) travaillant depuis Windows / MSYS

**TOUTES** les commandes `uv` doivent passer par `wsl -d ubuntu-24.04 -- bash -lc "..."`. Jamais d'`uv` natif depuis MSYS/PowerShell — ça recréerait un venv Windows et casserait le kernel utilisateur. Exemples :

```bash
# Depuis MSYS / PowerShell :
wsl -d ubuntu-24.04 -- bash -lc "cd ~/Projets/Notebooks/Notebooks-Cheat-Sheet && uv sync"
wsl -d ubuntu-24.04 -- bash -lc "cd ~/Projets/Notebooks/Notebooks-Cheat-Sheet && uv add <pkg>"
wsl -d ubuntu-24.04 -- bash -lc "cd ~/Projets/Notebooks/Notebooks-Cheat-Sheet && uv run python scripts/_sandbox/notebook_<nom>.py"
wsl -d ubuntu-24.04 -- bash -lc "cd ~/Projets/Notebooks/Notebooks-Cheat-Sheet && uv run python scripts/check_format.py --both <md> <ipynb>"
```

Si l'assistant se retrouve avec un `.venv` cassé Windows ↔ WSL (`error: failed to remove file ...lib64`) : c'est qu'il a accidentellement lancé UV Windows. Fix : `wsl -d ubuntu-24.04 -- bash -c "rm -rf ~/Projets/Notebooks/Notebooks-Cheat-Sheet/.venv"` puis re-sync via WSL.

### Piège du kernel fantôme (doublon homonyme)

Lancer `ipykernel install` **depuis Windows** (`uv run python -m ipykernel install ...` en MSYS) enregistre un kernel dans `C:\Users\FLORIAN\AppData\Roaming\jupyter\kernels\notebooks-refonte` pointant vers `.venv\Scripts\python.exe` (chemin Windows). Comme le venv réel est Linux (`.venv/bin/python3`), ce kernel est **cassé** et apparaît dans VSCode comme `notebooks-refonte (broken) (Python 3.12.10)` — **avec le même nom** que le bon kernel Linux, donc indistinguable dans le sélecteur. VSCode tourne en boucle si l'utilisateur sélectionne le mauvais.

**Règle** : `ipykernel install` se lance **uniquement via WSL** (comme tout `uv ...`). Le bon kernel vit à `~/.local/share/jupyter/kernels/notebooks-refonte/` côté Linux.

**Fix si le doublon existe** : supprimer le fantôme Windows :
```bash
rm -rf "C:/Users/FLORIAN/AppData/Roaming/jupyter/kernels/notebooks-refonte"
```
Puis recharger VSCode et re-sélectionner le kernel (il n'en reste qu'un, le Linux).

---

## 7. Kernel Jupyter (à ré-enregistrer après tout `uv sync` qui recrée `.venv`)

Si le `.venv` est supprimé/recréé (par exemple parce que `uv sync` a échoué et il a fallu `rm -rf .venv` côté WSL), le kernel Jupyter `notebooks-refonte` pointe vers un chemin disparu et Jupyter/VSCode affichera :

> `The kernel failed to start as the Python Environment 'notebooks-refonte' is no longer available.`

**Fix systématique** après tout `uv sync` qui recrée l'environnement :

```bash
uv run python -m ipykernel install --user --name=notebooks-refonte --display-name="Python (notebooks-refonte)"
```

C'est idempotent — peut être lancé sans risque même si le kernel existe déjà. À intégrer aussi dans tout script de setup qui réinstalle l'env.

---

## 8. État actuel du projet

- **0 notebook** ne passe les 5 critères AU SENS DE LA RÈGLE D'OR (basé sur le vrai original).
- Les 44 fichiers dans `2_NOUVEAUX/ipynb/` ont presque tous été bâtis sur des **gutés** :
  seuls `DL_Deep_Learning_Maths` et `EDA_Stats_Analyse_Desc_Visual` ont vraiment utilisé l'original
  (cf. `00_VERIFICATION_BASE.md`). Tout le reste est à refaire/recontrôler sur l'original.
- L'état complet par notebook est dans **`00_FAIT_A_FAIRE.md`**.
- Les vrais originaux sont dans `1_MES_NOTEBOOKS/` (zip committé + `ipynb/` gitignoré régénérable
  via `bash scripts/restore_originaux.sh` + `md/` jupytext).
- Env UV fonctionnel ; `scripts/check_format.py` et `scripts/verifier_base.py` opérationnels.

## 9. Règles couleurs dans les notebooks de viz / EDA

Pour éviter les "12000 styles différents" :

| Cas | Règle |
|---|---|
| **Variable univariée** (1 var représentée 1+ fois) | **1 seule couleur** = `primary_1` partout. Pas de variation hist/box/KDE en couleurs différentes. |
| **Multi-catégorie sans ordre naturel** (ex: `embarked` S/C/Q) | **Soit 1 couleur uniforme, soit N couleurs distinctes**. Pas de panachage 1+beige+beige. Pour N : utiliser `PALETTE[:N]` dans l'ordre. |
| **Multi-catégorie avec ordre sémantique bon→mauvais** (ex: `pclass 1<2<3`, `survived 0/1`) | `[accent, moyen, mauvais]` pour 3 niveaux ; `[accent, mauvais]` pour binaire (good/bad). |
| **Highlight d'une modalité spécifique** (ex: la max) | La modalité d'intérêt en `accent_dark`, les autres en `primary_1` (ou `beige` si on veut estomper). |
| **Séries continues neutres** (capteurs, métriques sans good/bad) | Couleurs neutres de la palette : `primary_1`, `lavender`, `dusty_rose`, `beige`. |
| **Heatmap continue** (corrélations, taux) | Cmap matplotlib divergente (`RdBu_r` centré sur 0 / 0.5) — pas la palette CHART. |

La palette CHART est :
```
PALETTE = [primary_1, mauvais, moyen, accent, accent_dark, lavender, dusty_rose, beige]
            #00798c   #d1495b   #edae49 #66a182  #2e4057    #9d83b8   #b8848e    #c9b78b
```

---

## 10. Pour reprendre proprement

1. Lire ce contrat (tu le lis).
2. Lire `00_consignes.md` (règles de format) et `00_status_notebooks.md` (état).
3. Choisir **UN** notebook. Appliquer le workflow 1-9 ci-dessus en entier.
4. Lancer `check_format.py --both`. Doit être vert.
5. Quand les 5 critères sont vrais, commit + push + passer au suivant.

Pas de batch. Pas de "je fais la vague NLP en parallèle". **Un. À. La. Fois.**

---

## 11. Git : push via SSH (alias `github.com-perso`)

Le remote `origin` est cloné en **HTTPS** (`https://github.com/floSa/Notebooks-Cheat-Sheet.git`)
mais **aucun identifiant HTTPS n'est configuré côté WSL** (pas de credential helper, pas de `gh`,
pas de `~/.git-credentials`). Un `git push` sur l'URL HTTPS **se bloque** en attente d'un login
interactif — impossible en mode non-interactif (assistant).

**Le push se fait par SSH**, via le bon alias d'hôte (le compte perso `floSa` possède ce repo) :

```bash
# ~/.ssh/config définit deux alias :
#   github.com-perso  → clé id_ed25519_github_perso  (compte floSa — CE repo)
#   github.com-pro    → clé id_ed25519_github_pro     (compte Florian-H-AOSIS)

# Option A (recommandée) — basculer origin en SSH perso une fois pour toutes :
git remote set-url origin git@github.com-perso:floSa/Notebooks-Cheat-Sheet.git
git push origin feat/restart-jupytext-workflow

# Vérifier l'auth SSH :
ssh -T git@github.com-perso   # doit répondre "Hi floSa!"
```

> ⚠️ Ne **jamais** utiliser `github.com-pro` pour ce repo (compte différent → permission denied).
> Toujours pousser sur `feat/restart-jupytext-workflow`, jamais sur `main`.

---

## 12. Multi-agents / multi-sessions — isolation par worktrees

Plusieurs agents/sessions peuvent travailler **en parallèle**, mais **chacun sur un notebook différent**. Le danger historique : partager un seul working tree + une seule branche → un `git add -A` d'une session embarque les fichiers non-committés des autres, ou une exécution de notebook salit l'arbre commun, ou un `reset/pull` clobbe le travail en cours d'un autre.

**Règle : un worktree git + une branche par notebook en cours.** Un agent ne travaille jamais directement dans le dossier principal partagé.

### Démarrer un notebook

```bash
cd ~/mes_projets/Notebooks_convertion
git worktree add ../wt-<nom_notebook> -b feat/nb-<nom_notebook>
cd ../wt-<nom_notebook>
# … appliquer le workflow §1 (1-9) ici, en isolation totale …
```

Le `.venv` et le kernel sont partagés (côté WSL) ; seul le code/markdown du notebook est isolé. La source des originaux (`1_MES_NOTEBOOKS/Notebooks.zip` + `md/`) est committée donc présente dans chaque worktree.

### Committer (par pathspec, jamais en masse)

- **JAMAIS** `git add -A`, `git add .`, ni `git commit -a`.
- Toujours committer SES fichiers par chemin explicite, p. ex. :
  `git commit -m "…" -- 2_NOUVEAUX/ipynb/<nom>.ipynb 2_NOUVEAUX/md/<nom>.md 0_DOCS/00_FAIT_A_FAIRE.md`
- Messages **sans** mention `claude` / `anthropic` / `Co-Authored-By` (règle repo).

### Intégrer sur `feat` (sérialisé)

```bash
cd ~/mes_projets/Notebooks_convertion        # worktree principal sur feat
git fetch origin && git merge --ff-only origin/feat/restart-jupytext-workflow   # se remettre a jour AVANT
git merge --no-ff feat/nb-<nom_notebook>     # une intégration à la fois
git push origin feat/restart-jupytext-workflow
git worktree remove ../wt-<nom_notebook> && git branch -d feat/nb-<nom_notebook>
```

Comme les notebooks diffèrent, les merges ne se chevauchent pas. **Si le push est rejeté (non-fast-forward) : ne pas forcer** ; `git fetch` + rebaser/merger le distant, puis re-push. Sérialiser : une seule intégration à la fois.

### Ne pas re-salir l'arbre

- **Ne pas exécuter les `.ipynb` finaux** (pas de nbstripout sur ce repo) : lancer un notebook réinjecte des sorties → diffs parasites. La conformité d'exécution est garantie par l'étape 6 (sandbox `.py`), pas par l'exécution du `.ipynb`.
- Les artefacts d'exécution (`catboost_info/`, `data_mnist/`, `runs/`, `model/`, `*.keras`, `*.pt`, `*.html`…) ne se committent pas.
