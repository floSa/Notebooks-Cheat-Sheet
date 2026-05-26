# ⚖️ Contrat de travail — Notebooks_convertion

Ce contrat est **binding**. Toute session de travail sur ce repo (humaine ou assistée) doit le respecter.

Il existe parce qu'une session antérieure a livré 44 notebooks annoncés "✅ fait" qui étaient en réalité défaillants :
- Générés depuis des connaissances générales **sans lire les notebooks d'origine** (juste un `Grep` des titres).
- Avec des "smoke tests" qui vérifiaient `import` mais **pas l'exécution des cellules dans l'ordre**.
- Contenant du code qui référence des variables jamais construites (`model.fit(X, y, ...)` sans `model`, `X`, `y`).
- Avec du contenu original supprimé (ex: TS_Maintenance : 1397 lignes de code original → 93 dans la refonte).

---

## 1. Définition de "fait" (5 critères binding)

Un notebook est marqué `✅ fait` **SI ET SEULEMENT SI les 5 critères suivants sont vérifiés** :

| # | Critère | Vérification |
|---|---|---|
| (a) | Le `.md` original dans `02_md_originaux/` a été **lu en entier** (pas juste les titres). Un mapping section-par-section est produit. | Fichier `scripts/_sandbox/plan_<nom>.md` existe : sections garde / modifie / supprime / ajoute, justification pour chaque suppression. |
| (b) | Un fichier `scripts/_sandbox/notebook_<nom>.py` contient les **cellules de code du notebook final dans l'ordre exact** et s'exécute end-to-end. | `uv run python scripts/_sandbox/notebook_<nom>.py` → exit code 0. |
| (c) | Aucune cellule code ne référence de variable / fonction non définie dans une cellule précédente. | Sous-cas de (b). |
| (d) | Aucune cellule code ne contient du pseudo-code (triple-string `"""..."""`, blocs entièrement en commentaires). | Audit automatique. |
| (e) | Le commit message liste explicitement : sections de l'original conservées, sections supprimées (+ pourquoi), sections nouvelles ajoutées. | Lisible dans `git log`. |

**Si un seul critère manque, le notebook n'est PAS fait.** Statut à utiliser à la place :

- `🟡 v0` : squelette posé, code non vérifié end-to-end.
- `🟠 partiel` : sections X, Y manquent par rapport à l'original.
- `🔴 cassé` : erreur connue.

---

## 2. Workflow obligatoire (ordre fixe — pas de raccourci)

Pour chaque notebook, dans l'ordre :

1. **Read** (pas Grep) le `.md` original. Si > 25k tokens, en plusieurs passes. **Lire tout le code**, pas seulement les titres.
2. **Plan détaillé** dans `scripts/_sandbox/plan_<nom>.md` : table sections-originales × décision (garde / modifie / supprime / ajoute). Justifier chaque suppression.
3. **Sandbox `.py`** : `scripts/_sandbox/notebook_<nom>.py` assemble linéairement le code prévu pour le notebook final. Données chargées, variables définies, modèles construits, tout dans l'ordre.
4. **Exécution** : `uv run python scripts/_sandbox/notebook_<nom>.py`. Itérer jusqu'à exit code 0 ET résultats cohérents (pas seulement imports OK).
5. **Écrire le `.md`** dans `03_md_ameliores/` en copiant le code testé du sandbox + l'enrobant des cellules markdown (titre seul, description, code, description, titre suivant).
6. **Vérifier descriptions** : entre une cellule code et un titre, il y a TOUJOURS une cellule MD description. Entre un titre et une cellule code, idem. Cf `00_consignes.md`.
7. **Convertir** : `uv run jupytext --to ipynb --output 04_notebooks_finaux/<nom>.ipynb 03_md_ameliores/<nom>.md`.
8. **Audit format** : 0 issue (titres seuls, 0 pseudo-code, 0 cellule code commentée). Sinon ne pas committer.
9. **Round-trip** : extraire le code de l'`.ipynb` final et le relancer. Doit tourner.
10. **Mettre à jour** `00_status_notebooks.md` avec le bon statut.
11. **Commit + push** avec le diff content listé dans le message.

---

## 3. Vocabulaire honnête (anti-mensonge)

### Interdit tant que les 5 critères ne sont pas vérifiés

- ❌ "✅ fait"
- ❌ "smoke test OK" (sauf si le smoke test est l'exécution end-to-end des cellules du notebook)
- ❌ "vérifié"
- ❌ "tout marche"

### À utiliser à la place

- 🟡 "v0 — squelette posé, code non exécuté end-to-end"
- 🟠 "partiel — sections X, Y manquent vs original"
- 🔴 "cassé — erreur ligne N"
- 📋 "audité format uniquement (titres seuls) — pas testé code"

### Toujours mentionner

- Si du contenu original a été supprimé : **lister ce qui a été supprimé** dans le commit.
- Si une cellule de code n'a pas été exécutée : le dire.
- Si une variable est référencée sans être construite : refuser de committer.

---

## 4. Périmètre — refus du "tout-en-un"

Si on demande N notebooks et que N × workflow_complet dépasse le budget contextuel raisonnable :

- **Première chose à dire** (avant de commencer) : "À workflow complet, je peux faire ~K notebooks par session. Tu en veux N. Soit on cible K bien faits, soit on dégrade explicitement le périmètre."
- **Jamais** de "je fais N en accélérant".
- **Si le contexte se charge** : sortir une livraison honnête de ce qui passe les 5 critères, marquer le reste avec son vrai statut.

Le temps n'est pas le problème. Le respect du workflow et l'honnêteté du rapporting le sont.

---

## 5. État actuel du projet

- **0 notebook** ne passe les 5 critères au moment où ce contrat est écrit.
- Les 44 fichiers dans `04_notebooks_finaux/` sont au mieux des 🟡 v0.
- Le `00_status_notebooks.md` est aligné sur cette réalité.
- Les `.md` originaux sont dans `02_md_originaux/` (gitignored, générables via `bash scripts/01_unzip.sh && bash scripts/02_ipynb_to_md.sh`).
- Env UV fonctionnel, scripts conversion OK.

## 6. Pour reprendre proprement

1. Lire ce contrat (tu le lis).
2. Lire `00_consignes.md` (règles de format) et `00_status_notebooks.md` (état).
3. Choisir **UN** notebook. Appliquer le workflow 1-11 ci-dessus en entier.
4. Quand il est vraiment ✅, commit + passer au suivant.

Pas de batch. Pas de "je fais la vague NLP en parallèle". **Un. À. La. Fois.**
