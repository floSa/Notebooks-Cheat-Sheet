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
# Web API en Python — Flask & FastAPI
<!-- #endregion -->

<!-- #region -->
Ce notebook prend en main **deux** frameworks d'API REST Python sur **exactement la même API** — un mini-catalogue de livres `books` — puis les **compare** à conditions strictement égales.

- **Partie A — Flask** : le micro-framework WSGI historique. Simple, mature, immense écosystème de plugins.
- **Partie B — FastAPI** : le framework ASGI moderne (async, type-safe via Pydantic, OpenAPI généré automatiquement). Standard de fait en 2026.
- **Partie C — Comparatif** : même route écrite des deux façons, tableau de décision, perfs, et « quand choisir quoi ».

La ressource est volontairement identique partout : `book = {id, title, author, year?}`, seedée avec trois classiques.
<!-- #endregion -->

<!-- #region -->
**Comment on exécute une API dans un notebook.** On ne lance jamais `app.run()` (Flask) ni `uvicorn` (FastAPI) depuis une cellule : ces appels démarrent un serveur **bloquant** qui ne rend jamais la main. À la place, on exerce l'application **en mémoire (in-process)** via son client de test — `app.test_client()` pour Flask, `TestClient(app)` (basé sur httpx) pour FastAPI. Aucun port n'est ouvert, tout est synchrone et reproductible.

Pour faire tourner ces APIs comme de **vrais serveurs** (curl, navigateur, Swagger UI), des versions autonomes lançables existent dans `apps/flask_api/app.py` et `apps/fastapi_api/main.py` (voir Partie C).
<!-- #endregion -->

<!-- #region -->
On commence par le matériel commun aux deux parties : les imports de base, les données `books` partagées, et un petit helper d'affichage de réponse.
<!-- #endregion -->

```python
from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Any

# Données partagées : catalogue `books` (trois classiques).
SEED_BOOKS: list[dict[str, Any]] = [
    {"id": 1, "title": "Le Petit Prince", "author": "Antoine de Saint-Exupéry", "year": 1943},
    {"id": 2, "title": "1984", "author": "George Orwell", "year": 1949},
    {"id": 3, "title": "L'Étranger", "author": "Albert Camus", "year": 1942},
]


def show(label: str, status_code: int, body: Any) -> None:
    """Affiche proprement une réponse HTTP (code + payload) — helper de démo."""
    rendered = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else body
    print(f"[{label}] {status_code} -> {rendered}")
```

<!-- #region -->
## Partie A — Flask
<!-- #endregion -->

<!-- #region -->
**Flask** est un micro-framework **WSGI** (synchrone, un thread par requête). Il fournit le strict minimum (routing, requête/réponse) et délègue le reste à des extensions (`Flask-SQLAlchemy`, `Flask-Login`, `Flask-Caching`…). C'est un excellent choix pour les petites APIs, le legacy, ou l'enseignement.
<!-- #endregion -->

<!-- #region -->
### 1. Hello world
<!-- #endregion -->

<!-- #region -->
Une application Flask = un objet `Flask` + des fonctions décorées par une route. `@app.get("/")` (raccourci Flask 3.x de `@app.route("/", methods=["GET"])`) associe l'URL `/` à la fonction. `jsonify` sérialise un dict en réponse `application/json`.
<!-- #endregion -->

```python
from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import HTTPException, NotFound
from werkzeug.utils import secure_filename

hello_app = Flask("hello")


@hello_app.get("/")
def hello() -> Response:
    """Route racine minimale."""
    return jsonify({"message": "Hello, Flask!", "status": "ok"})
```

<!-- #region -->
On exerce la route via le client de test : aucune socket, on obtient directement l'objet réponse.
<!-- #endregion -->

```python
with hello_app.test_client() as c:
    r = c.get("/")
    show("GET /", r.status_code, r.get_json())
```

<!-- #region -->
### 2. CRUD complet sur `books`
<!-- #endregion -->

<!-- #region -->
On définit l'application principale qui porte tout le cycle de vie d'une ressource REST :

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/books` | liste (avec filtre `?author=`) |
| `GET` | `/books/<id>` | détail (404 si absent) |
| `POST` | `/books` | création (validation manuelle) |
| `POST` | `/books/validated` | création (validation Pydantic) |
| `PUT` | `/books/<id>` | mise à jour |
| `DELETE` | `/books/<id>` | suppression (204) |
| `GET` | `/admin/books` | route protégée par clé API |

Le stockage est un simple `dict` en mémoire (à remplacer par une base en production). On commence par l'app, le store, le helper d'id, le schéma de validation Pydantic et le décorateur d'authentification.
<!-- #endregion -->

```python
from functools import wraps

from pydantic import BaseModel, Field, ValidationError

books_app = Flask("books")
_BOOKS: dict[int, dict[str, Any]] = {b["id"]: dict(b) for b in SEED_BOOKS}
_NEXT_ID: int = max(_BOOKS) + 1
API_KEY: str = os.environ.get("FLASK_API_KEY", "dev-key")


def _next_id() -> int:
    """Renvoie le prochain id libre du catalogue."""
    return max(_BOOKS, default=0) + 1


class BookIn(BaseModel):
    """Schéma de validation (Pydantic v2) pour la création d'un livre côté Flask."""

    title: str = Field(min_length=1, max_length=200)
    author: str = Field(min_length=1, max_length=120)
    year: int | None = Field(default=None, ge=0, le=2100)


def require_api_key(fn):
    """Décorateur : exige un header `X-API-Key` valide, sinon 401."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        if request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)

    return wrapper
```

<!-- #region -->
On déclare ensuite les routes. À noter : `request.get_json(silent=True)` renvoie `None` (au lieu de lever) si le corps n'est pas du JSON valide ; les codes de retour explicites (`201`, `204`) suivent la sémantique REST ; les 404 sont levées via `werkzeug.NotFound` et formatées par le gestionnaire d'erreurs centralisé (section 5).
<!-- #endregion -->

```python
@books_app.get("/books")
def list_books() -> Response:
    """Liste les livres ; filtre optionnel `?author=` (query param)."""
    author = request.args.get("author")
    items = list(_BOOKS.values())
    if author:
        items = [b for b in items if author.lower() in b["author"].lower()]
    return jsonify({"books": items, "count": len(items)})


@books_app.get("/books/<int:book_id>")
def get_book(book_id: int) -> Response:
    """Récupère un livre par id (404 si absent)."""
    book = _BOOKS.get(book_id)
    if book is None:
        raise NotFound(f"Book {book_id} not found")
    return jsonify(book)


@books_app.post("/books")
def create_book() -> tuple[Response, int]:
    """Crée un livre (validation manuelle simple : title + author requis)."""
    global _NEXT_ID
    data = request.get_json(silent=True) or {}
    if not data.get("title") or not data.get("author"):
        return jsonify({"error": "title and author required"}), 400
    book = {
        "id": _next_id(),
        "title": data["title"],
        "author": data["author"],
        "year": data.get("year"),
    }
    _BOOKS[book["id"]] = book
    _NEXT_ID = book["id"] + 1
    return jsonify(book), 201


@books_app.post("/books/validated")
def create_book_validated() -> tuple[Response, int]:
    """Crée un livre avec validation Pydantic v2 (422 si payload invalide)."""
    try:
        payload = BookIn.model_validate(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"errors": e.errors(include_url=False)}), 422
    book = {"id": _next_id(), **payload.model_dump()}
    _BOOKS[book["id"]] = book
    return jsonify(book), 201


@books_app.put("/books/<int:book_id>")
def update_book(book_id: int) -> Response:
    """Met à jour un livre existant (404 si absent)."""
    if book_id not in _BOOKS:
        raise NotFound(f"Book {book_id} not found")
    data = request.get_json(silent=True) or {}
    _BOOKS[book_id].update({k: v for k, v in data.items() if k != "id"})
    return jsonify(_BOOKS[book_id])


@books_app.delete("/books/<int:book_id>")
def delete_book(book_id: int) -> tuple[str, int]:
    """Supprime un livre (404 si absent, 204 sinon)."""
    if book_id not in _BOOKS:
        raise NotFound(f"Book {book_id} not found")
    del _BOOKS[book_id]
    return "", 204


@books_app.get("/admin/books")
@require_api_key
def admin_books() -> Response:
    """Route protégée par clé API (illustration auth)."""
    return jsonify({"count": len(_BOOKS), "books": list(_BOOKS.values())})
```

<!-- #region -->
Enfin, les **gestionnaires d'erreurs centralisés** : toute `HTTPException` (404, 400…) est formatée uniformément en JSON, et un filet attrape les exceptions imprévues en `500`.
<!-- #endregion -->

```python
@books_app.errorhandler(HTTPException)
def handle_http(e: HTTPException) -> tuple[Response, int]:
    """Formate toute HTTPException en JSON `{error, code}`."""
    return jsonify({"error": e.description, "code": e.code}), e.code or 500


@books_app.errorhandler(Exception)
def handle_unexpected(e: Exception) -> tuple[Response, int]:  # noqa: ARG001
    """Filet de sécurité : 500 JSON pour toute exception non gérée."""
    return jsonify({"error": "Internal server error"}), 500
```

<!-- #region -->
> **Fidélité à l'original & modernisation.** Cette API reprend exactement les routes du notebook Flask d'origine (liste / détail / création / update / suppression + handlers 404 et 400) sur les mêmes données. Ajouts 2026 : typage complet, docstrings, codes REST stricts (201/204), filtre query, validation Pydantic, auth, et gestion d'erreurs centralisée via `HTTPException`.
<!-- #endregion -->

<!-- #region -->
### 3. Démo CRUD exécutée
<!-- #endregion -->

<!-- #region -->
On déroule un cycle de vie complet : lister, créer, lire, mettre à jour, supprimer, puis tester un id inexistant pour vérifier la 404.
<!-- #endregion -->

```python
with books_app.test_client() as c:
    r = c.get("/books")
    show("GET /books", r.status_code, r.get_json())
    r = c.post("/books", json={"title": "Fahrenheit 451", "author": "Ray Bradbury", "year": 1953})
    show("POST /books", r.status_code, r.get_json())
    new_id = r.get_json()["id"]
    r = c.get(f"/books/{new_id}")
    show("GET /books/{id}", r.status_code, r.get_json())
    r = c.put(f"/books/{new_id}", json={"year": 1954})
    show("PUT /books/{id}", r.status_code, r.get_json())
    r = c.delete(f"/books/{new_id}")
    show("DELETE /books/{id}", r.status_code, "(no content)")
    r = c.get("/books/9999")
    show("GET /books/9999", r.status_code, r.get_json())
```

<!-- #region -->
### 4. JSON I/O, path & query params
<!-- #endregion -->

<!-- #region -->
Trois sources de données dans une requête Flask :

- **Body JSON** : `request.get_json(silent=True)`.
- **Path param** : capturé par le convertisseur de route `<int:book_id>` (typé et validé par Flask).
- **Query string** : `request.args.get("author")` pour `/books?author=...`.

On vérifie le filtre par auteur.
<!-- #endregion -->

```python
with books_app.test_client() as c:
    r = c.get("/books?author=Orwell")
    show("GET /books?author=Orwell", r.status_code, r.get_json())
```

<!-- #region -->
### 5. Validation avec Pydantic
<!-- #endregion -->

<!-- #region -->
Flask n'a **pas** de validation native (contrairement à FastAPI). On peut valider à la main, ou — recommandé en 2026 — déléguer à **Pydantic v2** : on valide le corps avec `BookIn.model_validate(...)` et on renvoie `422 Unprocessable Entity` avec le détail des erreurs en cas d'échec. Alternatives : `marshmallow`, `flask-pydantic`.
<!-- #endregion -->

```python
with books_app.test_client() as c:
    r = c.post("/books/validated", json={"title": "Dune", "author": "Frank Herbert", "year": 1965})
    show("POST valid", r.status_code, r.get_json())
    r = c.post("/books/validated", json={"title": "", "year": 9999})
    show("POST invalid", r.status_code, f"{len(r.get_json()['errors'])} erreurs de validation")
```

<!-- #region -->
### 6. Gestion d'erreurs centralisée
<!-- #endregion -->

<!-- #region -->
Grâce aux `@errorhandler` définis sur l'app, on n'a pas à formater les erreurs route par route : toute `HTTPException` levée (ici une 404) ressort avec la même enveloppe `{error, code}`.
<!-- #endregion -->

```python
with books_app.test_client() as c:
    r = c.get("/books/424242")
    show("404 formatée", r.status_code, r.get_json())
```

<!-- #region -->
### 7. Authentification par clé API
<!-- #endregion -->

<!-- #region -->
Le décorateur `require_api_key` protège une route : sans header `X-API-Key` valide → `401`. Pattern simple et courant pour du service-à-service. Pour de l'auth utilisateur (tokens, refresh, scopes), passer à `flask-jwt-extended`.
<!-- #endregion -->

```python
with books_app.test_client() as c:
    r = c.get("/admin/books")
    show("sans clé", r.status_code, r.get_json())
    r = c.get("/admin/books", headers={"X-API-Key": API_KEY})
    show("avec clé", r.status_code, f"{r.get_json()['count']} livres")
```

<!-- #region -->
### 8. Différents types de payload
<!-- #endregion -->

<!-- #region -->
Au-delà du JSON, une API reçoit souvent des formulaires, des fichiers, ou doit streamer de gros volumes.
<!-- #endregion -->

<!-- #region -->
#### 8.1 Form data
<!-- #endregion -->

<!-- #region -->
Les formulaires HTML (`application/x-www-form-urlencoded`) arrivent dans `request.form`, pas dans `request.get_json()`.
<!-- #endregion -->

```python
form_app = Flask("form")


@form_app.post("/login")
def login() -> Response:
    """Lecture d'un payload `application/x-www-form-urlencoded`."""
    return jsonify({"user": request.form.get("username")})


with form_app.test_client() as c:
    r = c.post("/login", data={"username": "alice", "password": "secret"})
    show("POST form", r.status_code, r.get_json())
```

<!-- #region -->
#### 8.2 File upload
<!-- #endregion -->

<!-- #region -->
Les fichiers (`multipart/form-data`) arrivent dans `request.files`. **Toujours** assainir le nom avec `secure_filename` avant de l'écrire sur disque (sinon faille de traversée de répertoire).
<!-- #endregion -->

```python
upload_app = Flask("upload")
UPLOAD_DIR = tempfile.mkdtemp(prefix="flask_uploads_")


@upload_app.post("/upload")
def upload() -> Response:
    """Réception d'un fichier `multipart/form-data`."""
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    safe = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    f.save(path)
    return jsonify({"filename": safe, "size": os.path.getsize(path)})


with upload_app.test_client() as c:
    r = c.post(
        "/upload",
        data={"file": (io.BytesIO(b"hello upload"), "note.txt")},
        content_type="multipart/form-data",
    )
    show("POST upload", r.status_code, r.get_json())
```

<!-- #region -->
#### 8.3 Streaming
<!-- #endregion -->

<!-- #region -->
Pour renvoyer un gros volume sans tout charger en mémoire, on retourne une `Response` autour d'un **générateur**. `stream_with_context` garde le contexte de requête disponible pendant la génération.
<!-- #endregion -->

```python
stream_app = Flask("stream")


@stream_app.get("/stream/<int:n>")
def stream_numbers(n: int) -> Response:
    """Réponse en streaming (génère n lignes sans tout charger en mémoire)."""

    def generate():
        for i in range(n):
            yield f"{i}\n"

    return Response(stream_with_context(generate()), mimetype="text/plain")


with stream_app.test_client() as c:
    r = c.get("/stream/5")
    show("GET /stream/5", r.status_code, repr(r.get_data(as_text=True)))
```

<!-- #region -->
### 9. Tests avec `test_client`
<!-- #endregion -->

<!-- #region -->
Le client de test sert aussi pour les **tests automatisés**. En pytest on l'expose via une fixture (`@pytest.fixture` renvoyant `app.test_client()`), puis on asserte sur `status_code` et `json`. Ici on exécute directement quelques assertions.
<!-- #endregion -->

```python
with books_app.test_client() as c:
    assert c.post("/books", json={"title": "T", "author": "A"}).status_code == 201
    assert c.get("/books/9999").status_code == 404
    print("Tests Flask OK")
```

<!-- #region -->
### 10. Mise en production
<!-- #endregion -->

<!-- #region -->
**Jamais** `app.run()` en production : c'est le serveur de développement (mono-thread, debug activé = exécution de code arbitraire). On sert l'app WSGI derrière **Gunicorn** :

```bash
# 4 workers, l'objet `app` du module apps/flask_api/app.py
gunicorn apps.flask_api.app:app -w 4 -b 0.0.0.0:5000
```

En production réelle : Gunicorn dans un conteneur Docker, derrière un reverse proxy (nginx, Caddy, Traefik) pour le TLS, le rate-limiting et les fichiers statiques.
<!-- #endregion -->

<!-- #region -->
### 11. OpenAPI / Swagger côté Flask
<!-- #endregion -->

<!-- #region -->
Flask **ne génère pas** la spec OpenAPI automatiquement — c'est son principal manque face à FastAPI. Pour l'obtenir :

- **flask-smorest** : génère l'OpenAPI à partir de blueprints + schémas marshmallow.
- **flask-pydantic** : validation + doc basées sur des modèles Pydantic.
- **flasgger** : OpenAPI à partir d'annotations dans les docstrings.

On verra en Partie B que FastAPI fournit tout cela sans aucune ligne supplémentaire.
<!-- #endregion -->

<!-- #region -->
## Partie B — FastAPI
<!-- #endregion -->

<!-- #region -->
**FastAPI** est bâti sur **Starlette** (ASGI, async natif) et **Pydantic** (validation par les types). Ses arguments de fonction typés font à la fois le **parsing**, la **validation** et la **documentation OpenAPI** — sans code supplémentaire. C'est le standard 2026 pour les nouvelles APIs, en particulier le serving de modèles ML/LLM.
<!-- #endregion -->

<!-- #region -->
### 1. Hello world
<!-- #endregion -->

<!-- #region -->
La « magie » de FastAPI tient dans les annotations : le type de retour, les types d'arguments, tout est exploité. `TestClient` (basé sur httpx) exerce l'app in-process, exactement comme `test_client` côté Flask.
<!-- #endregion -->

```python
import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, Path, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.testclient import TestClient

fa_hello = FastAPI(title="Hello API")


@fa_hello.get("/")
async def fa_root() -> dict[str, str]:
    """Route racine asynchrone."""
    return {"message": "Hello, FastAPI!"}


with TestClient(fa_hello) as c:
    r = c.get("/")
    show("GET /", r.status_code, r.json())
```

<!-- #region -->
### 2. Modèles Pydantic v2
<!-- #endregion -->

<!-- #region -->
On décrit la ressource avec des modèles Pydantic. Les contraintes (`Field(min_length=...)`, `ge`/`le`) sont **appliquées automatiquement** à l'entrée. La séparation `Create` / `Update` / `Out` est une bonne pratique : on ne valide pas les mêmes champs en création, en mise à jour partielle, et en sortie.
<!-- #endregion -->

```python
class BookBase(BaseModel):
    """Champs communs d'un livre (validés automatiquement)."""

    title: str = Field(min_length=1, max_length=200)
    author: str = Field(min_length=1, max_length=120)
    year: int | None = Field(default=None, ge=0, le=2100)


class BookCreate(BookBase):
    """Payload de création (identique à BookBase ici, séparé par convention)."""


class BookUpdate(BaseModel):
    """Payload de mise à jour partielle (tous les champs optionnels)."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    author: str | None = Field(default=None, min_length=1, max_length=120)
    year: int | None = Field(default=None, ge=0, le=2100)


class BookOut(BookBase):
    """Schéma de sortie (avec id) — pilote `response_model`."""

    id: int
```

<!-- #region -->
### 3. CRUD `books` (même ressource)
<!-- #endregion -->

<!-- #region -->
La même API que côté Flask, mais :

- Le **corps** est typé `payload: BookCreate` → validation et parsing automatiques (un payload invalide renvoie `422` **sans une ligne de code**).
- `response_model=...` filtre et valide la **sortie**.
- `status_code=...` fixe le code de succès.
- On déclare au passage une dépendance `pagination` (réutilisée par la liste, section 7) et l'auth OAuth2 (section 8).
<!-- #endregion -->

```python
SECRET_KEY = os.environ.get("APP_SECRET_KEY", "dev-secret-changeme")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_current_user(token: Annotated[str | None, Depends(oauth2_scheme)]) -> dict | None:
    """Dépendance d'auth : décode le JWT optionnel (401 si invalide)."""
    if token is None:
        return None
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


def pagination(skip: int = 0, limit: Annotated[int, Query(ge=1, le=100)] = 50) -> dict[str, int]:
    """Dépendance réutilisable de pagination (illustration DI)."""
    return {"skip": skip, "limit": limit}


fa_app = FastAPI(title="Books API", version="1.0.0", description="CRUD books — Pydantic v2 + DI + JWT.")
_FA_BOOKS: dict[int, dict[str, Any]] = {b["id"]: dict(b) for b in SEED_BOOKS}
_FA_NEXT: int = max(_FA_BOOKS) + 1


@fa_app.get("/books", response_model=list[BookOut])
async def fa_list(page: Annotated[dict, Depends(pagination)]) -> list[dict]:
    """Liste paginée via la dépendance `pagination` (DI)."""
    items = list(_FA_BOOKS.values())
    return items[page["skip"] : page["skip"] + page["limit"]]


@fa_app.get("/books/{book_id}", response_model=BookOut)
async def fa_get(book_id: Annotated[int, Path(ge=1)]) -> dict:
    """Récupère un livre (404 si absent)."""
    book = _FA_BOOKS.get(book_id)
    if not book:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Book {book_id} not found")
    return book


@fa_app.post("/books", response_model=BookOut, status_code=status.HTTP_201_CREATED)
async def fa_create(payload: BookCreate) -> dict:
    """Crée un livre (validation auto via le type `BookCreate`)."""
    global _FA_NEXT
    book = {"id": _FA_NEXT, **payload.model_dump()}
    _FA_BOOKS[_FA_NEXT] = book
    _FA_NEXT += 1
    return book


@fa_app.put("/books/{book_id}", response_model=BookOut)
async def fa_update(book_id: int, payload: BookUpdate) -> dict:
    """Mise à jour partielle (exclut les champs non fournis)."""
    if book_id not in _FA_BOOKS:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    _FA_BOOKS[book_id].update(payload.model_dump(exclude_none=True))
    return _FA_BOOKS[book_id]


@fa_app.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def fa_delete(book_id: int) -> None:
    """Supprime un livre (404 si absent)."""
    if book_id not in _FA_BOOKS:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    del _FA_BOOKS[book_id]


@fa_app.get("/items/{item_id}")
async def fa_items(
    item_id: Annotated[int, Path(ge=1)],
    sort: Annotated[str | None, Query(pattern="^(asc|desc)$")] = None,
) -> dict:
    """Démo validation path (`ge=1`) + query (`pattern=`, et non `regex=` déprécié)."""
    return {"item_id": item_id, "sort": sort}


@fa_app.post("/token")
async def fa_token(username: Annotated[str, Form()]) -> dict:
    """Émet un JWT (flow simplifié : pas de vérif mot de passe ici)."""
    return {
        "access_token": jwt.encode({"sub": username}, SECRET_KEY, algorithm="HS256"),
        "token_type": "bearer",
    }


@fa_app.get("/me")
async def fa_me(user: Annotated[dict | None, Depends(get_current_user)]) -> dict:
    """Route protégée : renvoie le payload du token (401 sans token)."""
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Auth required (Bearer token)")
    return user
```

<!-- #region -->
### 4. Démo CRUD exécutée
<!-- #endregion -->

<!-- #region -->
Même cycle qu'en Flask. Noter qu'un payload invalide (`title` vide) déclenche un `422` **automatique** — aucune validation manuelle écrite.
<!-- #endregion -->

```python
with TestClient(fa_app) as c:
    r = c.get("/books")
    show("GET /books", r.status_code, f"{len(r.json())} livres")
    r = c.post("/books", json={"title": "Neuromancer", "author": "William Gibson", "year": 1984})
    show("POST /books", r.status_code, r.json())
    fid = r.json()["id"]
    r = c.put(f"/books/{fid}", json={"year": 1985})
    show("PUT /books/{id}", r.status_code, r.json())
    r = c.delete(f"/books/{fid}")
    show("DELETE /books/{id}", r.status_code, "(no content)")
    r = c.post("/books", json={"title": "", "author": "X"})
    show("POST invalid", r.status_code, "422 (validation auto)")
    r = c.get("/books/9999")
    show("GET /books/9999", r.status_code, "404")
```

<!-- #region -->
### 5. Path / query / body validés
<!-- #endregion -->

<!-- #region -->
`Path(ge=1)` contraint le path param ; `Query(pattern="^(asc|desc)$")` contraint la query par une regex. **Important (2026)** : le paramètre s'appelle `pattern`, l'ancien `regex=` est **déprécié**. Toute violation renvoie un `422` documenté.
<!-- #endregion -->

```python
with TestClient(fa_app) as c:
    r = c.get("/items/3?sort=asc")
    show("items valid", r.status_code, r.json())
    r = c.get("/items/3?sort=bogus")
    show("items invalid", r.status_code, "422 (pattern non respecté)")
```

<!-- #region -->
### 6. Async & le footgun de l'event loop
<!-- #endregion -->

<!-- #region -->
FastAPI tourne sur un **event loop** asyncio. Règle d'or :

- Une opération **I/O bound** (DB, HTTP, fichier) doit être `await`-ée dans une route `async def`.
- Un calcul **CPU bound** (ou tout appel bloquant synchrone) dans une route `async def` **bloque l'event loop** et fait chuter le débit — c'est le **footgun n°1** en production.

Deux parades : déclarer la route en **`def` normal** (FastAPI l'exécute dans un threadpool), ou **offloader** explicitement via `asyncio.to_thread`.
<!-- #endregion -->

```python
fa_async = FastAPI()


@fa_async.get("/io")
async def io_bound() -> dict:
    """I/O bound correct : on `await` (ne bloque pas l'event loop)."""
    await asyncio.sleep(0)
    return {"kind": "io", "ok": True}


@fa_async.get("/cpu")
def cpu_bound() -> dict:
    """CPU bound : route `def` (synchrone) -> FastAPI l'exécute dans un threadpool."""
    return {"kind": "cpu", "total": sum(range(100_000))}


@fa_async.get("/offload")
async def offload() -> dict:
    """Offload explicite d'un calcul bloquant via `asyncio.to_thread`."""
    total = await asyncio.to_thread(lambda: sum(range(100_000)))
    return {"kind": "offload", "total": total}


with TestClient(fa_async) as c:
    for route in ("/io", "/cpu", "/offload"):
        r = c.get(route)
        show(f"GET {route}", r.status_code, r.json())
```

<!-- #region -->
### 7. Dependency Injection
<!-- #endregion -->

<!-- #region -->
**LA** fonctionnalité signature de FastAPI : on déclare des dépendances (params communs, session DB, utilisateur courant) comme des paramètres typés via `Depends(...)`. Elles sont **composables**, **réutilisables** et **surchargables en test**. Ici, la dépendance `pagination` est injectée dans la liste des livres.
<!-- #endregion -->

```python
with TestClient(fa_app) as c:
    r = c.get("/books?skip=1&limit=2")
    show("DI pagination", r.status_code, f"{len(r.json())} livres (skip=1, limit=2)")
```

<!-- #region -->
### 8. Authentification OAuth2 / JWT
<!-- #endregion -->

<!-- #region -->
`OAuth2PasswordBearer` extrait le token `Authorization: Bearer ...`. La dépendance `get_current_user` le décode (PyJWT) et protège `/me`. On émet d'abord un token via `/token`, puis on appelle la route protégée avec, et sans, ce token.
<!-- #endregion -->

```python
with TestClient(fa_app) as c:
    r = c.get("/me")
    show("/me sans token", r.status_code, r.json())
    tok = c.post("/token", data={"username": "alice"}).json()["access_token"]
    r = c.get("/me", headers={"Authorization": f"Bearer {tok}"})
    show("/me avec token", r.status_code, r.json())
```

<!-- #region -->
### 9. OpenAPI / Swagger auto
<!-- #endregion -->

<!-- #region -->
Sans rien écrire, FastAPI sert `/docs` (Swagger UI interactif), `/redoc` (ReDoc) et `/openapi.json` (spec OpenAPI 3.1). On inspecte ici le schéma généré : ses chemins et les modèles Pydantic qui en sont déduits. Cette spec permet aussi de **générer des clients** dans de nombreux langages.
<!-- #endregion -->

```python
schema = fa_app.openapi()
print("OpenAPI version :", schema["openapi"])
print("paths   :", sorted(schema["paths"].keys()))
print("schemas :", sorted(schema["components"]["schemas"].keys()))
```

<!-- #region -->
### 10. Lifespan (cycle de vie)
<!-- #endregion -->

<!-- #region -->
Pour initialiser/fermer des ressources (pool DB, client HTTP, modèle ML chargé en mémoire) au démarrage et à l'arrêt, on utilise un **`lifespan`** : un context manager asynchrone passé à `FastAPI(lifespan=...)`. Il **remplace** les anciens `@app.on_event("startup"/"shutdown")`, **dépréciés** (et silencieusement ignorés si on fournit un `lifespan`). Le bloc `with TestClient(...)` déclenche le startup à l'entrée et le shutdown à la sortie.
<!-- #endregion -->

```python
_lifecycle: dict[str, bool] = {"started": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie (remplace `@app.on_event`, déprécié)."""
    _lifecycle["started"] = True  # startup
    yield
    _lifecycle["started"] = False  # shutdown


fa_lifespan = FastAPI(lifespan=lifespan)


@fa_lifespan.get("/status")
async def fa_status() -> dict:
    """Expose l'état initialisé par le lifespan."""
    return {"started": _lifecycle["started"]}


with TestClient(fa_lifespan) as c:  # le `with` déclenche startup puis shutdown
    r = c.get("/status")
    show("GET /status (dans le lifespan)", r.status_code, r.json())
print("après le bloc with, shutdown joué ->", not _lifecycle["started"])
```

<!-- #region -->
### 11. Différents types de payload
<!-- #endregion -->

<!-- #region -->
Comme Flask, FastAPI gère form, fichiers et streaming — mais avec des types dédiés (`Form`, `UploadFile`, `StreamingResponse`).
<!-- #endregion -->

<!-- #region -->
#### 11.1 Form data
<!-- #endregion -->

<!-- #region -->
`Form()` lit un champ de formulaire. (Nécessite `python-multipart`, fourni par `fastapi[standard]`.)
<!-- #endregion -->

```python
fa_form = FastAPI()


@fa_form.post("/login")
async def fa_login(username: Annotated[str, Form()], password: Annotated[str, Form()]) -> dict:
    """Lecture d'un payload form (nécessite python-multipart)."""
    return {"user": username}


with TestClient(fa_form) as c:
    r = c.post("/login", data={"username": "bob", "password": "x"})
    show("POST form", r.status_code, r.json())
```

<!-- #region -->
#### 11.2 File upload
<!-- #endregion -->

<!-- #region -->
`UploadFile` expose un fichier en streaming avec lecture asynchrone (`await file.read()`), son nom et son type MIME.
<!-- #endregion -->

```python
fa_upload = FastAPI()


@fa_upload.post("/upload")
async def fa_upload_file(file: Annotated[UploadFile, File()]) -> dict:
    """Réception d'un fichier (lecture async)."""
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents), "type": file.content_type}


with TestClient(fa_upload) as c:
    r = c.post("/upload", files={"file": ("data.txt", b"fastapi upload", "text/plain")})
    show("POST upload", r.status_code, r.json())
```

<!-- #region -->
#### 11.3 Streaming & SSE
<!-- #endregion -->

<!-- #region -->
`StreamingResponse` enveloppe un générateur (sync ou async). C'est la base du **SSE** (Server-Sent Events, `media_type="text/event-stream"`), très utilisé en 2026 pour **streamer les réponses de LLM** token par token.
<!-- #endregion -->

```python
fa_stream = FastAPI()


@fa_stream.get("/stream/{n}")
async def fa_stream_numbers(n: int) -> StreamingResponse:
    """Réponse en streaming (générateur async)."""

    async def gen():
        for i in range(n):
            yield f"{i}\n"

    return StreamingResponse(gen(), media_type="text/plain")


with TestClient(fa_stream) as c:
    r = c.get("/stream/4")
    show("GET /stream/4", r.status_code, repr(r.text))
```

<!-- #region -->
### 12. Tests avec `TestClient`
<!-- #endregion -->

<!-- #region -->
`TestClient` s'utilise comme `requests`/`httpx`. Pour tester du **vrai async** (concurrence, websockets), on passe à `httpx.AsyncClient` + `pytest-asyncio`. Ici, quelques assertions synchrones.
<!-- #endregion -->

```python
with TestClient(fa_app) as c:
    assert c.post("/books", json={"title": "T2", "author": "A2"}).status_code == 201
    assert c.get("/books/9999").status_code == 404
    print("Tests FastAPI OK")
```

<!-- #region -->
### 13. Mise en production
<!-- #endregion -->

<!-- #region -->
FastAPI est une app **ASGI**, servie par **Uvicorn**. En production on lance plusieurs workers via Gunicorn :

```bash
# Dev (un process, auto-reload)
uvicorn apps.fastapi_api.main:app --reload --port 8000

# Prod (N workers Uvicorn pilotés par Gunicorn)
gunicorn apps.fastapi_api.main:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000
```

Comme pour Flask : conteneur Docker + reverse proxy devant pour TLS et rate-limiting.
<!-- #endregion -->

<!-- #region -->
## Partie C — Flask vs FastAPI
<!-- #endregion -->

<!-- #region -->
Les deux APIs ci-dessus exposent **exactement la même ressource** `books`. On peut donc comparer à conditions strictement égales.
<!-- #endregion -->

<!-- #region -->
### 1. La même route, deux écritures
<!-- #endregion -->

<!-- #region -->
`GET /books/2` renvoie le même livre des deux côtés. La différence est dans le **comment** : Flask construit la réponse à la main (`jsonify`, 404 explicite) ; FastAPI déduit la validation de sortie de `response_model=BookOut` et le typage.
<!-- #endregion -->

```python
with books_app.test_client() as fc, TestClient(fa_app) as ac:
    rf = fc.get("/books/2")
    ra = ac.get("/books/2")
    show("Flask   GET /books/2", rf.status_code, rf.get_json())
    show("FastAPI GET /books/2", ra.status_code, ra.json())
```

<!-- #region -->
### 2. Tableau comparatif
<!-- #endregion -->

<!-- #region -->
| Critère | Flask | FastAPI |
|---|---|---|
| **Protocole** | WSGI (synchrone) | ASGI (async natif) |
| **Async** | partiel (Flask 2+ accepte `async def`, mais reste WSGI) | natif, de bout en bout |
| **Validation** | manuelle / Pydantic / marshmallow | **native** via les types (Pydantic v2) |
| **OpenAPI / Swagger** | non natif (`flask-smorest`, `flasgger`) | **généré automatiquement** (`/docs`, `/openapi.json`) |
| **Perf (I/O simple)** | ~2 000–3 000 req/s | ~15 000–20 000 req/s (≈ 6–8×) |
| **Perf (DB-bound)** | référence | ≈ 2–4× (l'écart se resserre quand la DB domine) |
| **Courbe d'apprentissage** | très douce | douce, mais async + typage à maîtriser |
| **Écosystème** | très large, mature (Flask-*) | jeune mais dense, très actif |
| **Injection de dépendances** | non (extensions) | **native** (`Depends`) |

> Les chiffres de perf sont des ordres de grandeur (benchmarks 2026, endpoints I/O bound) — à relativiser selon la charge réelle, surtout si une base de données est le goulot d'étranglement.
<!-- #endregion -->

<!-- #region -->
### 3. Quand choisir quoi en 2026
<!-- #endregion -->

<!-- #region -->
**Choisir Flask** quand : on maintient du legacy, on a besoin d'un microservice de 2–3 routes, on s'appuie sur un plugin Flask spécifique (Flask-Admin, Flask-Login…), ou l'équipe maîtrise déjà Flask sur un projet court.

**Choisir FastAPI** quand : nouveau projet, latence/débit critiques, besoin de validation Pydantic + OpenAPI automatique, I/O concurrent (websockets, SSE), ou **serving de modèles ML/LLM** (streaming de tokens via SSE).

En pratique, ce n'est plus une comparaison déséquilibrée : c'est un arbitrage contextuel (équipe, contraintes, tolérance à l'outillage récent).
<!-- #endregion -->

<!-- #region -->
### 4. Aller plus loin — versions serveur lançables
<!-- #endregion -->

<!-- #region -->
Le notebook exécute tout in-process. Pour manipuler ces APIs comme de **vrais serveurs** (curl, navigateur, Swagger UI), des versions autonomes existent dans le dépôt :

```bash
# Flask
uv run flask --app apps/flask_api/app.py run --port 5000
curl http://localhost:5000/health

# FastAPI (+ Swagger UI sur /docs)
uv run uvicorn apps.fastapi_api.main:app --reload --port 8000
# puis ouvrir http://localhost:8000/docs
```
<!-- #endregion -->

<!-- #region -->
### 5. Sources
<!-- #endregion -->

<!-- #region -->
- [Flask — documentation officielle](https://flask.palletsprojects.com/)
- [FastAPI — documentation officielle](https://fastapi.tiangolo.com/) · [Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)
- [Pydantic v2 — documentation](https://docs.pydantic.dev/latest/)
- [Starlette](https://www.starlette.io/) (base ASGI de FastAPI) · [Uvicorn](https://www.uvicorn.org/) · [Gunicorn](https://gunicorn.org/)
- [Werkzeug](https://werkzeug.palletsprojects.com/) (base WSGI de Flask)
- Comparatifs 2026 : [tech-insider](https://tech-insider.org/fastapi-vs-flask-2026/), [clickittech](https://www.clickittech.com/ai/fastapi-vs-flask-for-production-ai-apis/)
- Apps lançables : `apps/flask_api/app.py`, `apps/fastapi_api/main.py`.
<!-- #endregion -->
