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
# 🌶️ Flask — API REST en Python
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel complet** sur **Flask** : framework Python micro pour construire des APIs REST. Reste pertinent en 2026 pour les **petites APIs**, l'**enseignement**, ou les projets legacy.

> Pour la version **async / type-safe / production** moderne, voir **`FastAPI_API`**. Pour le comparatif Flask vs FastAPI, voir la section 8 ci-dessous (ou notebook séparé selon préférence).

Couverture :

1. **Hello World** — routing minimal.
2. **CRUD complet** sur ressource (`users`) — GET/POST/PUT/DELETE.
3. **JSON I/O** avec `request.get_json()` et `jsonify`.
4. **Path / query / body parameters**.
5. **Validation** — manuelle ou via `marshmallow` / `pydantic`.
6. **Erreurs HTTP propres** (400, 404, 422, 500).
7. **Auth** (clé API, JWT).
8. **Différents types de payload** : JSON, form, file upload, streaming.
9. **Tests** avec `test_client`.
10. **Déploiement** (Gunicorn, Docker).
<!-- #endregion -->

<!-- #region -->
## 1. Hello World
<!-- #endregion -->

<!-- #region -->
```python
"""
# pip install flask
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, world!", "status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
"""
```

Test :
```bash
curl http://localhost:5000/
# {"message": "Hello, world!", "status": "ok"}
```
<!-- #endregion -->

<!-- #region -->
## 2. CRUD complet sur `users`
<!-- #endregion -->

<!-- #region -->
```python
"""
from flask import Flask, jsonify, request
from werkzeug.exceptions import NotFound

app = Flask(__name__)

# Mock storage en mémoire (à remplacer par SQL en prod)
USERS = {}
NEXT_ID = 1


@app.route("/users", methods=["GET"])
def list_users():
    return jsonify(list(USERS.values()))


@app.route("/users/<int:user_id>", methods=["GET"])
def get_user(user_id: int):
    user = USERS.get(user_id)
    if user is None:
        raise NotFound(f"User {user_id} not found")
    return jsonify(user)


@app.route("/users", methods=["POST"])
def create_user():
    global NEXT_ID
    data = request.get_json(silent=True) or {}
    if not data.get("name"):
        return jsonify({"error": "name required"}), 400
    user = {"id": NEXT_ID, "name": data["name"], "age": data.get("age", 0)}
    USERS[NEXT_ID] = user
    NEXT_ID += 1
    return jsonify(user), 201


@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id: int):
    if user_id not in USERS:
        raise NotFound(f"User {user_id} not found")
    data = request.get_json(silent=True) or {}
    USERS[user_id].update(data)
    return jsonify(USERS[user_id])


@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id: int):
    if user_id not in USERS:
        raise NotFound(f"User {user_id} not found")
    del USERS[user_id]
    return "", 204
"""
```
<!-- #endregion -->

<!-- #region -->
## 3. JSON I/O
<!-- #endregion -->

<!-- #region -->
- `request.get_json(silent=True)` : parse le body JSON, retourne `None` si pas du JSON valide (au lieu de raise).
- `jsonify(...)` : sérialise un dict/list en `Response` avec `Content-Type: application/json`.
- `request.args.get("key", default)` : query string `?key=value`.
- `request.headers.get("X-API-Key")` : headers HTTP.
- `request.files["file"]` : pour file upload.
<!-- #endregion -->

<!-- #region -->
## 4. Validation avec Pydantic (2026)
<!-- #endregion -->

<!-- #region -->
```python
"""
from pydantic import BaseModel, Field, ValidationError

class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    age: int = Field(ge=0, le=150)
    email: str | None = None


@app.route("/users", methods=["POST"])
def create_user_validated():
    try:
        user_data = UserCreate.model_validate(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"errors": e.errors()}), 422
    # ... use user_data.model_dump()
"""
```

Avec Flask, Pydantic n'est pas natif (contrairement à FastAPI). On peut aussi utiliser **marshmallow** (l'historique) ou **flask-pydantic**.
<!-- #endregion -->

<!-- #region -->
## 5. Gestion d'erreurs centralisée
<!-- #endregion -->

<!-- #region -->
```python
"""
from werkzeug.exceptions import HTTPException

@app.errorhandler(HTTPException)
def handle_http(e: HTTPException):
    return jsonify({"error": e.description, "code": e.code}), e.code

@app.errorhandler(Exception)
def handle_unexpected(e: Exception):
    app.logger.exception(e)
    return jsonify({"error": "Internal server error"}), 500
"""
```
<!-- #endregion -->

<!-- #region -->
## 6. Auth simple (clé API en header)
<!-- #endregion -->

<!-- #region -->
```python
"""
from functools import wraps
import os

API_KEY = os.environ.get("API_KEY", "dev-key")

def require_api_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper

@app.route("/admin/users", methods=["GET"])
@require_api_key
def admin_users():
    return jsonify(list(USERS.values()))
"""
```

Pour du **JWT** : `flask-jwt-extended`.
<!-- #endregion -->

<!-- #region -->
## 7. Différents types de payload
<!-- #endregion -->

<!-- #region -->
### 7.1 Form data (`application/x-www-form-urlencoded`)
<!-- #endregion -->

<!-- #region -->
```python
"""
@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    return jsonify({"user": username})
"""
```
<!-- #endregion -->

<!-- #region -->
### 7.2 File upload (`multipart/form-data`)
<!-- #endregion -->

<!-- #region -->
```python
"""
import os
from werkzeug.utils import secure_filename

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    safe = secure_filename(f.filename)
    f.save(os.path.join(UPLOAD_DIR, safe))
    return jsonify({"filename": safe, "size": os.path.getsize(os.path.join(UPLOAD_DIR, safe))})
"""
```
<!-- #endregion -->

<!-- #region -->
### 7.3 Streaming (gros fichiers download)
<!-- #endregion -->

<!-- #region -->
```python
"""
from flask import stream_with_context, Response

@app.route("/stream/<int:n>")
def stream_numbers(n: int):
    def generate():
        for i in range(n):
            yield f"{i}\\n"
    return Response(stream_with_context(generate()), mimetype="text/plain")
"""
```
<!-- #endregion -->

<!-- #region -->
## 8. Tests
<!-- #endregion -->

<!-- #region -->
```python
"""
import pytest
from app import app  # le module Flask

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_create_user(client):
    response = client.post("/users", json={"name": "Alice", "age": 30})
    assert response.status_code == 201
    assert response.json["name"] == "Alice"

def test_get_user_not_found(client):
    response = client.get("/users/9999")
    assert response.status_code == 404
"""
```
<!-- #endregion -->

<!-- #region -->
## 9. Déploiement production
<!-- #endregion -->

<!-- #region -->
**Jamais** `app.run()` en prod (serveur dev mono-thread).

- **Gunicorn** (WSGI) : `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- **uWSGI** : alternative historique.
- **Docker** : containerize avec gunicorn + nginx en reverse proxy.
- Pour de la **vraie async** : Flask 2.x supporte les coroutines (`async def`) mais reste WSGI sous le capot. Pour de la vraie perf async, basculer sur FastAPI.
<!-- #endregion -->

<!-- #region -->
## 10. OpenAPI / Swagger
<!-- #endregion -->

<!-- #region -->
Flask **ne génère pas l'OpenAPI** automatiquement. Solutions :

- **flask-smorest** — extension qui auto-génère OpenAPI à partir des routes et schémas marshmallow.
- **flasgger** — annotations docstrings → OpenAPI.
- **Faire manuellement** un endpoint `/openapi.json` (peu maintenable).

> C'est l'un des **gros défauts de Flask** vs FastAPI, qui génère l'OpenAPI nativement à partir du typage Python.
<!-- #endregion -->

<!-- #region -->
## 11. Flask vs FastAPI — quand choisir Flask en 2026
<!-- #endregion -->

<!-- #region -->
**Choisir Flask** quand :

- Code legacy à maintenir.
- API très simple (microservice de 2-3 routes).
- L'écosystème de plugins Flask est requis (Flask-Login, Flask-Admin, Flask-Caching).
- Tu maîtrises Flask et le projet est court terme.

**Préférer FastAPI** quand :

- Nouveau projet.
- API à haute performance / latence critique.
- Validation Pydantic + OpenAPI auto = besoin.
- WebSocket / streaming async.
- Long-poll / SSE.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [Flask — docs officielles](https://flask.palletsprojects.com/)
- [Flask Mega-Tutorial — Miguel Grinberg](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
- [Gunicorn docs](https://gunicorn.org/)
- [werkzeug docs](https://werkzeug.palletsprojects.com/) (base de Flask)
- Notebooks liés : `FastAPI_API`, `Streamlit_brique`.
<!-- #endregion -->
