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
# 🚀 FastAPI — API REST async moderne
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel complet** sur **FastAPI** (Sebastian Ramírez, 2018+) — framework Python **async**, **type-safe** (Pydantic), avec **OpenAPI / Swagger UI** générés automatiquement. Devenu le **standard 2026** pour les APIs Python.

> Pour Flask et le comparatif Flask vs FastAPI, voir **`Flask_API`**.

Couverture :

1. **Hello World** — la magie du type hint.
2. **CRUD** complet (`users`) avec **Pydantic v2**.
3. **Path / query / body** avec validation auto.
4. **Async** I/O bound (`async def`).
5. **Différents types de payload** : JSON, form, file upload, streaming.
6. **Dependency Injection** (DB sessions, auth).
7. **Auth** — OAuth2 password flow / JWT.
8. **OpenAPI / Swagger UI** auto.
9. **Tests** avec `httpx` + `pytest`.
10. **Déploiement** : Uvicorn / Gunicorn + Uvicorn workers, Docker.
11. **Bonnes pratiques 2026** : structure, Pydantic settings, monitoring.
<!-- #endregion -->

<!-- #region -->
## 1. Hello World
<!-- #endregion -->

<!-- #region -->
```python
"""
# pip install fastapi uvicorn[standard]
from fastapi import FastAPI

app = FastAPI(title="Demo API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

# Lancement : uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Auto-doc : http://localhost:8000/docs  ← Swagger UI
# Schema : http://localhost:8000/openapi.json
"""
```
<!-- #endregion -->

<!-- #region -->
## 2. CRUD complet avec Pydantic
<!-- #endregion -->

<!-- #region -->
```python
"""
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, EmailStr

app = FastAPI()


class UserBase(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    email: EmailStr
    age: int = Field(ge=0, le=150)


class UserCreate(UserBase):
    password: str = Field(min_length=8)


class UserOut(UserBase):
    id: int

    model_config = {"from_attributes": True}   # = ORM mode (Pydantic v2)


# Mock storage
USERS: dict[int, dict] = {}
NEXT_ID = 1


@app.get("/users", response_model=list[UserOut])
async def list_users():
    return list(USERS.values())


@app.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: int):
    user = USERS.get(user_id)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"User {user_id} not found")
    return user


@app.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(payload: UserCreate):
    global NEXT_ID
    user = {**payload.model_dump(exclude={"password"}), "id": NEXT_ID}
    USERS[NEXT_ID] = user
    NEXT_ID += 1
    return user


@app.put("/users/{user_id}", response_model=UserOut)
async def update_user(user_id: int, payload: UserBase):
    if user_id not in USERS:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    USERS[user_id].update(payload.model_dump())
    return USERS[user_id]


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    if user_id not in USERS:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    del USERS[user_id]
"""
```

**La magie** :

- Le typage des arguments fait **automatiquement** la validation + le parsing.
- `response_model` fait la **sérialisation et la validation de la réponse**.
- L'OpenAPI est généré automatiquement avec tous les schémas, codes d'erreur, descriptions.
<!-- #endregion -->

<!-- #region -->
## 3. Path / query / body
<!-- #endregion -->

<!-- #region -->
```python
"""
from fastapi import Query, Path, Body

@app.get("/items/{item_id}")
async def get_item(
    item_id: int = Path(ge=1, description="ID positif"),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    sort: str | None = Query(None, regex="^(asc|desc)$"),
):
    return {"item_id": item_id, "skip": skip, "limit": limit, "sort": sort}
"""
```

Ces validations sont **automatiquement** documentées dans Swagger.
<!-- #endregion -->

<!-- #region -->
## 4. Async I/O bound
<!-- #endregion -->

<!-- #region -->
**FastAPI tourne sur Starlette + Uvicorn** → vrai async. Toute opération **I/O bound** (DB, HTTP, fichier) **doit** être awaited :

```python
"""
import httpx

@app.get("/proxy")
async def proxy_call():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.example.com/data")
        return r.json()
"""
```

**Important** : si une route `async def` contient du **CPU-bound** lourd (ML inference, calcul), elle bloque l'event loop. Solutions :

- **Modifier la fonction en `def` normal** → FastAPI la lance dans un thread pool.
- Utiliser `run_in_threadpool` ou `asyncio.to_thread`.
- **Task queue** (Celery, Dramatiq) pour offload.
<!-- #endregion -->

<!-- #region -->
## 5. Différents types de payload
<!-- #endregion -->

<!-- #region -->
### 5.1 Form data
<!-- #endregion -->

<!-- #region -->
```python
"""
from fastapi import Form

@app.post("/login")
async def login(username: str = Form(), password: str = Form()):
    return {"user": username}
"""
```
<!-- #endregion -->

<!-- #region -->
### 5.2 File upload
<!-- #endregion -->

<!-- #region -->
```python
"""
from fastapi import UploadFile, File

@app.post("/upload")
async def upload(file: UploadFile = File()):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents), "type": file.content_type}
"""
```
<!-- #endregion -->

<!-- #region -->
### 5.3 Streaming response
<!-- #endregion -->

<!-- #region -->
```python
"""
from fastapi.responses import StreamingResponse

@app.get("/stream/{n}")
async def stream_numbers(n: int):
    async def generate():
        for i in range(n):
            yield f"{i}\\n"
    return StreamingResponse(generate(), media_type="text/plain")
"""
```

Pour du **SSE** (Server-Sent Events, populaire en 2026 pour streaming LLM responses) :

```python
"""
from fastapi.responses import StreamingResponse

@app.get("/llm-stream")
async def llm_stream():
    async def stream():
        for token in llm_generate_tokens():
            yield f"data: {token}\\n\\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
"""
```
<!-- #endregion -->

<!-- #region -->
## 6. Dependency Injection
<!-- #endregion -->

<!-- #region -->
**LA feature** qui distingue FastAPI : tu déclares des **dépendances** (DB session, current user, config) comme des paramètres typés.

```python
"""
from fastapi import Depends
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()
"""
```

Les dépendances sont **cachées par scope** (request, application), **testables** (override en test), **composables**.
<!-- #endregion -->

<!-- #region -->
## 7. Auth — OAuth2 password flow + JWT
<!-- #endregion -->

<!-- #region -->
```python
"""
from fastapi.security import OAuth2PasswordBearer
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET = "secret-key-changeme"

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")


@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return user
"""
```
<!-- #endregion -->

<!-- #region -->
## 8. OpenAPI / Swagger UI
<!-- #endregion -->

<!-- #region -->
**Out of the box**, FastAPI sert :

- **`/docs`** : Swagger UI interactif.
- **`/redoc`** : ReDoc (alternative UI).
- **`/openapi.json`** : spec OpenAPI 3.x JSON brut.

Permet :

- Tester l'API depuis le navigateur.
- Générer des **clients** dans plein de langages (`openapi-generator`).
- Documenter automatiquement pour les consommateurs.

**Personnaliser** :

```python
"""
app = FastAPI(
    title="My API", version="1.0.0",
    description="API de gestion des trucs",
    contact={"name": "Team", "email": "team@x.com"},
    license_info={"name": "MIT"},
)
"""
```
<!-- #endregion -->

<!-- #region -->
## 9. Tests
<!-- #endregion -->

<!-- #region -->
```python
"""
# pip install httpx pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_create_user():
    r = client.post("/users", json={
        "name": "Alice", "email": "a@x.com", "age": 30, "password": "pass1234"
    })
    assert r.status_code == 201
    assert r.json()["name"] == "Alice"
    assert "password" not in r.json()   # n'est pas dans UserOut

def test_get_user_not_found():
    r = client.get("/users/9999")
    assert r.status_code == 404


# Pour vraiment async :
# pip install pytest-asyncio
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_async():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/")
        assert r.status_code == 200
"""
```
<!-- #endregion -->

<!-- #region -->
## 10. Déploiement production
<!-- #endregion -->

<!-- #region -->
**Standard 2026** :

```bash
# Single process async
uvicorn app:app --host 0.0.0.0 --port 8000

# Multi-process (recommandé prod) — gunicorn lance N workers Uvicorn
gunicorn app:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000

# Docker
# FROM python:3.12-slim
# WORKDIR /app
# COPY . .
# RUN pip install -r requirements.txt
# CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000"]
```

**Reverse proxy** (nginx, Caddy, Traefik) devant pour TLS, rate limiting, static files.
<!-- #endregion -->

<!-- #region -->
## 11. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### 11.1 Structure projet
<!-- #endregion -->

<!-- #region -->
```
app/
├── main.py                  # FastAPI app + middleware
├── api/
│   └── v1/
│       ├── users.py         # APIRouter
│       ├── items.py
│       └── auth.py
├── core/
│   ├── config.py            # Pydantic Settings
│   └── security.py          # JWT, hashing
├── models/                   # SQLAlchemy models
├── schemas/                  # Pydantic schemas
├── crud/                     # DB operations
├── deps.py                   # Dependencies réutilisables
└── tests/
```
<!-- #endregion -->

<!-- #region -->
### 11.2 Configuration avec Pydantic Settings
<!-- #endregion -->

<!-- #region -->
```python
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")


settings = Settings()
"""
```

→ Validation auto au démarrage. Pas de typo silencieuse.
<!-- #endregion -->

<!-- #region -->
### 11.3 Monitoring
<!-- #endregion -->

<!-- #region -->
- **Prometheus** : `prometheus-fastapi-instrumentator` pour exporter les métriques.
- **OpenTelemetry** : traces distribuées (`opentelemetry-instrumentation-fastapi`).
- **Sentry** : capture des exceptions.
- **structlog** : logging structuré JSON.
<!-- #endregion -->

<!-- #region -->
### 11.4 Rate limiting
<!-- #endregion -->

<!-- #region -->
- **`slowapi`** : extension `flask-limiter`-like pour FastAPI.
- Ou côté reverse proxy (nginx limit_req).
<!-- #endregion -->

<!-- #region -->
### 11.5 Background tasks
<!-- #endregion -->

<!-- #region -->
Pour des tâches courtes après response : `BackgroundTasks` natif FastAPI.
Pour des tâches longues / planifiées : **Celery** (avec Redis/RabbitMQ) ou **Dramatiq**.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [FastAPI — docs officielles](https://fastapi.tiangolo.com/)
- [Pydantic v2 — docs](https://docs.pydantic.dev/latest/)
- [Starlette — base async de FastAPI](https://www.starlette.io/)
- [Uvicorn — server ASGI](https://www.uvicorn.org/)
- [Full Stack FastAPI Template (officiel)](https://github.com/fastapi/full-stack-fastapi-template)
- Notebooks liés : `Flask_API`, `Streamlit_brique`, `ML_MLFlow_Bench`.
<!-- #endregion -->
