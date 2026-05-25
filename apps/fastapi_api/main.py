"""FastAPI — démo CRUD async complète, exécutable.

Lance :
    uv run uvicorn apps.fastapi_api.main:app --reload --port 8000

Swagger UI :
    http://localhost:8000/docs

Test :
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/users -H "Content-Type: application/json" \\
         -d '{"name":"Alice","email":"alice@example.com","age":30,"password":"pass1234"}'
    curl http://localhost:8000/users
"""

from __future__ import annotations

import os
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    secret_key: str = "dev-secret-changeme"
    api_key: str = "dev-key"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")


settings = Settings()


# ---------- Schemas Pydantic ----------
class UserBase(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    email: EmailStr
    age: int = Field(ge=0, le=150)


class UserCreate(UserBase):
    password: str = Field(min_length=8)


class UserUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=50)
    email: EmailStr | None = None
    age: int | None = Field(None, ge=0, le=150)


class UserOut(UserBase):
    id: int


# ---------- App ----------
app = FastAPI(
    title="Demo FastAPI",
    version="1.0.0",
    description="API démo CRUD avec Pydantic v2 + DI + JWT.",
)

USERS: dict[int, dict] = {}
NEXT_ID: int = 1

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


# ---------- Dependencies ----------
def get_current_user(token: Annotated[str | None, Depends(oauth2_scheme)]) -> dict | None:
    """Vérifie le JWT optionnel et renvoie le payload."""
    if token is None:
        return None
    try:
        return jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "demo-fastapi"}


@app.get("/users", response_model=list[UserOut])
async def list_users(skip: int = 0, limit: int = 100):
    return list(USERS.values())[skip : skip + limit]


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
async def update_user(user_id: int, payload: UserUpdate):
    if user_id not in USERS:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    USERS[user_id].update(payload.model_dump(exclude_none=True))
    return USERS[user_id]


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    if user_id not in USERS:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    del USERS[user_id]


@app.get("/me")
async def get_me(user: Annotated[dict | None, Depends(get_current_user)] = None):
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Auth required (Bearer token)")
    return user
