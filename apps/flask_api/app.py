"""Flask API — démo CRUD complète, exécutable.

Lance :
    uv run flask --app apps/flask_api/app.py run --port 5000

Test :
    curl http://localhost:5000/health
    curl -X POST http://localhost:5000/users -H "Content-Type: application/json" -d '{"name":"Alice","age":30}'
    curl http://localhost:5000/users
    curl http://localhost:5000/users/1
    curl -X PUT http://localhost:5000/users/1 -H "Content-Type: application/json" -d '{"age":31}'
    curl -X DELETE http://localhost:5000/users/1
"""

from __future__ import annotations

import os
from functools import wraps

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, NotFound

app = Flask(__name__)

# Mock storage (à remplacer par DB en prod)
USERS: dict[int, dict] = {}
NEXT_ID: int = 1

API_KEY = os.environ.get("FLASK_API_KEY", "dev-key")


# ---------- Auth ----------
def require_api_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


# ---------- Routes ----------
@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "demo-flask"})


@app.get("/users")
def list_users():
    return jsonify(list(USERS.values()))


@app.get("/users/<int:user_id>")
def get_user(user_id: int):
    user = USERS.get(user_id)
    if user is None:
        raise NotFound(f"User {user_id} not found")
    return jsonify(user)


@app.post("/users")
def create_user():
    global NEXT_ID
    data = request.get_json(silent=True) or {}
    if not data.get("name"):
        return jsonify({"error": "name required"}), 400
    age = data.get("age", 0)
    if not isinstance(age, int) or age < 0:
        return jsonify({"error": "age must be int >= 0"}), 422
    user = {"id": NEXT_ID, "name": data["name"], "age": age}
    USERS[NEXT_ID] = user
    NEXT_ID += 1
    return jsonify(user), 201


@app.put("/users/<int:user_id>")
def update_user(user_id: int):
    if user_id not in USERS:
        raise NotFound(f"User {user_id} not found")
    data = request.get_json(silent=True) or {}
    USERS[user_id].update(data)
    return jsonify(USERS[user_id])


@app.delete("/users/<int:user_id>")
def delete_user(user_id: int):
    if user_id not in USERS:
        raise NotFound(f"User {user_id} not found")
    del USERS[user_id]
    return "", 204


@app.get("/admin/users")
@require_api_key
def admin_users():
    return jsonify({"count": len(USERS), "users": list(USERS.values())})


# ---------- Error handlers ----------
@app.errorhandler(HTTPException)
def handle_http(e: HTTPException):
    return jsonify({"error": e.description, "code": e.code}), e.code


@app.errorhandler(Exception)
def handle_unexpected(e: Exception):  # noqa: ARG001
    app.logger.exception("unexpected")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
