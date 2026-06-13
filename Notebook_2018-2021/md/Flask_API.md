---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    name: python3
---

```python id="qQEl7uUw50Ol"
!pip install flask
```

```python id="Loix1K8v43kb"
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# Données simulées
books = [
    {"id": 1, "title": "Le Petit Prince", "author": "Antoine de Saint-Exupéry"},
    {"id": 2, "title": "1984", "author": "George Orwell"},
    {"id": 3, "title": "L'Étranger", "author": "Albert Camus"}
]

# Génère un nouvel ID unique pour un nouveau livre
def get_next_id():
    return max(book["id"] for book in books) + 1 if books else 1

# Récupère tous les livres
@app.route('/books', methods=['GET'])
def get_books():
    return jsonify({"books": books})

# Récupère un livre par son ID
@app.route('/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = next((book for book in books if book["id"] == book_id), None)
    if not book:
        abort(404, description="Book not found")
    return jsonify(book)

# Crée un nouveau livre
@app.route('/books', methods=['POST'])
def create_book():
    if not request.json or "title" not in request.json or "author" not in request.json:
        abort(400, description="Invalid request data")
    new_book = {
        "id": get_next_id(),
        "title": request.json["title"],
        "author": request.json["author"]
    }
    books.append(new_book)
    return jsonify(new_book), 201

# Met à jour un livre existant par son ID
@app.route('/books/<int:book_id>', methods=['PUT'])
def update_book(book_id):
    book = next((book for book in books if book["id"] == book_id), None)
    if not book:
        abort(404, description="Book not found")
    if not request.json:
        abort(400, description="Invalid request data")
    book["title"] = request.json.get("title", book["title"])
    book["author"] = request.json.get("author", book["author"])
    return jsonify(book)

# Supprime un livre par son ID
@app.route('/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    book = next((book for book in books if book["id"] == book_id), None)
    if not book:
        abort(404, description="Book not found")
    books.remove(book)
    return jsonify({"result": True})

# Gestion des erreurs 404 et 400
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": error.description}), 404

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": error.description}), 400

# Point d'entrée pour lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)

```
