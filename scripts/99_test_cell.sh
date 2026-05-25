#!/usr/bin/env bash
# Lance un .py de test dans l'env UV.
# Sert à vérifier qu'un bloc de code marche avant de l'intégrer dans le .md final.
# Usage: ./99_test_cell.sh scripts/_sandbox/foo.py

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <fichier.py>" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec uv run python "$@"
