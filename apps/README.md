# 🚀 Apps exécutables — Flask, FastAPI, Streamlit

Apps de démonstration en pur Python pour les 3 frameworks couverts par les notebooks `Flask_API`, `FastAPI_API`, `Streamlit_brique`.

> Contrairement aux notebooks (pseudo-code en cellules), ces apps sont **exécutables** directement.

## Lancement

### Flask
```bash
uv run flask --app apps/flask_api/app.py run --port 5000
# Test : curl http://localhost:5000/health
```

### FastAPI
```bash
uv add fastapi 'uvicorn[standard]' pydantic-settings 'pyjwt[crypto]' email-validator
uv run uvicorn apps.fastapi_api.main:app --reload --port 8000
# Test : open http://localhost:8000/docs
```

### Streamlit
```bash
uv add streamlit
uv run streamlit run apps/streamlit_demo/app.py
# Test : open http://localhost:8501
```

## Structure

```
apps/
├── README.md                  # ce fichier
├── flask_api/
│   └── app.py                 # CRUD users + auth API key + error handlers
├── fastapi_api/
│   └── main.py                # CRUD users + Pydantic v2 + JWT optionnel
└── streamlit_demo/
    └── app.py                 # Demo Cal Housing : data + viz + ML interactif
```

## Notes prod

- **Jamais** `app.run()` (Flask) ou `uvicorn --reload` en prod.
- Flask prod : `gunicorn apps.flask_api.app:app -w 4 -b 0.0.0.0:5000`
- FastAPI prod : `gunicorn apps.fastapi_api.main:app -k uvicorn.workers.UvicornWorker -w 4`
- Streamlit prod : container Docker + reverse proxy (nginx/Caddy).
