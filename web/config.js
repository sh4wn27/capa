/* CAPA backend URL.
   Local dev:  start with `uv run uvicorn capa.api.predict:app --reload --port 8000`
   Production: update to your Modal URL after `modal deploy modal_app.py`            */
window.CAPA_CONFIG = {
  apiUrl: 'http://localhost:8000'
};
