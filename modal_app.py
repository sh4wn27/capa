"""Modal deployment for the CAPA FastAPI prediction backend.

Deploys capa/api/predict.py as a Modal ASGI web endpoint on a T4 GPU.

Usage
-----
Dev server (hot-reload, ephemeral URL printed to stdout):
    python3 -m modal serve modal_app.py

Deploy to production (persistent URL):
    python3 -m modal deploy modal_app.py

Set the deployed URL as CAPA_BACKEND_URL in your Next.js / Vercel project:
    vercel env add CAPA_BACKEND_URL production

Architecture
------------
Container image (Modal 1.x API):
    debian_slim + CUDA libraries
    pip deps installed from the pyproject.toml dependency list
    capa/ package added via image.add_local_python_source()  → /root/capa
    data/processed/           added via image.add_local_dir() → /root/data/processed/
    runs/test_run/best_model.pt added via image.add_local_file() → /root/runs/best/model.pt

GPU: T4 (16 GB VRAM) — adequate for ESM-2 650M at inference time.
     The CAPA model head itself is only ~5 MB; ESM-2 is cached and frozen.
"""

from __future__ import annotations

from pathlib import Path

import modal

# ── App ──────────────────────────────────────────────────────────────────────
app = modal.App("capa-backend")

PROJECT_ROOT = Path(__file__).parent

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # Required by h5py and scipy (used by lifelines / scikit-learn)
        "libhdf5-dev",
        "libgomp1",
    )
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.3.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "h5py>=3.11.0",
        "scikit-learn>=1.4.0",
    )
    # ── Environment variables (must come BEFORE add_local_* calls) ──────────
    .env(
        {
            # Checkpoint path (picked up by capa/api/predict.py lifespan hook)
            "CAPA_CHECKPOINT": "/root/runs/best/model.pt",
            # Pydantic-settings env overrides for DataConfig / EmbeddingConfig
            "CAPA_DATA__RAW_DIR": "/root/data/raw",
            "CAPA_DATA__PROCESSED_DIR": "/root/data/processed",
            "CAPA_EMBED__CACHE_PATH": "/root/data/processed/hla_embeddings.h5",
            # Use CUDA on the T4
            "CAPA_EMBED__DEVICE": "cuda",
            # Allow the Next.js frontend from any origin (tighten for production
            # by overriding via a Modal Secret or CAPA_CORS_ORIGINS env var)
            "CAPA_CORS_ORIGINS": "*",
        }
    )
    # ── Local file additions MUST come last ──────────────────────────────────
    # add_local_* calls inject files at container startup (not into the image
    # layer), so no build steps can follow them. Keeping them last also means
    # the expensive pip/apt layers above are cached even when source changes.
    #
    # capa Python package — adds capa/ to /root on the container PYTHONPATH
    .add_local_python_source("capa")
    # HDF5 embedding cache + HLA sequences JSON
    .add_local_dir(
        str(PROJECT_ROOT / "data" / "processed"),
        remote_path="/root/data/processed",
    )
    # Model checkpoint: runs/test_run/best_model.pt → /root/runs/best/model.pt
    .add_local_file(
        str(PROJECT_ROOT / "runs" / "test_run" / "best_model.pt"),
        remote_path="/root/runs/best/model.pt",
    )
)

# ── Web endpoint ──────────────────────────────────────────────────────────────


@app.function(
    image=image,
    gpu="T4",
    # Keep the container alive for 5 minutes between requests to avoid
    # repeated ESM-2 / checkpoint cold-start latency.
    scaledown_window=300,
    # 120 s hard timeout — ESM-2 embed on first request can be slow on CPU
    # fallback; T4 inference is typically <1 s per request after warm-up.
    timeout=120,
    # Uncomment to inject secrets (e.g. a private HuggingFace token):
    # secrets=[modal.Secret.from_name("capa-backend-secrets")],
)
# Up to 10 requests handled concurrently within a single container.
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app() -> object:
    """Return the CAPA FastAPI application to Modal's ASGI runtime.

    Modal calls this once per container. The FastAPI lifespan hook fires on
    first access and loads the model checkpoint into GPU memory.
    """
    # capa/ is on /root which is already on PYTHONPATH via add_local_python_source
    from capa.api.predict import app as _app  # noqa: PLC0415

    return _app
