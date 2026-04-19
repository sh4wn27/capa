# CAPA — Computational Architecture for Predicting Alloimmunity

CAPA is an open-source computational framework that uses protein language models (ESM-2) to create structure-aware HLA mismatch representations and predict post-transplant immune complications via deep competing-risks survival analysis.

Instead of categorical HLA match/mismatch scores, CAPA learns **continuous, biologically meaningful embeddings** that capture the immunological distance between donor-recipient allele pairs, then models GvHD, relapse, and transplant-related mortality (TRM) as competing risks.

---

## Why CAPA?

| Traditional approach | CAPA |
|---|---|
| Binary match / mismatch per locus | Continuous embedding distance |
| Treats all mismatches as equal | Captures structural/functional similarity |
| Cox PH for single endpoint | DeepHit for competing risks |
| Black-box predictions | Cross-attention interpretability |
| Single-donor evaluation | Multi-donor ranking comparison |

---

## Installation

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/sh4wn27/capa.git
cd capa
uv sync
```

For GPU support (recommended for ESM-2 inference), install PyTorch with CUDA separately per [pytorch.org](https://pytorch.org/get-started/locally/) and then run `uv sync`.

---

## Quick Start

### 1. Download data and sequences

```bash
uv run python scripts/download_hla_seqs.py
uv run python scripts/preprocess.py
```

### 2. Train the model

```bash
uv run python scripts/train.py --config configs/default.yaml
```

Override any hyperparameter inline:

```bash
CAPA_TRAINING__LR=5e-4 uv run python scripts/train.py --config configs/default.yaml
```

### 3. Evaluate

```bash
uv run python scripts/evaluate.py --checkpoint runs/best/model.pt --config configs/default.yaml
```

### 4. Predict (Python API)

```python
from capa.api.predict import predict_risk

result = predict_risk(
    donor_hla={"A": "A*02:01", "B": "B*07:02", "DRB1": "DRB1*15:01"},
    recipient_hla={"A": "A*24:02", "B": "B*07:02", "DRB1": "DRB1*15:01"},
)
# result.gvhd.risk_score, result.relapse.risk_score, result.trm.risk_score
# result.gvhd.cumulative_incidence  — 100-bin CIF over 0–730 days
```

### 5. Multi-donor comparison (Python API)

```python
import httpx

resp = httpx.post("http://localhost:8000/compare", json={
    "recipient_hla": {"A": "A*24:02", "DRB1": "DRB1*15:01"},
    "donors": [
        {"label": "Donor A", "donor_hla": {"A": "A*02:01", "DRB1": "DRB1*15:01"}},
        {"label": "Donor B", "donor_hla": {"A": "A*24:02", "DRB1": "DRB1*07:01"}},
        {"label": "Donor C", "donor_hla": {"A": "A*02:01", "DRB1": "DRB1*07:01"}},
    ],
})
data = resp.json()
print(data["best_donor_label"])       # e.g. "Donor A"
for d in data["donors"]:
    print(d["rank"], d["label"], d["gvhd_risk"])
```

### 6. Web frontend (local)

```bash
cd web && npm install && npm run dev   # http://localhost:3000
```

Or with Docker:

```bash
docker compose up backend             # Python API only
docker compose --profile fullstack up # API + Next.js frontend
```

---

## Project Structure

```
capa/              # Python package
├── config.py      # Pydantic settings & hyperparameters (YAML + env var override)
├── data/          # Dataset loading, HLA parsing, stratified splits
├── embeddings/    # ESM-2 encoder + HDF5 embedding cache
├── model/         # Cross-attention interaction + DeepHit survival head
├── training/      # Training loop, evaluation metrics, isotonic calibration
├── interpret/     # Attention maps, SHAP explanations
└── api/           # FastAPI inference server + Pydantic schemas
configs/           # YAML config files (default.yaml)
scripts/           # CLI entry points (train, evaluate, preprocess, compare)
web/               # Next.js 14 frontend (Vercel deployment)
notebooks/         # EDA, embedding exploration, model development
paper/             # LaTeX manuscript
tests/             # pytest test suite (92% coverage)
```

---

## Model Architecture

```
Donor HLA alleles (A/B/C/DRB1/DQB1/DPB1) + Recipient HLA alleles + Clinical covariates
      │
      ▼
[HLA Sequence Lookup]  →  amino acid sequences per allele
      │
      ▼
[ESM-2 Encoder]  →  1280-dim embedding per allele
      │             (frozen by default; last-N-layer fine-tuning supported)
      ▼
[Donor Matrix: n_loci × 1280]   [Recipient Matrix: n_loci × 1280]
      │         + locus positional embeddings (optional)
      └──────────┬──────────────────────────┘
                 ▼
[Cross-Attention Interaction Network]  →  128-dim interaction features
                 │
                 ▼
[Concat with Clinical Covariates]  →  combined feature vector
                 │
                 ▼
[DeepHit Survival Head]  →  P(event_k, time_t)  for k ∈ {GvHD, relapse, TRM}
                 │
                 ▼
Cumulative incidence curves · risk scores · attention weights
                 │
                 ▼
[Isotonic Calibration]  →  calibrated per-(event × time-bin) probabilities
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Override any field with an environment variable using the `CAPA_` prefix and `__` as separator:

```bash
CAPA_MODEL__INTERACTION_DIM=256 uv run python scripts/train.py
```

Priority: **env vars** > **YAML file** > **pydantic defaults**.

---

## Development

```bash
uv run pytest                  # run tests (789 passing, 92% coverage)
uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy capa/              # type check
```

CI runs on every push: lint, type check, unit tests, API smoke tests, and Next.js production build.

---

## Data

- **UCI Bone Marrow Transplant Dataset** — 187 pediatric allogeneic HSCT patients. Download instructions in `data/README.md`.
- **IPD-IMGT/HLA Database** — full protein sequences for all known HLA alleles. Downloaded automatically by `scripts/download_hla_seqs.py`.

---

## License

MIT — see [LICENSE](LICENSE).
