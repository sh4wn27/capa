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

---

## Installation

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/capa-project/capa.git
cd capa
uv sync
```

For GPU support (recommended for ESM-2 inference), install PyTorch with CUDA separately per [pytorch.org](https://pytorch.org/get-started/locally/) and then run `uv sync`.

---

## Quick Start

### 1. Download data and sequences

```bash
# Download IPD-IMGT/HLA allele sequences
uv run python scripts/download_hla_seqs.py

# Download UCI BMT dataset and preprocess
uv run python scripts/preprocess.py
```

### 2. Train the model

```bash
uv run python scripts/train.py --config configs/default.yaml
```

### 3. Evaluate

```bash
uv run python scripts/evaluate.py --checkpoint runs/best/model.pt
```

### 4. Predict (Python API)

```python
from capa.api.predict import predict_risk

result = predict_risk(
    donor_hla={"A": "A*02:01", "B": "B*07:02", "DRB1": "DRB1*15:01"},
    recipient_hla={"A": "A*24:02", "B": "B*07:02", "DRB1": "DRB1*15:01"},
    clinical_covariates={"age_recipient": 8, "disease": "ALL", "conditioning": "MAC"},
)
# result.gvhd_risk, result.relapse_risk, result.trm_risk — cumulative incidence curves
```

### 5. Web frontend (local)

```bash
cd web
npm install
npm run dev   # http://localhost:3000
```

---

## Project Structure

```
capa/              # Python package
├── config.py      # Pydantic settings & hyperparameters
├── data/          # Dataset loading, HLA parsing, splits
├── embeddings/    # ESM-2 encoder + HDF5 embedding cache
├── model/         # Cross-attention interaction + DeepHit survival head
├── training/      # Training loop, evaluation metrics
├── interpret/     # Attention maps, SHAP explanations
└── api/           # Inference pipeline + Pydantic schemas
scripts/           # CLI entry points (train, evaluate, preprocess)
web/               # Next.js 14 frontend (Vercel deployment)
notebooks/         # EDA, embedding exploration, model development
paper/             # LaTeX manuscript
tests/             # pytest test suite (>80% coverage target)
```

---

## Model Architecture

```
Donor HLA alleles + Recipient HLA alleles + Clinical covariates
      │
      ▼
[HLA Sequence Lookup]  →  amino acid sequences per allele
      │
      ▼
[ESM-2 Encoder]  →  1280-dim embedding per allele (frozen, cached)
      │
      ▼
[Donor Matrix: n_loci × 1280]   [Recipient Matrix: n_loci × 1280]
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
```

---

## Development

```bash
uv run pytest                  # run tests
uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy capa/              # type check
```

---

## Data

- **UCI Bone Marrow Transplant Dataset** — 187 pediatric allogeneic HSCT patients. Download instructions in `data/README.md`.
- **IPD-IMGT/HLA Database** — full protein sequences for all known HLA alleles. Downloaded automatically by `scripts/download_hla_seqs.py`.

---

## Current Phase

**Phase 1**: Data pipeline + HLA embedding engine. See `capa/data/` and `capa/embeddings/`.

---

## License

MIT — see [LICENSE](LICENSE).
