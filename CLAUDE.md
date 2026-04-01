# CAPA — Computational Architecture for Predicting Alloimmunity

## Project Overview

CAPA is an open-source computational framework that uses protein language models (ESM-2) to create structure-aware HLA mismatch representations and predict post-transplant immune complications (acute GvHD, relapse, transplant-related mortality) via deep competing-risks survival analysis. The goal is to replace categorical HLA match/mismatch scoring with continuous, biologically meaningful embeddings that capture the immunological distance between donor-recipient allele pairs.

## Repository Structure

```
capa/
├── CLAUDE.md                  # You are here
├── README.md                  # Project overview, installation, quick start
├── pyproject.toml             # Project metadata, dependencies (use uv)
├── LICENSE                    # MIT License
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions: lint, test, type-check
├── data/
│   ├── raw/                   # UCI BMT dataset (bone-marrow.csv), IPD-IMGT/HLA sequences
│   ├── processed/             # Cleaned, split, feature-engineered datasets
│   └── README.md              # Data provenance, download instructions
├── capa/
│   ├── __init__.py
│   ├── config.py              # Pydantic settings, hyperparameters, paths
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # UCI BMT dataset loading + cleaning
│   │   ├── hla_parser.py      # Parse HLA typing strings → structured alleles
│   │   └── splits.py          # Stratified train/val/test splits (competing risks aware)
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── esm_embedder.py    # ESM-2 allele sequence → embedding vectors
│   │   ├── hla_sequences.py   # IPD-IMGT/HLA allele → protein sequence lookup
│   │   └── cache.py           # Embedding cache (HDF5) to avoid recomputation
│   ├── model/
│   │   ├── __init__.py
│   │   ├── interaction.py     # Donor-recipient cross-attention interaction network
│   │   ├── survival.py        # DeepHit / cause-specific deep survival head
│   │   ├── capa_model.py      # Full CAPA model (embedding → interaction → survival)
│   │   └── losses.py          # Competing risks loss (ranking + calibration)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop with early stopping, LR scheduling
│   │   └── evaluate.py        # C-index, Brier score, calibration plots
│   ├── interpret/
│   │   ├── __init__.py
│   │   ├── attention_maps.py  # Extract + visualize cross-attention weights
│   │   └── shap_explain.py    # SHAP values for clinical covariates
│   └── api/
│       ├── __init__.py
│       ├── predict.py         # Inference pipeline: HLA strings → risk scores
│       └── schemas.py         # Pydantic input/output schemas for API
├── web/                       # Next.js frontend (deployed on Vercel)
│   ├── package.json
│   ├── next.config.js
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx           # Landing page
│   │   ├── predict/
│   │   │   └── page.tsx       # Interactive prediction tool
│   │   ├── about/
│   │   │   └── page.tsx       # About page (project, team, story)
│   │   └── api/
│   │       └── predict/
│   │           └── route.ts   # API route → calls Python backend
│   ├── components/
│   │   ├── HLAInput.tsx       # HLA typing input form
│   │   ├── RiskChart.tsx      # Competing risks visualization (recharts)
│   │   ├── AttentionHeatmap.tsx # Donor-recipient mismatch attention map
│   │   └── Hero.tsx           # Landing page hero section
│   └── lib/
│       └── api.ts             # Frontend API client
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis of UCI BMT
│   ├── 02_embeddings.ipynb    # HLA embedding exploration + UMAP viz
│   ├── 03_model_dev.ipynb     # Model prototyping + ablation studies
│   └── 04_figures.ipynb       # Publication-ready figures
├── paper/
│   ├── main.tex               # LaTeX manuscript
│   ├── references.bib         # BibTeX references
│   ├── figures/               # Generated figures (PDF/SVG)
│   └── supplementary.tex      # Supplementary methods + tables
├── scripts/
│   ├── download_hla_seqs.py   # Download IPD-IMGT/HLA allele sequences
│   ├── preprocess.py          # Full data preprocessing pipeline
│   ├── train.py               # Training entry point (CLI with argparse)
│   ├── evaluate.py            # Evaluation entry point
│   └── generate_figures.py    # Generate all paper figures
└── tests/
    ├── test_hla_parser.py
    ├── test_embeddings.py
    ├── test_model.py
    └── test_api.py
```

## Tech Stack

- **Language**: Python 3.11+
- **Package manager**: uv (NOT pip)
- **Deep learning**: PyTorch 2.x
- **Protein LM**: ESM-2 (facebook/esm2_t33_650M_UR50D) via HuggingFace transformers
- **Survival analysis**: lifelines (baseline comparisons), custom PyTorch (deep models)
- **Data**: pandas, numpy, polars (for large operations)
- **Visualization**: matplotlib, seaborn (paper figures), plotly (notebooks)
- **Web frontend**: Next.js 14+ with TypeScript, Tailwind CSS, shadcn/ui
- **Deployment**: Vercel (frontend), Modal or Railway (Python backend API)
- **Testing**: pytest, pytest-cov (>80% coverage target)
- **Linting**: ruff (linting + formatting), mypy (type checking)
- **CI**: GitHub Actions

## Coding Standards

- Type hints on ALL function signatures (enforced by mypy --strict)
- Docstrings on all public functions (NumPy style)
- No `print()` — use `logging` module with structured output
- Config via Pydantic BaseSettings (no hardcoded paths/hyperparameters)
- All data paths relative to project root via config
- Reproducibility: all random seeds configurable, results must be deterministic given seed
- Tests for every module; use fixtures for sample data

## Key Technical Decisions

1. **ESM-2 over one-hot encoding**: HLA alleles are protein sequences. ESM-2 embeddings capture structural/functional similarity that categorical encoding misses entirely.
2. **Cross-attention for interaction modeling**: The immunological "conflict" between donor and recipient HLA is inherently relational. Cross-attention lets the model learn which specific allele-pair interactions matter.
3. **DeepHit for competing risks**: GvHD, relapse, and TRM are competing events. DeepHit handles this natively with a joint distribution over event types and times.
4. **UCI BMT dataset as primary validation**: 187 pediatric patients, well-characterized, publicly available. Limitation: small N. Acknowledge this; frame deep model as proof-of-concept with traditional model comparisons.

## Data

- **UCI Bone Marrow Transplant Dataset**: https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children
  - 187 pediatric patients, allogeneic HSCT
  - Features: donor/recipient HLA (antigen + allele level), clinical covariates, outcomes (survival time, GvHD, relapse, cause of death)
- **IPD-IMGT/HLA Database**: https://www.ebi.ac.uk/ipd/imgt/hla/
  - Full protein sequences for all known HLA alleles
  - Need: allele name → amino acid sequence mapping

## Model Architecture Summary

```
Input: Donor HLA alleles (A, B, C, DRB1, ...) + Recipient HLA alleles + Clinical covariates
  │
  ▼
[HLA Sequence Lookup] → amino acid sequences per allele
  │
  ▼
[ESM-2 Encoder] → 1280-dim embedding per allele (frozen, cached)
  │
  ▼
[Donor Matrix: n_loci × 1280]  [Recipient Matrix: n_loci × 1280]
  │                                │
  └──────────┬─────────────────────┘
             ▼
[Cross-Attention Interaction Network] → interaction features (128-dim)
  │
  ▼
[Concat with Clinical Covariates] → combined feature vector
  │
  ▼
[DeepHit Survival Head] → P(event_k, time_t) for k ∈ {GvHD, relapse, TRM}
  │
  ▼
Output: Cumulative incidence curves per event, risk scores, attention weights
```

## Important Commands

```bash
# Setup
uv sync                              # Install dependencies
uv run python scripts/download_hla_seqs.py  # Download HLA sequences
uv run python scripts/preprocess.py  # Preprocess UCI BMT data

# Development
uv run pytest                        # Run tests
uv run ruff check .                  # Lint
uv run ruff format .                 # Format
uv run mypy capa/                    # Type check

# Training
uv run python scripts/train.py --config configs/default.yaml
uv run python scripts/evaluate.py --checkpoint runs/best/model.pt

# Web
cd web && npm install && npm run dev  # Local dev server
```

## Current Phase

Phase 1: Data pipeline + HLA embedding engine. Start here.

## Notes for Claude Code

- When implementing a module, ALWAYS write tests alongside it in tests/
- Use descriptive commit messages: "feat(embeddings): implement ESM-2 HLA allele encoder with HDF5 caching"
- If unsure about a biological/clinical detail, leave a TODO comment with the question
- The UCI BMT dataset column names are messy — document every mapping clearly
- ESM-2 inference is GPU-heavy; implement CPU fallback and embedding caching early
- For the web frontend, prioritize the prediction tool page over the landing page
- Paper figures should be generated programmatically (scripts/generate_figures.py), never manually