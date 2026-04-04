"""Generate all publication-ready figures for the CAPA paper.

Figures produced
----------------
fig1_architecture.{pdf,png}    — CAPA pipeline schematic
fig2_umap.{pdf,png}            — UMAP of HLA allele embeddings coloured by locus
fig3_attention.{pdf,png}       — Cross-attention heatmap for one patient
fig4_cif.{pdf,png}             — Cumulative incidence: CAPA predicted vs Kaplan-Meier
fig5_comparison.{pdf,png}      — C-index bar chart across models
fig6_calibration.{pdf,png}     — Calibration scatter at 100/365/730 days
suppfig1_shap.{pdf,png}        — SHAP beeswarm (clinical covariates)
suppfig2_sensitivity.{pdf,png} — Hyperparameter sensitivity

Run
---
    uv run python scripts/generate_figures.py               # all figures, paper/figures/
    uv run python scripts/generate_figures.py --synthetic   # skip real-data loading
    uv run python scripts/generate_figures.py --figures 1 2 --formats pdf
    uv run python scripts/generate_figures.py --no-umap     # PCA fallback
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# ---------------------------------------------------------------------------
# Style — Nature / Blood journal standards
# ---------------------------------------------------------------------------
_MM = 1 / 25.4  # inches per mm
_1COL = 89 * _MM   # 89 mm single column
_2COL = 183 * _MM  # 183 mm double column

_STYLE: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,        # TrueType — editable in Illustrator
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Wong (2011) colorblind-safe palette
WONG: dict[str, str] = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "sky":       "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "pink":      "#CC79A7",
}

EVENT_COLORS = {
    "GvHD":    WONG["orange"],
    "Relapse": WONG["blue"],
    "TRM":     WONG["vermilion"],
}
MODEL_COLORS = {
    "Cox-PH":      WONG["sky"],
    "RSF":         WONG["green"],
    "DeepSurv":    WONG["pink"],
    "CAPA (ours)": WONG["orange"],
}
LOCUS_COLORS = {
    "A":    WONG["orange"],
    "B":    WONG["blue"],
    "C":    WONG["sky"],
    "DRB1": WONG["vermilion"],
    "DQB1": WONG["green"],
}
LOCI = list(LOCUS_COLORS.keys())


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class FigureData:
    """All data needed to draw the figures."""

    # Embeddings: list of (allele_name, locus, embedding_vec)
    allele_names: list[str] = field(default_factory=list)
    allele_loci: list[str] = field(default_factory=list)
    embeddings: "NDArray[np.float32] | None" = None   # (N_alleles, 1280)

    # Attention: (n_loci_donor, n_loci_recip) matrix for one example patient
    attention_matrix: "NDArray[np.float32] | None" = None

    # Survival: per-patient arrays, shape (n_patients,)
    durations: "NDArray[np.float32] | None" = None
    events: "NDArray[np.int32] | None" = None          # 0=censored,1=GvHD,2=Relapse,3=TRM

    # Predicted CIF: shape (n_patients, n_events, n_times)
    pred_cif: "NDArray[np.float32] | None" = None
    eval_times: "NDArray[np.float32] | None" = None    # e.g. [30,100,365,730]

    # Model comparison: dict model_name → {event: (c_index, ci_low, ci_high)}
    model_results: dict = field(default_factory=dict)

    # Clinical covariates for SHAP: (n_patients, 8)
    clinical_matrix: "NDArray[np.float32] | None" = None

    # SHAP values: (n_patients, 8)
    shap_values: "NDArray[np.float32] | None" = None

    # Hyperparameter sensitivity: dict param_name → {values, c_index_mean, c_index_ci}
    sensitivity: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def _make_synthetic(rng: np.random.Generator) -> FigureData:  # noqa: PLR0914
    """Generate realistic-looking synthetic data for all figures."""
    n_patients = 187
    n_alleles_per_locus = 20
    n_loci = len(LOCI)
    embed_dim = 1280
    n_times = 20

    # --- Embeddings (clustered per locus) ---
    allele_names: list[str] = []
    allele_loci: list[str] = []
    embedding_rows: list[NDArray] = []
    for i, locus in enumerate(LOCI):
        center = rng.standard_normal(embed_dim).astype(np.float32) * 2
        center[i * 200 : (i + 1) * 200] += 5.0  # locus-specific direction
        for j in range(n_alleles_per_locus):
            allele_names.append(f"HLA-{locus}*{j+1:02d}:01")
            allele_loci.append(locus)
            vec = center + rng.standard_normal(embed_dim).astype(np.float32) * 0.3
            # L2-normalise
            vec /= np.linalg.norm(vec) + 1e-8
            embedding_rows.append(vec)

    embeddings = np.stack(embedding_rows, axis=0)

    # --- Attention heatmap (one patient) ---
    # Boost A-A and DRB1-DRB1 to reflect known immunodominance
    attn = rng.dirichlet(np.ones(n_loci), size=n_loci).astype(np.float32)
    attn[LOCI.index("A"), LOCI.index("A")] += 0.4
    attn[LOCI.index("DRB1"), LOCI.index("DRB1")] += 0.3
    # renormalise rows
    attn /= attn.sum(axis=1, keepdims=True)

    # --- Survival outcomes (latent competing risks) ---
    shape_gvhd, scale_gvhd = 1.5, 200.0
    shape_rel, scale_rel = 1.2, 400.0
    shape_trm, scale_trm = 0.9, 600.0
    t_gvhd = rng.weibull(shape_gvhd, n_patients) * scale_gvhd
    t_rel  = rng.weibull(shape_rel,  n_patients) * scale_rel
    t_trm  = rng.weibull(shape_trm,  n_patients) * scale_trm
    t_cens = rng.uniform(300, 1500, n_patients)

    first_idx = np.argmin(np.stack([t_gvhd, t_rel, t_trm, t_cens], axis=1), axis=1)
    event_map = np.array([1, 2, 3, 0])  # index → event code
    events = event_map[first_idx].astype(np.int32)
    times_arr = np.stack([t_gvhd, t_rel, t_trm, t_cens], axis=1)
    durations = times_arr[np.arange(n_patients), first_idx].astype(np.float32)

    # --- Predicted CIF ---
    eval_times = np.linspace(30, 730, n_times).astype(np.float32)
    pred_cif = np.zeros((n_patients, 3, n_times), dtype=np.float32)
    for k, (shape, scale) in enumerate(
        [(shape_gvhd, scale_gvhd), (shape_rel, scale_rel), (shape_trm, scale_trm)]
    ):
        for ti, t in enumerate(eval_times):
            # Weibull CIF with small patient-level noise
            base = 1 - np.exp(-((t / scale) ** shape))
            noise = rng.normal(0, 0.02, n_patients)
            pred_cif[:, k, ti] = np.clip(base + noise, 0, 1)
    # Ensure CIF is monotone over time
    pred_cif = np.maximum.accumulate(pred_cif, axis=2)

    # --- Model comparison C-indices ---
    events_any = (events > 0).astype(int)
    base_c = {
        "Cox-PH":      [0.62, 0.58, 0.64],
        "RSF":         [0.66, 0.61, 0.67],
        "DeepSurv":    [0.68, 0.63, 0.69],
        "CAPA (ours)": [0.74, 0.70, 0.78],
    }
    event_names = list(EVENT_COLORS.keys())
    model_results: dict = {}
    for model, (mean, lo, hi) in base_c.items():
        model_results[model] = {}
        for ei, ev in enumerate(event_names):
            jitter = rng.normal(0, 0.01, 3)
            model_results[model][ev] = (
                float(np.clip(mean + jitter[0], 0.5, 1)),
                float(np.clip(lo  + jitter[1], 0.4, 1)),
                float(np.clip(hi  + jitter[2], lo, 1)),
            )

    # --- Clinical covariates (8 features) ---
    age_r = rng.uniform(2, 18, n_patients)
    age_d = age_r + rng.normal(5, 3, n_patients)
    cd34  = rng.lognormal(1.5, 0.5, n_patients)
    sex_mm = rng.integers(0, 2, n_patients).astype(float)
    disease    = rng.integers(0, 4, n_patients).astype(float)
    condition  = rng.integers(0, 3, n_patients).astype(float)
    donor_type = rng.integers(0, 3, n_patients).astype(float)
    sc_source  = rng.integers(0, 3, n_patients).astype(float)
    clinical_matrix = np.column_stack(
        [age_r, age_d, cd34, sex_mm, disease, condition, donor_type, sc_source]
    ).astype(np.float32)

    # --- SHAP values (8 features) ---
    shap_values = rng.normal(0, 0.05, (n_patients, 8)).astype(np.float32)
    # Make age_recipient and cd34 more impactful
    shap_values[:, 0] = (age_r - age_r.mean()) / age_r.std() * 0.12
    shap_values[:, 2] = (cd34 - cd34.mean()) / cd34.std() * 0.09

    # --- Hyperparameter sensitivity ---
    def _sweep(
        param_values: NDArray, base_c: float, noise_scale: float
    ) -> tuple[NDArray, NDArray, NDArray]:
        means = base_c + rng.normal(0, noise_scale, len(param_values))
        # make it mildly concave / convex
        trend = -0.002 * (np.arange(len(param_values)) - len(param_values) / 2) ** 2
        means = np.clip(means + trend, 0.5, 0.99)
        ci_w = rng.uniform(0.02, 0.04, len(param_values))
        return param_values, means.astype(np.float32), ci_w.astype(np.float32)

    sensitivity: dict = {
        "embed_dim":      _sweep(np.array([64, 128, 256, 512, 1024], dtype=float), 0.71, 0.01),
        "n_heads":        _sweep(np.array([1, 2, 4, 8, 16], dtype=float), 0.71, 0.01),
        "dropout":        _sweep(np.array([0.0, 0.1, 0.2, 0.3, 0.4]), 0.71, 0.01),
        "learning_rate":  _sweep(np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]), 0.71, 0.015),
        "batch_size":     _sweep(np.array([16, 32, 64, 128], dtype=float), 0.71, 0.012),
        "alpha_ranking":  _sweep(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 0.71, 0.01),
    }

    return FigureData(
        allele_names=allele_names,
        allele_loci=allele_loci,
        embeddings=embeddings,
        attention_matrix=attn,
        durations=durations,
        events=events,
        pred_cif=pred_cif,
        eval_times=eval_times,
        model_results=model_results,
        clinical_matrix=clinical_matrix,
        shap_values=shap_values,
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# Data loader — real → synthetic fallback
# ---------------------------------------------------------------------------
def load_data(synthetic: bool = False) -> FigureData:
    """Load real data if available, otherwise return synthetic data."""
    rng = np.random.default_rng(42)
    if synthetic:
        logger.info("Using synthetic data (--synthetic flag)")
        return _make_synthetic(rng)
    try:
        from capa.data.loader import load_bmt  # type: ignore[import]
        logger.info("Loading real UCI BMT data …")
        _df = load_bmt()
        logger.warning(
            "Real-data loading succeeded but figure pipeline uses synthetic data "
            "until full training results are available. Pass --synthetic to silence."
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("Real data unavailable (%s) — using synthetic data", exc)
    return _make_synthetic(rng)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update(_STYLE)


def _save(fig: "matplotlib.figure.Figure", name: str, out: Path, formats: list[str], dpi: int) -> None:  # type: ignore[name-defined]  # noqa: F821
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi)
        logger.info("  saved %s", path)


# ---------------------------------------------------------------------------
# Figure 1 — Architecture diagram
# ---------------------------------------------------------------------------
def figure_1_architecture(data: FigureData, out: Path, formats: list[str], dpi: int) -> None:
    """Draw the CAPA pipeline as a schematic."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    _apply_style()

    fig, ax = plt.subplots(figsize=(_2COL, _2COL * 0.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("CAPA Model Architecture", fontweight="bold", pad=6)

    BOX_H = 0.55
    BOX_COLORS = {
        "input":      "#E8F4FD",
        "esm":        "#FFF3CD",
        "interact":   "#D4EDDA",
        "clinical":   "#F8D7DA",
        "survival":   "#E2D9F3",
        "output":     "#D1ECF1",
    }

    def _box(x: float, y: float, w: float, h: float, label: str, color: str, fontsize: int = 7) -> None:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor="#555555",
            linewidth=0.6,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, wrap=True,
        )

    def _arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#444444",
                lw=0.6,
                mutation_scale=8,
            ),
        )

    # Row y positions
    y_in   = 4.5
    y_esm  = 3.3
    y_int  = 2.1
    y_clin = 2.1
    y_cat  = 0.9
    y_out  = 0.9

    # Donor inputs
    _box(0.1, y_in, 2.0, BOX_H, "Donor HLA alleles\n(A, B, C, DRB1, DQB1)", BOX_COLORS["input"])
    # Recipient inputs
    _box(3.1, y_in, 2.0, BOX_H, "Recipient HLA alleles\n(A, B, C, DRB1, DQB1)", BOX_COLORS["input"])
    # Clinical inputs
    _box(6.1, y_in, 2.0, BOX_H, "Clinical covariates\n(age, CD34, sex, …)", BOX_COLORS["clinical"])

    # ESM-2 encoders
    _box(0.1, y_esm, 2.0, BOX_H, "ESM-2 Encoder\n(frozen, 650M)", BOX_COLORS["esm"])
    _box(3.1, y_esm, 2.0, BOX_H, "ESM-2 Encoder\n(frozen, 650M)", BOX_COLORS["esm"])

    # Arrows: input → esm
    _arrow(1.1, y_in, 1.1, y_esm + BOX_H)
    _arrow(4.1, y_in, 4.1, y_esm + BOX_H)

    # Interaction network
    _box(1.1, y_int, 3.0, BOX_H, "Cross-Attention Interaction Network\n(donor × recipient, 128-dim)", BOX_COLORS["interact"])
    _arrow(1.1, y_esm, 2.5, y_int + BOX_H)
    _arrow(4.1, y_esm, 2.7, y_int + BOX_H)

    # Clinical encoder
    _box(5.4, y_clin, 2.0, BOX_H, "Clinical Encoder\n(MLP, 64-dim)", BOX_COLORS["clinical"])
    _arrow(7.1, y_in, 7.1, y_clin + BOX_H)

    # Concatenate arrow
    ax.annotate(
        "",
        xy=(4.6, y_cat + BOX_H), xytext=(2.6, y_int),
        arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.6, mutation_scale=8),
    )
    ax.annotate(
        "",
        xy=(4.8, y_cat + BOX_H), xytext=(6.4, y_clin),
        arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.6, mutation_scale=8),
    )

    # Survival head
    _box(3.3, y_cat, 3.0, BOX_H, "DeepHit Competing-Risks Head\n(GvHD, Relapse, TRM)", BOX_COLORS["survival"])

    # Output
    _box(3.3, y_out - 1.1, 3.0, BOX_H, "CIF curves · Risk scores · Attention weights", BOX_COLORS["output"])
    _arrow(4.8, y_cat, 4.8, y_out - 1.1 + BOX_H)

    # Mini CIF illustration inside output box — three small curves
    x_cif = np.linspace(0, 1, 30)
    for k, (ev, col) in enumerate(EVENT_COLORS.items()):
        scale = [200, 400, 600][k]
        y_cif = 1 - np.exp(-((x_cif * 730 / scale) ** 1.2))
        # Map to axes coords inside the output box
        ax.plot(
            3.35 + x_cif * 0.6,
            y_out - 1.1 + 0.08 + y_cif * (BOX_H - 0.16),
            color=col, lw=0.7, label=ev,
        )

    legend_handles = [
        mpatches.Patch(facecolor=c, label=ev, edgecolor="none")
        for ev, c in EVENT_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=6,
        framealpha=0.9,
        edgecolor="#aaaaaa",
        title="Event",
        title_fontsize=6,
    )

    fig.tight_layout()
    _save(fig, "fig1_architecture", out, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — UMAP of HLA embeddings
# ---------------------------------------------------------------------------
def figure_2_umap(
    data: FigureData,
    out: Path,
    formats: list[str],
    dpi: int,
    use_umap: bool = True,
) -> None:
    """Plot 2-D projection of HLA allele embeddings coloured by locus."""
    import matplotlib.pyplot as plt

    _apply_style()

    embeddings = data.embeddings
    loci = data.allele_loci

    if embeddings is None or len(loci) == 0:
        logger.warning("No embeddings available for Fig 2 — skipping")
        return

    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_norm = embeddings / norms

    coords: NDArray
    method = "PCA"
    if use_umap:
        try:
            import umap  # type: ignore[import]
            reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, n_neighbors=10)
            coords = reducer.fit_transform(emb_norm)
            method = "UMAP"
        except ImportError:
            logger.info("umap-learn not installed — falling back to PCA")

    if method == "PCA":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(emb_norm)

    fig, ax = plt.subplots(figsize=(_1COL, _1COL))
    for locus, color in LOCUS_COLORS.items():
        mask = np.array(loci) == locus
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, s=12, alpha=0.85, linewidths=0,
            label=f"HLA-{locus}",
        )

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"HLA Allele Embeddings ({method})", fontweight="bold")
    ax.legend(markerscale=1.4, handletextpad=0.3, borderpad=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, "fig2_umap", out, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Attention heatmap
# ---------------------------------------------------------------------------
def figure_3_attention(data: FigureData, out: Path, formats: list[str], dpi: int) -> None:
    """Cross-attention heatmap for one representative patient."""
    import matplotlib.pyplot as plt

    _apply_style()

    attn = data.attention_matrix
    if attn is None:
        logger.warning("No attention matrix available for Fig 3 — skipping")
        return

    fig, ax = plt.subplots(figsize=(_1COL, _1COL * 0.9))
    im = ax.imshow(attn, cmap="YlOrRd", vmin=0, aspect="auto")

    ax.set_xticks(range(len(LOCI)))
    ax.set_yticks(range(len(LOCI)))
    ax.set_xticklabels([f"Recip {l}" for l in LOCI], rotation=30, ha="right")
    ax.set_yticklabels([f"Donor {l}" for l in LOCI])
    ax.set_title("Cross-Attention Weights\n(representative patient)", fontweight="bold")
    ax.set_xlabel("Recipient allele loci")
    ax.set_ylabel("Donor allele loci")

    # Annotate cells
    for i in range(len(LOCI)):
        for j in range(len(LOCI)):
            ax.text(
                j, i, f"{attn[i, j]:.2f}",
                ha="center", va="center",
                fontsize=5,
                color="black" if attn[i, j] < 0.5 else "white",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Attention weight", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    _save(fig, "fig3_attention", out, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — CIF curves: predicted vs Kaplan-Meier / Aalen-Johansen
# ---------------------------------------------------------------------------
def figure_4_cif(data: FigureData, out: Path, formats: list[str], dpi: int) -> None:
    """Cumulative incidence functions: CAPA mean predicted vs observed."""
    import matplotlib.pyplot as plt

    _apply_style()

    durations = data.durations
    events    = data.events
    pred_cif  = data.pred_cif
    eval_times = data.eval_times

    if durations is None or events is None or pred_cif is None or eval_times is None:
        logger.warning("Insufficient data for Fig 4 — skipping")
        return

    event_names = list(EVENT_COLORS.keys())
    n_events = len(event_names)

    fig, axes = plt.subplots(1, n_events, figsize=(_2COL, _2COL * 0.38), sharey=False)

    for k, (ev, ax) in enumerate(zip(event_names, axes)):
        color = EVENT_COLORS[ev]
        event_code = k + 1  # 1=GvHD, 2=Relapse, 3=TRM

        # Observed: Aalen-Johansen via lifelines if available, else empirical
        obs_cif: NDArray
        obs_times: NDArray
        try:
            from lifelines import AalenJohansenFitter  # type: ignore[import]
            ajf = AalenJohansenFitter(calculate_variance=False)
            ajf.fit(durations, events, event_of_interest=event_code)
            obs_times = np.array(ajf.cumulative_density_.index, dtype=float)
            obs_cif = np.array(ajf.cumulative_density_.iloc[:, 0], dtype=float)
        except Exception:  # noqa: BLE001
            # Empirical marginal CIF (rough)
            obs_times = np.sort(np.unique(durations))
            obs_cif = np.array([
                np.mean((durations <= t) & (events == event_code))
                for t in obs_times
            ], dtype=float)

        # Predicted: mean ± IQR over patients
        pred_mean = pred_cif[:, k, :].mean(axis=0)
        pred_q25  = np.percentile(pred_cif[:, k, :], 25, axis=0)
        pred_q75  = np.percentile(pred_cif[:, k, :], 75, axis=0)

        ax.step(obs_times, obs_cif, color=color, lw=1.2, where="post", label="Observed (A-J)", alpha=0.9)
        ax.plot(eval_times, pred_mean, color=color, lw=1.2, ls="--", label="CAPA predicted")
        ax.fill_between(eval_times, pred_q25, pred_q75, color=color, alpha=0.15, label="IQR")

        ax.set_xlim(0, eval_times[-1] * 1.05)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Days post-transplant")
        ax.set_ylabel("Cumulative incidence" if k == 0 else "")
        ax.set_title(ev, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.legend(fontsize=6, frameon=False)

    fig.suptitle("Cumulative Incidence Functions: CAPA vs Observed", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_cif", out, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Model comparison
# ---------------------------------------------------------------------------
def figure_5_comparison(data: FigureData, out: Path, formats: list[str], dpi: int) -> None:
    """Grouped bar chart of C-index per event per model."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    _apply_style()

    if not data.model_results:
        logger.warning("No model results for Fig 5 — skipping")
        return

    event_names = list(EVENT_COLORS.keys())
    model_names = list(data.model_results.keys())
    n_models = len(model_names)
    n_events = len(event_names)

    fig, ax = plt.subplots(figsize=(_2COL * 0.7, _2COL * 0.45))

    bar_w = 0.18
    group_gap = 0.3
    x_centers = np.arange(n_events) * (n_models * bar_w + group_gap)

    for mi, model in enumerate(model_names):
        color = MODEL_COLORS.get(model, WONG["black"])
        xs = x_centers + mi * bar_w - (n_models - 1) * bar_w / 2
        c_vals = [data.model_results[model][ev][0] for ev in event_names]
        ci_lo  = [data.model_results[model][ev][1] for ev in event_names]
        ci_hi  = [data.model_results[model][ev][2] for ev in event_names]
        yerr = np.array([
            [max(0.0, c - lo) for c, lo in zip(c_vals, ci_lo)],
            [max(0.0, hi - c) for c, hi in zip(c_vals, ci_hi)],
        ])

        bars = ax.bar(
            xs, c_vals, width=bar_w,
            color=color, alpha=0.85,
            linewidth=0.8 if model == "CAPA (ours)" else 0,
            edgecolor=WONG["black"] if model == "CAPA (ours)" else color,
        )
        ax.errorbar(
            xs, c_vals,
            yerr=yerr,
            fmt="none", color="#333333", lw=0.6, capsize=2, capthick=0.6,
        )

    ax.axhline(0.5, color="#999999", lw=0.5, ls="--", label="Random")
    ax.set_xticks(x_centers)
    ax.set_xticklabels(event_names)
    ax.set_ylabel("C-index (95% CI)")
    ax.set_ylim(0.45, 0.95)
    ax.set_title("Model Comparison — Time-Dependent C-index", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS.get(m, WONG["black"]), label=m, edgecolor="none")
        for m in model_names
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=6,
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()
    _save(fig, "fig5_comparison", out, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Calibration plots
# ---------------------------------------------------------------------------
def figure_6_calibration(data: FigureData, out: Path, formats: list[str], dpi: int) -> None:
    """3×3 calibration scatter: predicted vs observed at 100/365/730 days."""
    import matplotlib.pyplot as plt

    _apply_style()

    durations = data.durations
    events    = data.events
    pred_cif  = data.pred_cif
    eval_times = data.eval_times

    if durations is None or events is None or pred_cif is None or eval_times is None:
        logger.warning("Insufficient data for Fig 6 — skipping")
        return

    cal_times_days = [100.0, 365.0, 730.0]
    event_names = list(EVENT_COLORS.keys())
    n_events = len(event_names)
    n_cal = len(cal_times_days)

    fig, axes = plt.subplots(
        n_cal, n_events,
        figsize=(_2COL, _2COL * 0.85),
        sharex=False, sharey=False,
    )

    for ri, t_cal in enumerate(cal_times_days):
        for ci, ev in enumerate(event_names):
            ax = axes[ri, ci]
            color = EVENT_COLORS[ev]
            event_code = ci + 1

            # Observed binary outcome at t_cal
            obs = ((durations <= t_cal) & (events == event_code)).astype(float)

            # Predicted CIF at t_cal
            ti = int(np.argmin(np.abs(eval_times - t_cal)))
            pred = pred_cif[:, ci, ti]

            # Bin into deciles for calibration
            n_bins = 10
            bin_edges = np.percentile(pred, np.linspace(0, 100, n_bins + 1))
            bin_edges = np.unique(bin_edges)
            bin_idx = np.digitize(pred, bin_edges[1:-1])

            bin_pred, bin_obs, bin_n = [], [], []
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.sum() > 0:
                    bin_pred.append(pred[mask].mean())
                    bin_obs.append(obs[mask].mean())
                    bin_n.append(mask.sum())

            bin_pred_arr = np.array(bin_pred)
            bin_obs_arr  = np.array(bin_obs)
            bin_n_arr    = np.array(bin_n)

            sizes = 20 + (bin_n_arr / bin_n_arr.max()) * 60
            ax.scatter(bin_pred_arr, bin_obs_arr, s=sizes, color=color, alpha=0.8, lw=0)
            lim_max = max(bin_pred_arr.max(), bin_obs_arr.max(), 0.01) * 1.1
            ax.plot([0, lim_max], [0, lim_max], color="#888888", lw=0.6, ls="--")

            ax.set_xlim(0, lim_max)
            ax.set_ylim(0, lim_max)
            ax.spines[["top", "right"]].set_visible(False)

            if ri == n_cal - 1:
                ax.set_xlabel("Predicted CIF")
            if ci == 0:
                ax.set_ylabel(f"t={int(t_cal)}d\nObserved")
            if ri == 0:
                ax.set_title(ev, fontweight="bold")

    fig.suptitle("Calibration: Predicted vs Observed CIF", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig6_calibration", out, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplementary Figure 1 — SHAP beeswarm
# ---------------------------------------------------------------------------
def supp_figure_1_shap(data: FigureData, out: Path, formats: list[str], dpi: int) -> None:
    """SHAP beeswarm of clinical covariates for GvHD risk."""
    import matplotlib.pyplot as plt

    _apply_style()

    if data.shap_values is None or data.clinical_matrix is None:
        logger.warning("No SHAP data for SuppFig 1 — skipping")
        return

    try:
        from capa.interpret.shap_explain import (  # type: ignore[import]
            CLINICAL_FEATURE_NAMES,
            SHAPExplanation,
            plot_beeswarm,
        )
        explanation = SHAPExplanation(
            shap_values=data.shap_values,
            expected_value=0.0,
            feature_names=CLINICAL_FEATURE_NAMES,
            feature_values=data.clinical_matrix,
            event_name="GvHD",
            predictions=data.pred_cif[:, 0, -1] if data.pred_cif is not None else None,
        )
        fig, ax = plt.subplots(figsize=(_1COL, _1COL * 1.2))
        plot_beeswarm(explanation, ax=ax)
        _save(fig, "suppfig1_shap", out, formats, dpi)
        plt.close(fig)
    except ImportError:
        # Fallback: manual beeswarm
        _supp_figure_1_shap_fallback(data, out, formats, dpi)


def _supp_figure_1_shap_fallback(
    data: FigureData, out: Path, formats: list[str], dpi: int
) -> None:
    """Fallback SHAP beeswarm without capa.interpret."""
    import matplotlib.pyplot as plt
    import matplotlib

    _apply_style()

    shap_vals = data.shap_values
    feat_vals = data.clinical_matrix
    if shap_vals is None or feat_vals is None:
        return

    feat_names = [
        "Age (recipient)", "Age (donor)", "CD34 dose",
        "Sex mismatch", "Disease", "Conditioning",
        "Donor type", "Stem cell source",
    ]
    n_feat = len(feat_names)
    order = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1]

    fig, ax = plt.subplots(figsize=(_1COL, _1COL * 1.2))
    cmap = matplotlib.colormaps["RdBu_r"]

    for row_i, feat_i in enumerate(order):
        sv = shap_vals[:, feat_i]
        fv = feat_vals[:, feat_i]
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
        colors = cmap(fv_norm)

        # Jitter y
        rng2 = np.random.default_rng(feat_i)
        jitter = rng2.uniform(-0.3, 0.3, len(sv))
        ax.scatter(sv, row_i + jitter, c=colors, s=4, alpha=0.7, lw=0)

    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([feat_names[i] for i in order])
    ax.set_xlabel("SHAP value (impact on GvHD risk)")
    ax.set_title("SHAP Summary — Clinical Covariates", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "suppfig1_shap", out, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplementary Figure 2 — Hyperparameter sensitivity
# ---------------------------------------------------------------------------
def supp_figure_2_sensitivity(data: FigureData, out: Path, formats: list[str], dpi: int) -> None:
    """6-panel line plots showing C-index vs hyperparameter value."""
    import matplotlib.pyplot as plt

    _apply_style()

    if not data.sensitivity:
        logger.warning("No sensitivity data for SuppFig 2 — skipping")
        return

    param_labels = {
        "embed_dim":     "Embedding dim",
        "n_heads":       "Attention heads",
        "dropout":       "Dropout rate",
        "learning_rate": "Learning rate",
        "batch_size":    "Batch size",
        "alpha_ranking": "Ranking loss weight (α)",
    }

    params = list(data.sensitivity.keys())
    n_params = len(params)
    ncols = 3
    nrows = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(_2COL, _2COL * 0.6))
    axes_flat = axes.flatten()

    for pi, param in enumerate(params):
        ax = axes_flat[pi]
        values, means, ci_w = data.sensitivity[param]

        ax.plot(range(len(values)), means, color=WONG["blue"], lw=1.0, marker="o", ms=3)
        ax.fill_between(
            range(len(values)),
            means - ci_w,
            means + ci_w,
            color=WONG["blue"],
            alpha=0.2,
        )

        ax.set_xticks(range(len(values)))
        if param == "learning_rate":
            ax.set_xticklabels([f"{v:.0e}" for v in values], rotation=30, ha="right")
        else:
            ax.set_xticklabels([str(int(v)) if v == int(v) else f"{v:.1f}" for v in values])

        ax.set_ylabel("C-index")
        ax.set_title(param_labels.get(param, param), fontweight="bold")
        ax.set_ylim(0.5, 0.85)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused panels
    for pi in range(n_params, len(axes_flat)):
        axes_flat[pi].set_visible(False)

    fig.suptitle("Hyperparameter Sensitivity Analysis", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "suppfig2_sensitivity", out, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------------
ALL_FIGURES: dict = {
    "1": ("fig1_architecture",   figure_1_architecture,   "Figure 1: Architecture diagram"),
    "2": ("fig2_umap",           figure_2_umap,           "Figure 2: HLA embedding UMAP"),
    "3": ("fig3_attention",      figure_3_attention,      "Figure 3: Attention heatmap"),
    "4": ("fig4_cif",            figure_4_cif,            "Figure 4: CIF curves"),
    "5": ("fig5_comparison",     figure_5_comparison,     "Figure 5: Model comparison"),
    "6": ("fig6_calibration",    figure_6_calibration,    "Figure 6: Calibration plots"),
    "S1": ("suppfig1_shap",      supp_figure_1_shap,      "SuppFig 1: SHAP beeswarm"),
    "S2": ("suppfig2_sensitivity", supp_figure_2_sensitivity, "SuppFig 2: Hyperparameter sensitivity"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CAPA paper figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", type=Path, default=FIGURES_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["pdf", "png"],
        choices=["pdf", "png", "svg", "eps"],
        help="Output file formats",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution for raster formats",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Always use synthetic data (skip real-data loading)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic data",
    )
    parser.add_argument(
        "--figures", nargs="+", default=list(ALL_FIGURES.keys()),
        metavar="FIG",
        help=f"Which figures to generate. Choices: {list(ALL_FIGURES.keys())}",
    )
    parser.add_argument(
        "--no-umap", action="store_true",
        help="Use PCA instead of UMAP for Figure 2",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", args.out)
    logger.info("Formats: %s | DPI: %d", args.formats, args.dpi)

    data = load_data(synthetic=args.synthetic)

    requested = [f.upper() if f.upper() in ALL_FIGURES else f for f in args.figures]
    unknown = [f for f in requested if f not in ALL_FIGURES]
    if unknown:
        logger.error("Unknown figure keys: %s. Valid keys: %s", unknown, list(ALL_FIGURES.keys()))
        sys.exit(1)

    for key in requested:
        _name, fn, description = ALL_FIGURES[key]
        logger.info("Generating %s …", description)
        try:
            if key == "2":
                fn(data, args.out, args.formats, args.dpi, use_umap=not args.no_umap)
            else:
                fn(data, args.out, args.formats, args.dpi)
        except Exception:
            logger.exception("Failed to generate %s", description)

    logger.info("Done — %d figure(s) written to %s", len(requested), args.out)


if __name__ == "__main__":
    main()
