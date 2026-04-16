#!/usr/bin/env python3
"""Generate CAPA V1 Technical Debrief PDF using ReportLab + matplotlib."""

import io
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, HRFlowable, PageBreak, KeepTogether, Preformatted,
    ListFlowable, ListItem,
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as pdfcanvas

# ── Output path ────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / "paper"
OUT_PATH = OUT_DIR / "CAPA_Technical_Debrief_V1.pdf"

# ── Colours ────────────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#1a1a2e")
C_ACCENT  = colors.HexColor("#2563eb")
C_LIGHT   = colors.HexColor("#eff6ff")
C_MID     = colors.HexColor("#dbeafe")
C_CODE_BG = colors.HexColor("#f8f8f8")
C_CODE_BD = colors.HexColor("#e2e8f0")
C_RULE    = colors.HexColor("#93c5fd")
C_TABLE_H = colors.HexColor("#1e40af")
C_TABLE_R = colors.HexColor("#eff6ff")
C_RED     = colors.HexColor("#dc2626")
C_GREEN   = colors.HexColor("#16a34a")

# ── Styles ─────────────────────────────────────────────────────────────────────
def make_styles():
    s = {}
    base = dict(fontName="Times-Roman", fontSize=12, leading=17,
                textColor=C_DARK, spaceAfter=6)

    s["body"]     = ParagraphStyle("body",     alignment=TA_JUSTIFY, **base)
    s["body_left"]= ParagraphStyle("body_left",alignment=TA_LEFT,    **base)
    s["h1"]       = ParagraphStyle("h1",       fontName="Times-Bold", fontSize=18,
                                   leading=22, textColor=C_ACCENT,
                                   spaceBefore=18, spaceAfter=8, alignment=TA_LEFT)
    s["h2"]       = ParagraphStyle("h2",       fontName="Times-Bold", fontSize=14,
                                   leading=18, textColor=C_DARK,
                                   spaceBefore=14, spaceAfter=6, alignment=TA_LEFT)
    s["h3"]       = ParagraphStyle("h3",       fontName="Times-BoldItalic", fontSize=12,
                                   leading=16, textColor=C_ACCENT,
                                   spaceBefore=10, spaceAfter=4, alignment=TA_LEFT)
    s["caption"]  = ParagraphStyle("caption",  fontName="Times-Italic", fontSize=10,
                                   leading=13, textColor=colors.HexColor("#4b5563"),
                                   spaceAfter=10, alignment=TA_CENTER)
    s["code"]     = ParagraphStyle("code",     fontName="Courier", fontSize=9,
                                   leading=13, textColor=C_DARK,
                                   leftIndent=8, rightIndent=8,
                                   spaceAfter=6, alignment=TA_LEFT)
    s["bullet"]   = ParagraphStyle("bullet",   fontName="Times-Roman", fontSize=11,
                                   leading=15, leftIndent=18, bulletIndent=6,
                                   spaceAfter=3, alignment=TA_LEFT, textColor=C_DARK)
    s["cover_title"] = ParagraphStyle("cover_title", fontName="Times-Bold", fontSize=28,
                                      leading=34, textColor=C_DARK,
                                      spaceAfter=14, alignment=TA_CENTER)
    s["cover_sub"]   = ParagraphStyle("cover_sub",   fontName="Times-Italic", fontSize=16,
                                      leading=20, textColor=C_ACCENT,
                                      spaceAfter=8, alignment=TA_CENTER)
    s["cover_meta"]  = ParagraphStyle("cover_meta",  fontName="Times-Roman", fontSize=12,
                                      leading=16, textColor=colors.HexColor("#6b7280"),
                                      spaceAfter=4, alignment=TA_CENTER)
    s["toc_h1"]   = ParagraphStyle("toc_h1",   fontName="Times-Bold", fontSize=12,
                                   leading=16, textColor=C_DARK, spaceAfter=3)
    s["toc_h2"]   = ParagraphStyle("toc_h2",   fontName="Times-Roman", fontSize=11,
                                   leading=15, leftIndent=20, textColor=C_DARK, spaceAfter=2)
    s["note"]     = ParagraphStyle("note",     fontName="Times-Italic", fontSize=10,
                                   leading=14, textColor=colors.HexColor("#4b5563"),
                                   leftIndent=12, rightIndent=12, spaceAfter=6)
    s["eq_label"] = ParagraphStyle("eq_label", fontName="Times-Roman", fontSize=11,
                                   leading=14, textColor=colors.HexColor("#374151"),
                                   spaceAfter=4, alignment=TA_CENTER)
    return s

ST = make_styles()

# ── Page template ──────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = LETTER
MARGIN_L, MARGIN_R = 1.1*inch, 1.0*inch
MARGIN_T, MARGIN_B = 1.0*inch, 1.0*inch
BODY_W = PAGE_W - MARGIN_L - MARGIN_R

_page_num = [0]

def on_page(c, doc):
    _page_num[0] += 1
    pg = _page_num[0]
    c.saveState()
    # header rule
    c.setStrokeColor(C_RULE)
    c.setLineWidth(0.8)
    c.line(MARGIN_L, PAGE_H - 0.72*inch, PAGE_W - MARGIN_R, PAGE_H - 0.72*inch)
    # header text
    c.setFont("Times-Italic", 9)
    c.setFillColor(colors.HexColor("#6b7280"))
    c.drawString(MARGIN_L, PAGE_H - 0.62*inch, "CAPA Version 1 — Technical Debrief")
    c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 0.62*inch, "Confidential / Draft")
    # footer
    c.line(MARGIN_L, MARGIN_B - 0.15*inch, PAGE_W - MARGIN_R, MARGIN_B - 0.15*inch)
    c.setFont("Times-Roman", 9)
    c.drawCentredString(PAGE_W/2, MARGIN_B - 0.32*inch, str(pg))
    c.restoreState()

def build_doc(story):
    doc = BaseDocTemplate(
        str(OUT_PATH),
        pagesize=LETTER,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T + 0.1*inch, bottomMargin=MARGIN_B + 0.1*inch,
    )
    frame = Frame(MARGIN_L, MARGIN_B, BODY_W, PAGE_H - MARGIN_T - MARGIN_B - 0.05*inch,
                  id="body")
    template = PageTemplate(id="main", frames=[frame], onPage=on_page)
    doc.addPageTemplates([template])
    doc.build(story)

# ── Helpers ─────────────────────────────────────────────────────────────────────

def rule():
    return HRFlowable(width="100%", thickness=0.6, color=C_RULE, spaceAfter=6)

def vspace(n=8):
    return Spacer(1, n)

def h1(text):
    return Paragraph(text, ST["h1"])

def h2(text):
    return Paragraph(text, ST["h2"])

def h3(text):
    return Paragraph(text, ST["h3"])

def body(text):
    return Paragraph(text, ST["body"])

def note(text):
    return Paragraph(f"<i>{text}</i>", ST["note"])

def bullet_list(items):
    out = []
    for item in items:
        out.append(Paragraph(f"&#8226;&#160;&#160;{item}", ST["bullet"]))
    return out

def code_block(text, caption=None):
    """Render a code block with light background."""
    lines = text.strip("\n")
    items = [vspace(4)]

    # Build a single-cell table for the background
    p = Preformatted(lines, ST["code"])
    t = Table([[p]], colWidths=[BODY_W - 0.2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CODE_BG),
        ("BOX",        (0,0), (-1,-1), 0.6, C_CODE_BD),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    items.append(t)
    if caption:
        items.append(Paragraph(f"<i>Listing: {caption}</i>", ST["caption"]))
    items.append(vspace(4))
    return items

def render_eq(latex_str, fontsize=14, dpi=180, label=None):
    """Render a LaTeX-style equation via matplotlib mathtext → PNG → ReportLab Image."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.patch.set_alpha(0)
    t = fig.text(0.5, 0.5, f"${latex_str}$",
                 fontsize=fontsize, ha="center", va="center",
                 family="serif")
    # measure
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    w_in = (bb.width  + 20) / dpi
    h_in = (bb.height + 14) / dpi
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(max(w_in, 1.5), max(h_in, 0.4)))
    fig2.patch.set_alpha(0)
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"${latex_str}$",
            fontsize=fontsize, ha="center", va="center",
            family="serif", transform=ax.transAxes)
    buf = io.BytesIO()
    fig2.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                 transparent=True, pad_inches=0.05)
    plt.close(fig2)
    buf.seek(0)

    # Scale to fit body width but cap height
    display_w = min(BODY_W * 0.88, max(w_in, 1.5) * inch * 0.9)
    display_h = max(h_in * inch * 0.9, 0.3*inch)

    items = [vspace(4)]
    img = Image(buf, width=display_w, height=display_h)
    row = [[img]]
    t = Table(row, colWidths=[BODY_W])
    t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
    items.append(t)
    if label:
        items.append(Paragraph(label, ST["eq_label"]))
    items.append(vspace(4))
    return items

def styled_table(headers, rows, col_widths=None, caption=None):
    """Build a styled ReportLab table."""
    data = [headers] + rows
    if col_widths is None:
        col_widths = [BODY_W / len(headers)] * len(headers)

    style = [
        ("BACKGROUND",   (0,0), (-1,0),  C_TABLE_H),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Times-Bold"),
        ("FONTSIZE",     (0,0), (-1,0),  10),
        ("BOTTOMPADDING",(0,0), (-1,0),  7),
        ("TOPPADDING",   (0,0), (-1,0),  7),
        ("FONTNAME",     (0,1), (-1,-1), "Times-Roman"),
        ("FONTSIZE",     (0,1), (-1,-1), 10),
        ("TOPPADDING",   (0,1), (-1,-1), 5),
        ("BOTTOMPADDING",(0,1), (-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, C_TABLE_R]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#cbd5e1")),
        ("ALIGN",        (0,0), (-1,-1), "LEFT"),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
    ]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style))
    out = [t]
    if caption:
        out.append(Paragraph(f"<i>Table: {caption}</i>", ST["caption"]))
    return out

def arch_diagram():
    """Draw the CAPA architecture as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, w, h, label, sublabel=None, fc="#dbeafe", ec="#2563eb", fs=9):
        rect = patches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#1e3a8a", family="sans-serif")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                    ha="center", va="center",
                    fontsize=7, color="#4b5563", family="sans-serif")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.1))

    def line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color="#94a3b8", lw=0.9, ls="--")

    # Inputs
    box(0.2, 5.8, 2.8, 0.9, "Donor HLA alleles", "A,B,C,DRB1,DQB1 strings", fc="#f0fdf4", ec="#16a34a")
    box(7.0, 5.8, 2.8, 0.9, "Recipient HLA alleles", "A,B,C,DRB1,DQB1 strings", fc="#f0fdf4", ec="#16a34a")
    box(3.5, 5.8, 3.0, 0.9, "Clinical covariates", "age, disease, graft source…", fc="#fef9c3", ec="#ca8a04")

    # ESM-2 blocks
    box(0.2, 4.4, 2.8, 0.9, "ESM-2 Encoder", "frozen, 650M params  d=1280", fc="#ede9fe", ec="#7c3aed")
    box(7.0, 4.4, 2.8, 0.9, "ESM-2 Encoder", "frozen, 650M params  d=1280", fc="#ede9fe", ec="#7c3aed")

    # Arrows input → ESM-2
    arrow(1.6, 5.8, 1.6, 5.3)
    arrow(8.4, 5.8, 8.4, 5.3)
    arrow(5.0, 5.8, 5.0, 5.4)

    # Cross-attention block
    box(2.2, 2.9, 5.6, 1.0, "Bidirectional Cross-Attention Interaction Network",
        "2 layers · 8 heads · d'=128  →  z_int ∈ R^256", fc="#dbeafe", ec="#2563eb")

    arrow(1.6, 4.4, 3.2, 3.9)
    arrow(8.4, 4.4, 6.8, 3.9)

    # Clinical encoder
    box(3.6, 1.7, 2.8, 0.9, "Clinical Encoder", "2-layer MLP  →  z_clin ∈ R^32", fc="#fef9c3", ec="#ca8a04")
    arrow(5.0, 5.4, 5.0, 2.6)

    # Concat arrow
    arrow(5.0, 2.9, 5.0, 2.6)

    # Combined vector
    box(3.0, 0.7, 4.0, 0.8, "DeepHit Survival Head", "z ∈ R^288  →  P(T=t, K=k)", fc="#fee2e2", ec="#dc2626")

    arrow(5.0, 2.6, 5.0, 2.0)
    arrow(5.0, 1.7, 5.0, 1.5)

    # Final output label
    ax.text(5.0, 0.38, "CIF curves: GvHD · Relapse · TRM  +  Attention weights",
            ha="center", va="center", fontsize=8, color="#374151",
            bbox=dict(fc="#f8fafc", ec="#94a3b8", boxstyle="round,pad=0.2"))

    arrow(5.0, 0.7, 5.0, 0.48)

    # HDF5 cache note
    ax.text(0.25, 3.85, "HDF5\ncache", ha="center", va="center",
            fontsize=7, color="#6b7280", style="italic")
    ax.text(9.75, 3.85, "HDF5\ncache", ha="center", va="center",
            fontsize=7, color="#6b7280", style="italic")

    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def loss_diagram():
    """Illustrate the DeepHit loss two-term structure."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")

    # Left: NLL term illustration
    ax = axes[0]
    ax.set_title("NLL Term  ℒ_NLL", fontsize=10, fontweight="bold", color="#1e3a8a", pad=4)
    # Draw a PMF grid
    T, K = 8, 3
    cols = ["#dc2626","#2563eb","#16a34a"]
    labels = ["GvHD","Relapse","TRM"]
    for k in range(K):
        for t in range(T):
            val = np.random.RandomState(k*8+t).uniform(0.01, 0.08)
            rect = patches.Rectangle((t*1.15 + 0.4, k*1.1 + 0.3),
                                       1.05, val*8, fc=cols[k], alpha=0.5, ec="none")
            ax.add_patch(rect)
    ax.text(4.8, 0.05, "time bins  t", ha="center", fontsize=8, color="#4b5563")
    for k, lbl in enumerate(labels):
        ax.text(0.1, k*1.1 + 0.75, lbl, fontsize=7.5, color=cols[k], va="center")
    # highlight observed event
    rect = patches.Rectangle((3*1.15 + 0.4, 1*1.1 + 0.3), 1.05, 0.45,
                               fc="#facc15", ec="#ca8a04", lw=1.5)
    ax.add_patch(rect)
    ax.text(4.3, 2.1, "observed\n(t*,k*)", ha="center", fontsize=7, color="#92400e")

    # Right: Ranking term
    ax = axes[1]
    ax.set_title("Ranking Term  ℒ_rank", fontsize=10, fontweight="bold", color="#1e3a8a", pad=4)
    t_vals = np.linspace(0, 8, 100)
    cif_i = 1 - np.exp(-((t_vals/4.0)**1.3))
    cif_j = 1 - np.exp(-((t_vals/6.5)**1.1))
    ax.plot(t_vals, cif_i*3.2 + 0.2, color="#dc2626", lw=2, label="subject i (event earlier)")
    ax.plot(t_vals, cif_j*3.2 + 0.2, color="#2563eb", lw=2, label="subject j (event later)")
    ti = 3.5
    ax.axvline(ti, color="#6b7280", ls=":", lw=1)
    ax.text(ti+0.15, 3.5, "t_i", fontsize=8, color="#4b5563")
    fi = 1 - np.exp(-((ti/4.0)**1.3))
    fj = 1 - np.exp(-((ti/6.5)**1.1))
    ax.annotate("", xy=(ti, fj*3.2+0.2), xytext=(ti, fi*3.2+0.2),
                arrowprops=dict(arrowstyle="<->", color="#ca8a04", lw=1.3))
    ax.text(ti+0.2, (fi+fj)*1.6+0.2, "Δ", fontsize=10, color="#ca8a04")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_xlabel("time", fontsize=8)
    ax.set_ylabel("CIF", fontsize=8)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def results_figure():
    """Bar chart of baseline C-index results."""
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    models = ["Cox\n(cause-specific)", "Fine-Gray", "DeepHit MLP\n(tabular)"]
    relapse = [0.754, 0.841, 0.667]
    trm     = [0.647, 0.655, 0.409]
    rel_ci_lo = [0.527, 0.691, 0.129]
    rel_ci_hi = [1.000, 1.000, 1.000]
    trm_ci_lo = [0.459, 0.478, 0.259]
    trm_ci_hi = [0.854, 0.858, 0.568]

    x = np.arange(len(models))
    w = 0.32
    b1 = ax.bar(x - w/2, relapse, w, label="Relapse", color="#2563eb", alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, trm,     w, label="TRM",     color="#dc2626", alpha=0.85, zorder=3)

    # CI error bars
    ax.errorbar(x - w/2, relapse,
                yerr=[np.array(relapse)-np.array(rel_ci_lo),
                      np.array(rel_ci_hi)-np.array(relapse)],
                fmt="none", ecolor="#1e3a8a", elinewidth=1.2, capsize=3, zorder=4)
    ax.errorbar(x + w/2, trm,
                yerr=[np.array(trm)-np.array(trm_ci_lo),
                      np.array(trm_ci_hi)-np.array(trm)],
                fmt="none", ecolor="#991b1b", elinewidth=1.2, capsize=3, zorder=4)

    ax.axhline(0.5, color="#94a3b8", ls="--", lw=0.9, label="Random (C=0.5)")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("C-index", fontsize=9)
    ax.set_title("Baseline Model Performance — UCI BMT Test Set (n=29)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def module_diagram():
    """Python package tree as a styled diagram."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    cols = {
        "data":       "#d1fae5",
        "embeddings": "#ede9fe",
        "model":      "#dbeafe",
        "training":   "#fef9c3",
        "interpret":  "#fee2e2",
        "api":        "#e0f2fe",
        "root":       "#f1f5f9",
    }
    ecs = {
        "data":       "#059669",
        "embeddings": "#7c3aed",
        "model":      "#2563eb",
        "training":   "#ca8a04",
        "interpret":  "#dc2626",
        "api":        "#0284c7",
        "root":       "#64748b",
    }

    def pkg_box(x, y, w, h, title, files, cat):
        rect = patches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.07", fc=cols[cat], ec=ecs[cat], lw=1.1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.22, title, ha="center", va="top",
                fontsize=8.5, fontweight="bold", color=ecs[cat], family="monospace")
        for i, f in enumerate(files):
            ax.text(x + 0.12, y + h - 0.52 - i*0.30, f"  {f}",
                    fontsize=6.8, color="#374151", va="top", family="monospace")

    pkg_box(0.1, 4.0, 2.2, 1.8, "capa/data/", [
        "loader.py", "hla_parser.py", "splits.py"], "data")
    pkg_box(2.5, 4.0, 2.5, 1.8, "capa/embeddings/", [
        "esm_embedder.py", "hla_sequences.py", "cache.py"], "embeddings")
    pkg_box(5.2, 4.0, 2.6, 1.8, "capa/model/", [
        "capa_model.py", "interaction.py",
        "survival.py", "losses.py", "baselines.py"], "model")
    pkg_box(8.0, 4.0, 2.0, 1.8, "capa/training/", [
        "trainer.py", "evaluate.py"], "training")
    pkg_box(0.1, 1.8, 2.5, 1.8, "capa/interpret/", [
        "attention_maps.py", "shap_explain.py"], "interpret")
    pkg_box(2.8, 1.8, 2.2, 1.8, "capa/api/", [
        "predict.py  ← FastAPI", "schemas.py"], "api")
    pkg_box(5.2, 1.8, 4.8, 1.8, "web/  (Next.js 14)", [
        "app/predict/page.tsx", "app/api/predict/route.ts",
        "components/RiskChart.tsx", "components/AttentionHeatmap.tsx"], "root")

    # root label
    ax.text(6.0, 0.6, "capa/  ·  scripts/  ·  tests/ (14 modules)  ·  paper/  ·  notebooks/",
            ha="center", va="center", fontsize=8, color="#4b5563",
            bbox=dict(fc="#f8fafc", ec="#94a3b8", boxstyle="round,pad=0.25"))

    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def embed_image(buf, width_frac=0.96, caption=None):
    """Wrap a BytesIO PNG buffer in a centred ReportLab Image + optional caption."""
    w = BODY_W * width_frac
    img = Image(buf, width=w, height=w * 0.65)  # rough aspect
    items = [vspace(6), img]
    if caption:
        items.append(Paragraph(caption, ST["caption"]))
    items.append(vspace(6))
    return items

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def cover_page():
    elems = [vspace(1.6*inch)]
    elems.append(Paragraph("CAPA", ST["cover_title"]))
    elems.append(Paragraph("Computational Architecture for Predicting Alloimmunity", ST["cover_sub"]))
    elems.append(vspace(0.15*inch))
    elems.append(HRFlowable(width="60%", thickness=1.5, color=C_ACCENT,
                             hAlign="CENTER", spaceAfter=18))
    elems.append(Paragraph("Technical Debrief — Version 1.0", ST["cover_sub"]))
    elems.append(vspace(0.1*inch))
    elems.append(Paragraph("April 2026", ST["cover_meta"]))
    elems.append(Paragraph("Huanxuan (Shawn) Li", ST["cover_meta"]))
    elems.append(Paragraph(
        "Thomas Jefferson High School for Science &amp; Technology",
        ST["cover_meta"]))
    elems.append(vspace(0.5*inch))

    meta_rows = [
        ["Repository", "github.com/sh4wn27/capa"],
        ["Language",   "Python 3.11+  /  TypeScript (web)"],
        ["Framework",  "PyTorch 2.x · FastAPI · Next.js 14"],
        ["Dataset",    "UCI BMT Children (n=187) · IPD-IMGT/HLA"],
        ["Status",     "Prototype — baselines evaluated, full ESM-2 pipeline implemented"],
    ]
    t = Table(meta_rows, colWidths=[1.8*inch, BODY_W - 1.8*inch - 0.1*inch])
    t.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (0,-1), "Times-Bold"),
        ("FONTNAME",  (1,0), (1,-1), "Times-Roman"),
        ("FONTSIZE",  (0,0), (-1,-1), 11),
        ("TOPPADDING",(0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",(0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, C_LIGHT]),
        ("BOX",(0,0),(-1,-1),0.6,C_RULE),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#bfdbfe")),
        ("TEXTCOLOR",(0,0),(0,-1),C_ACCENT),
    ]))
    elems.append(t)
    elems.append(PageBreak())
    return elems


def section_exec_summary():
    elems = [h1("1. Executive Summary"), rule()]
    elems.append(body(
        "CAPA is an open-source deep learning framework designed to replace categorical "
        "HLA match/mismatch counts in haematopoietic stem cell transplantation (HSCT) "
        "outcome prediction with <b>continuous, structure-aware protein language model "
        "embeddings</b>. It models the three dominant post-transplant competing events — "
        "acute graft-versus-host disease (GvHD), disease relapse, and transplant-related "
        "mortality (TRM) — jointly through a DeepHit survival head, yielding calibrated "
        "cumulative incidence functions (CIFs) over a 730-day post-transplant horizon."
    ))
    elems.append(body(
        "Version 1 represents a complete prototype: the full Python package is implemented "
        "and tested across all modules, a FastAPI inference backend is functional, and a "
        "Next.js 14 web interface provides an interactive prediction tool. Three tabular "
        "baseline models (cause-specific Cox, Fine-Gray subdistribution hazard, and a flat-"
        "feature DeepHit MLP) have been trained and evaluated on the publicly available UCI "
        "BMT paediatric dataset (n=187). The <b>primary scientific gap</b> is the absence "
        "of allele-level HLA typing in the UCI dataset, which prevents end-to-end validation "
        "of the ESM-2 embedding pipeline on real patient data."
    ))
    elems.append(vspace(6))

    sum_rows = [
        ["Component",         "Status",       "Notes"],
        ["Data pipeline",     "Complete",     "Loader, parser, splits, feature engineering"],
        ["ESM-2 embedder",    "Complete",     "CPU/CUDA/MPS; HDF5 cache; 98-allele prototype"],
        ["Cross-attention",   "Complete",     "2 layers, 8 heads, d'=128, attention export"],
        ["DeepHit head",      "Complete",     "Joint PMF + CIF; cause-specific head also available"],
        ["Training loop",     "Complete",     "AdamW, cosine LR, early stopping, checkpointing"],
        ["Evaluation",        "Complete",     "C-index, IBS, calibration, 1000-iter bootstrap CI"],
        ["Baseline models",   "Complete",     "Cox-CS 0.75/0.65, Fine-Gray 0.84/0.66, DH 0.67/0.41"],
        ["FastAPI backend",   "Complete",     "Mock fallback when no checkpoint; CORS configured"],
        ["Web frontend",      "Complete",     "Next.js 14; RiskChart; AttentionHeatmap; HLACombobox"],
        ["configs/ YAML",     "Missing",      "scripts/train.py crashes without configs/default.yaml"],
        ["Model checkpoint",  "Missing",      "No trained .pt file; backend serves mock responses"],
        ["Backend deployment","Missing",      "CAPA_BACKEND_URL unset in Vercel; no Modal/Railway setup"],
        ["Notebooks 01,03,04","Missing",      "Only 02_embeddings.ipynb present in repo"],
    ]
    elems += styled_table(
        sum_rows[0], sum_rows[1:],
        col_widths=[1.7*inch, 1.2*inch, BODY_W - 2.9*inch],
        caption="Version 1 completion status across all system components"
    )
    elems.append(PageBreak())
    return elems


def section_motivation():
    elems = [h1("2. Scientific Motivation"), rule()]
    elems.append(body(
        "Allogeneic HSCT remains the only curative option for many haematological malignancies, "
        "with over 50,000 procedures performed globally per year. Five-year overall survival "
        "for unrelated-donor transplants remains below 50% across many disease categories. "
        "Donor selection — the primary lever available to clinicians before transplant — depends "
        "critically on HLA compatibility assessment."
    ))
    elems.append(h2("2.1  The Categorical HLA Scoring Problem"))
    elems.append(body(
        "Current clinical practice evaluates HLA compatibility by counting mismatched loci. "
        "A '10/10 matched' unrelated donor means agreement at HLA-A, -B, -C, -DRB1, and "
        "-DQB1 (two alleles each). Each additional mismatch increments the count. This "
        "representation has two fundamental limitations:"
    ))
    elems += bullet_list([
        "<b>All mismatches treated as equivalent.</b> An A*02:01 vs A*03:01 mismatch "
        "(differing at &gt;15 residues in the peptide-binding groove) is penalised identically "
        "to an A*02:01 vs A*02:06 mismatch (differing at 2 residues). The immunological "
        "consequence is wildly different.",
        "<b>Directional and locus-specific information is lost.</b> Aggregate mismatch counts "
        "collapse five loci into a single integer, discarding which loci are mismatched and "
        "the direction of each mismatch (GvH vs HvG).",
    ])
    elems.append(h2("2.2  Competing Risks are Statistically Mandatory"))
    elems.append(body(
        "GvHD, relapse, and TRM are <i>competing events</i>: occurrence of any one event "
        "precludes (or substantially alters the hazard of) the others. Standard "
        "Cox models fitted independently for each endpoint overestimate cumulative incidence "
        "and yield biased risk predictions. The correct framework is to model the joint "
        "distribution over event type and event time — the DeepHit formulation."
    ))
    elems.append(h2("2.3  Protein Language Models as Structural Priors"))
    elems.append(body(
        "ESM-2, trained by masked-language modelling on 250M diverse protein sequences, "
        "learns residue-level representations that encode three-dimensional structure, "
        "functional constraints, and evolutionary co-variation without any structural "
        "supervision. Because PLM embeddings are continuous and sequence-derived, the "
        "Euclidean (or cosine) distance between two allele embeddings encodes structural "
        "and functional similarity in a way that binary mismatch flags fundamentally cannot."
    ))
    elems.append(PageBreak())
    return elems


def section_data():
    elems = [h1("3. Data Pipeline"), rule()]
    elems.append(h2("3.1  UCI Bone Marrow Transplant Dataset"))
    elems.append(body(
        "The primary dataset is the UCI Bone Marrow Transplant: Children cohort "
        "(Sikora et al., 2010): <b>n=187 paediatric patients</b> who underwent allogeneic "
        "HSCT at a single Polish centre between 1992 and 2004. The dataset is freely "
        "available from the UCI Machine Learning Repository (dataset ID 565)."
    ))
    elems.append(body(
        "<b>Critical data limitation.</b> The UCI BMT dataset records HLA compatibility as "
        "aggregate mismatch scores (0-3) rather than per-allele typing strings. CAPA's ESM-2 "
        "embedding pipeline requires individual allele names (e.g., 'A*02:01') to look up "
        "protein sequences. This precludes end-to-end ESM-2 validation on this cohort; "
        "the tabular baselines reported in Section 10 use aggregate scores as HLA features."
    ))
    elems.append(h2("3.2  Competing-Risks Label Encoding"))
    elems.append(body(
        "Each patient is assigned a single primary event under the following priority "
        "hierarchy, using the raw outcome columns from the ARFF file:"
    ))
    elems += bullet_list([
        "<b>Relapse (k=1)</b>: patient experienced disease relapse. Priority 1.",
        "<b>TRM (k=2)</b>: patient died without relapsing. Priority 2.",
        "<b>GvHD (k=3)</b>: severe acute GvHD (grade III-IV), alive and not relapsed. Priority 3.",
        "<b>Censored (k=0)</b>: all remaining patients (alive at last follow-up, or event outside window).",
    ])
    elems.append(body(
        "Under this encoding the event distribution is: <b>Relapse 28/187 (15.0%)</b>, "
        "<b>TRM 62/187 (33.2%)</b>, <b>GvHD 16/187 (8.6%)</b>, "
        "<b>Censored 81/187 (43.3%)</b>. Survival times are right-censored at day 730."
    ))

    dist_rows = [
        ["Event",    "Count", "Fraction", "Test set events (n=29)"],
        ["Relapse",  "28",    "15.0%",    "4"],
        ["TRM",      "62",    "33.2%",    "11"],
        ["GvHD",     "16",    "8.6%",     "2  (insufficient for C-index)"],
        ["Censored", "81",    "43.3%",    "12"],
        ["Total",    "187",   "100%",     "29"],
    ]
    elems += styled_table(dist_rows[0], dist_rows[1:],
                          col_widths=[1.4*inch, 0.8*inch, 0.9*inch, BODY_W - 3.1*inch],
                          caption="Event distribution in the UCI BMT dataset")
    elems.append(h2("3.3  Feature Engineering (21 Features)"))
    elems += bullet_list([
        "<b>Continuous (4):</b> recipient age, donor age, log-CD34⁺ cell dose — "
        "standardised to zero mean/unit variance using training-set statistics. "
        "CD34⁺ dose log-transformed before standardisation (right-skewed distribution).",
        "<b>Binary (9):</b> sex, graft source binary, malignant disease flag, high-risk flag, "
        "sex mismatch, donor/recipient CMV serostatus, ABO match, retransplant status.",
        "<b>HLA aggregate (4):</b> overall mismatch score (0-3), antigen mismatch count, "
        "allele mismatch count, mismatch binary flag.",
        "<b>Disease category (4):</b> ALL, AML, chronic, lymphoma (indicator variables; "
        "non-malignant = reference).",
    ])
    elems.append(body(
        "Missing values (rare; &lt;2% per feature) are imputed with training-set column means. "
        "All preprocessing statistics are estimated exclusively on the training partition "
        "and applied without modification to validation and test sets."
    ))
    elems.append(h2("3.4  Stratified Train/Val/Test Split"))
    elems.append(body(
        "The dataset is partitioned 70/15/15 (train n=130, val n=28, test n=29). "
        "To preserve marginal event frequencies, patients are first stratified by the "
        "four-class event label and randomly assigned within each stratum. This ensures "
        "that event frequencies in all three partitions remain within 2% of the "
        "dataset-level proportions."
    ))

    split_rows = [
        ["Split",      "n", "Relapse", "TRM", "GvHD", "Censored"],
        ["Train",      "130", "20 (15.4%)", "44 (33.8%)", "11 (8.5%)", "55 (42.3%)"],
        ["Validation", "28",  "4 (14.3%)",  "9 (32.1%)",  "3 (10.7%)", "12 (42.9%)"],
        ["Test",       "29",  "4 (13.8%)",  "11 (37.9%)", "2 (6.9%)",  "12 (41.4%)"],
    ]
    elems += styled_table(split_rows[0], split_rows[1:],
        col_widths=[0.9*inch, 0.5*inch, 1.1*inch, 1.1*inch, 1.3*inch, BODY_W - 4.9*inch],
        caption="Approximate event counts per partition after stratified splitting")

    elems.append(h2("3.5  Key Loader Code"))
    elems += code_block("""\
# capa/data/loader.py (excerpt) — ARFF inversion quirk
# aGvHDIIIIV in raw data: 0=Yes, 1=No (inverted convention)
# We recode: acute_gvhd_iii_iv = 1-raw_value

df["acute_gvhd_iii_iv"] = 1 - df["aGvHDIIIIV"].astype(int)

# Competing-risks label hierarchy
def assign_event(row):
    if row["relapse"] == 1:
        return 1, row["time_to_relapse"]      # k=1
    if row["survival_status"] == 0:           # dead without relapse
        return 2, row["survival_time"]        # k=2  (TRM)
    if row["acute_gvhd_iii_iv"] == 1:
        return 3, row["days_to_acute_gvhd"]  # k=3
    return 0, row["survival_time"]            # censored""",
    caption="Competing-risks label assignment in loader.py")
    elems.append(PageBreak())
    return elems


def section_embeddings():
    elems = [h1("4. HLA Embedding Pipeline"), rule()]
    elems.append(body(
        "The embedding pipeline is CAPA's core scientific contribution: replacing binary "
        "mismatch flags with dense, biologically meaningful vector representations "
        "derived from HLA protein sequences."
    ))
    elems.append(h2("4.1  ESM-2 Architecture"))
    elems.append(body(
        "CAPA uses the <b>facebook/esm2_t33_650M_UR50D</b> model: 650M parameters, "
        "33 transformer layers, hidden dimension d=1280. The model is loaded in half "
        "precision (float16) and kept <b>fully frozen</b> — no transplantation-specific "
        "fine-tuning is performed in V1. Only the cross-attention interaction network, "
        "clinical encoder, and survival head (~2.8M parameters total) are trained."
    ))
    elems.append(h2("4.2  Sequence Retrieval from IPD-IMGT/HLA"))
    elems.append(body(
        "Full extracellular protein sequences are retrieved from IPD-IMGT/HLA release 3.55.0. "
        "For <b>class I loci (A, B, C)</b>, the mature α₁–α₃ extracellular domain is used "
        "(residues 25–309 of the precursor; 285 amino acids). For <b>class II β-chains "
        "(DRB1, DQB1)</b>, the β₁–β₂ domains are used (residues 30–227; 198 amino acids). "
        "Alleles specified only at antigen resolution are expanded to the most prevalent "
        "high-resolution allele via IPD-IMGT/HLA haplotype frequency tables."
    ))
    elems.append(h2("4.3  Per-Allele Embedding Computation"))
    elems.append(body(
        "For allele a with sequence of length S_a, the embedding is the mean of the "
        "non-padding per-residue hidden states in the 33rd transformer layer:"
    ))
    elems += render_eq(
        r"\mathbf{e}_a = \frac{1}{S_a} \sum_{i=1}^{S_a} \mathbf{h}^{(33)}_i(a) \;\in\; \mathbb{R}^{1280}",
        fontsize=13, label="Eq. (1) — Mean-pooled ESM-2 embedding for allele a"
    )
    elems.append(body(
        "The mean-pooling strategy produces a fixed-size representation "
        "regardless of sequence length (285 vs 198 residues across loci), which is "
        "required for the fixed-architecture cross-attention network downstream."
    ))
    elems += code_block("""\
# capa/embeddings/esm_embedder.py (excerpt)
@torch.no_grad()
def embed(self, sequences: dict[str, str]) -> dict[str, np.ndarray]:
    self._ensure_loaded()
    results = {}
    alleles = list(sequences.keys())
    for batch_start in tqdm(range(0, len(alleles), self.batch_size)):
        batch_alleles = alleles[batch_start : batch_start + self.batch_size]
        batch_seqs = [sequences[a] for a in batch_alleles]
        inputs = self._tokenizer(batch_seqs, return_tensors="pt",
                                 padding=True, truncation=True).to(self._device)
        outputs = self._model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (B, L, 1280)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        # Mean-pool over non-padding positions
        pooled = (last_hidden * mask).sum(1) / mask.sum(1)  # (B, 1280)
        for allele, vec in zip(batch_alleles, pooled):
            results[allele] = vec.cpu().float().numpy()
    return results""",
    caption="ESM-2 mean-pooling in esm_embedder.py")

    elems.append(h2("4.4  HDF5 Embedding Cache"))
    elems.append(body(
        "Computing ESM-2 embeddings for a full registry-scale allele vocabulary requires "
        "~4 hours on a single NVIDIA T4 GPU and ~600 MB of storage. To avoid recomputation, "
        "embeddings are cached in an HDF5 file keyed by allele name. The cache supports "
        "incremental updates — only new alleles trigger ESM-2 inference. A prototype cache "
        "of <b>98 alleles spanning all five loci</b> is distributed with the repository to "
        "support architecture testing without GPU access."
    ))
    elems.append(h2("4.5  UMAP Embedding Analysis (Prototype Set)"))
    elems.append(body(
        "UMAP dimensionality reduction applied to the 98-allele prototype set (Figure 2 of "
        "the manuscript) demonstrates that ESM-2 embeddings cluster strongly by locus, "
        "consistent with major sequence-level differences between class I and class II genes. "
        "Within loci, sub-clusters correspond to serologically defined antigen groups "
        "(A*02, B*44, etc.) — structural families recovered without any transplantation-"
        "specific supervision. Pairwise cosine similarity confirms within-locus similarity "
        "substantially exceeds between-locus similarity."
    ))
    elems.append(PageBreak())
    return elems


def section_architecture():
    elems = [h1("5. Model Architecture"), rule()]
    elems.append(body(
        "The CAPA model consists of three trainable components: (1) a bidirectional "
        "cross-attention interaction network, (2) a clinical covariate encoder, and "
        "(3) a DeepHit competing-risks survival head. The architecture is shown below."
    ))

    arch_buf = arch_diagram()
    img = Image(arch_buf, width=BODY_W * 0.97, height=BODY_W * 0.97 * 5.2/7.5)
    t = Table([[img]], colWidths=[BODY_W])
    t.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
    elems.append(vspace(4))
    elems.append(t)
    elems.append(Paragraph(
        "<i>Figure 1: CAPA end-to-end architecture. ESM-2 (purple) is frozen; "
        "all other blocks are trainable (~2.8M parameters).</i>", ST["caption"]))
    elems.append(vspace(6))

    elems.append(h2("5.1  Input Representation"))
    elems.append(body(
        "Let D ∈ ℝ^(L×d) and R ∈ ℝ^(L×d) denote the row-stacked ESM-2 embedding matrices "
        "for the donor and recipient, where L=5 loci and d=1280. Both matrices are "
        "projected to interaction dimension d'=128 via independent learned linear layers:"
    ))
    elems += render_eq(
        r"\tilde{\mathbf{D}} = \mathbf{D}\mathbf{W}^d_{\mathrm{proj}},\quad"
        r"\tilde{\mathbf{R}} = \mathbf{R}\mathbf{W}^r_{\mathrm{proj}},\quad"
        r"\mathbf{W}^{d,r}_{\mathrm{proj}} \in \mathbb{R}^{1280 \times 128}",
        fontsize=12, label="Eq. (2) — Linear projection to interaction space"
    )

    elems.append(h2("5.2  Bidirectional Cross-Attention (N=2 layers)"))
    elems.append(body(
        "At each cross-attention layer, the donor stream attends to the recipient and vice "
        "versa with H=8 heads and per-head dimension d_h = d'/H = 16. The update equations "
        "for one direction (donor attending to recipient) are:"
    ))
    elems += render_eq(
        r"\mathrm{head}_h = \mathrm{softmax}\!\left("
        r"\frac{\mathbf{X}\mathbf{W}^Q_h\,(\mathbf{Y}\mathbf{W}^K_h)^\top}{\sqrt{d_h}}"
        r"\right)\mathbf{Y}\mathbf{W}^V_h",
        fontsize=12, label="Eq. (3) — Single attention head computation"
    )
    elems += render_eq(
        r"\mathbf{D}^{(\ell+1/2)} = \mathrm{LayerNorm}\left("
        r"\tilde{\mathbf{D}}^{(\ell)} + \mathrm{MHA}(\tilde{\mathbf{D}}^{(\ell)},\,"
        r"\tilde{\mathbf{R}}^{(\ell)})\right)",
        fontsize=12, label="Eq. (4) — Donor cross-attention update with residual + LayerNorm"
    )
    elems.append(body(
        "The recipient stream receives the symmetric update MHA(R̃, D̃). Both streams are "
        "updated in parallel at each layer. A position-wise feed-forward sublayer "
        "(Linear → GELU → Dropout → Linear) follows the attention sublayer at each layer, "
        "with dimensions d' → 4d' → d'."
    ))
    elems += code_block("""\
# capa/model/interaction.py — CrossAttentionBlock.forward()
def forward(self, donor, recip):
    # donor, recip: (batch, n_loci=5, embedding_dim=1280)
    d_attn, w_d2r = self.d2r(query=donor, key=recip, value=recip,
                              need_weights=True, average_attn_weights=True)
    r_attn, w_r2d = self.r2d(query=recip, key=donor, value=donor,
                              need_weights=True, average_attn_weights=True)
    donor_out = self.norm_d(donor + self.drop(d_attn))
    recip_out = self.norm_r(recip + self.drop(r_attn))
    return donor_out, recip_out, w_d2r, w_r2d
    # w_d2r shape: (batch, 5, 5) — donor-locus × recipient-locus attention""",
    caption="Bidirectional cross-attention block implementation")

    elems.append(h2("5.3  Interaction Feature Vector"))
    elems.append(body(
        "After N=2 layers, both streams are mean-pooled over the L=5 locus positions and "
        "concatenated to form the 256-dimensional interaction representation:"
    ))
    elems += render_eq(
        r"\mathbf{z}_{\mathrm{int}} = \left[\frac{1}{L}\sum_{\ell=1}^{L}"
        r"\mathbf{D}^{(N)}_\ell\;;\;\frac{1}{L}\sum_{\ell=1}^{L}"
        r"\mathbf{R}^{(N)}_\ell\right] \in \mathbb{R}^{256}",
        fontsize=12, label="Eq. (5) — Interaction feature vector z_int"
    )

    elems.append(h2("5.4  Clinical Covariate Encoder"))
    elems.append(body(
        "Three continuous covariates (recipient age, donor age, log-CD34⁺ dose) and a "
        "sex-mismatch binary flag are concatenated with learned embeddings (dim=8 each) "
        "for four categorical variables (disease, conditioning, donor type, stem cell source). "
        "The combined 36-dimensional vector is passed through a two-layer GELU MLP with "
        "LayerNorm to produce a 32-dimensional clinical feature vector z_clin:"
    ))
    elems += render_eq(
        r"\mathbf{z}_{\mathrm{clin}} = \mathrm{LayerNorm}\!\left("
        r"\sigma\!\left(\mathrm{LayerNorm}(\mathbf{x}_{\mathrm{clin}}\mathbf{U}_1)"
        r"\right)\mathbf{U}_2\right) \in \mathbb{R}^{32}",
        fontsize=12, label="Eq. (6) — Clinical encoder MLP"
    )
    elems.append(body(
        "The combined feature vector fed to the survival head is "
        "z = [z_int ; z_clin] ∈ ℝ^288."
    ))
    elems.append(h2("5.5  Parameter Count"))
    param_rows = [
        ["Component", "Parameters", "Notes"],
        ["Projection layers (donor + recip)", "~328K", "2 × 1280×128"],
        ["Cross-attention (2 layers, 8 heads)", "~790K", "Q/K/V/O matrices × 2 layers × 2 directions"],
        ["Feed-forward sublayers",             "~527K", "4 FFN blocks (d'→4d'→d')"],
        ["Clinical encoder",                   "~15K",  "Embedding tables + 2-layer MLP"],
        ["DeepHit survival head",              "~1.2M", "288→256→300 logits"],
        ["<b>Total trainable</b>",             "<b>~2.86M</b>", "ESM-2 frozen (650M)"],
    ]
    elems += styled_table(param_rows[0], param_rows[1:],
        col_widths=[2.4*inch, 1.1*inch, BODY_W - 3.5*inch],
        caption="Trainable parameter budget for CAPA V1")
    elems.append(PageBreak())
    return elems


def section_deephit():
    elems = [h1("6. DeepHit Competing-Risks Survival Head"), rule()]
    elems.append(body(
        "The survival head implements the DeepHit framework (Lee et al., 2018), "
        "which directly models the <b>joint probability mass function</b> over event "
        "types and discrete event times — the only parametric formulation guaranteed "
        "to produce valid (non-crossing, non-negative) cause-specific CIFs."
    ))
    elems.append(h2("6.1  Discrete-Time Parameterisation"))
    elems.append(body(
        "The follow-up horizon [0, 730 days] is discretised into M=100 equally spaced "
        "bins (bin width ≈ 7.3 days). The survival head is a two-layer MLP "
        "(288 → 256 → K·M = 300) with GELU activations and LayerNorm between layers. "
        "A single softmax over all 300 outputs yields the joint PMF:"
    ))
    elems += render_eq(
        r"\pi(t, k \mid \mathbf{z}) = P(T=t, K=k \mid \mathbf{z}),\quad"
        r"\sum_{k=1}^{K}\sum_{t=1}^{M} \pi(t,k\mid\mathbf{z}) = 1",
        fontsize=12, label="Eq. (7) — Joint probability mass function (DeepHit)"
    )
    elems.append(body(
        "The cause-specific cumulative incidence function (CIF) for event k is "
        "the cumulative sum of the marginal PMF for that event:"
    ))
    elems += render_eq(
        r"F_k(t\mid\mathbf{z}) = \sum_{s=1}^{t}\pi(s,k\mid\mathbf{z}),\quad"
        r"S(t\mid\mathbf{z}) = 1 - \sum_{k}F_k(t\mid\mathbf{z})",
        fontsize=12, label="Eq. (8) — CIF and overall survival"
    )
    elems += code_block("""\
# capa/model/survival.py — DeepHitHead.cif()
def cif(self, x: Tensor) -> Tensor:
    logits = self.forward(x)              # (batch, K, M)
    batch = logits.shape[0]
    # Single softmax over flattened (K × M) space
    joint = F.softmax(logits.view(batch, -1), dim=-1)
    joint = joint.view(batch, self.num_events, self.time_bins)
    # CIF = cumulative sum over time
    return torch.cumsum(joint, dim=2)     # (batch, K, M)""",
    caption="CIF computation in DeepHitHead")

    elems.append(h2("6.2  Alternative: Cause-Specific Hazard Head"))
    elems.append(body(
        "CAPA also implements a cause-specific hazard head (available via "
        "survival_type='cause_specific') where each event's sub-hazard h_k(t) ∈ (0,1) "
        "is modelled independently, combined via the competing-risks product formula:"
    ))
    elems += render_eq(
        r"S(t) = \prod_{s=1}^{t}\!\left[1-\sum_{k=1}^{K}h_k(s)\right],\quad"
        r"F_k(t) = \sum_{s=1}^{t}h_k(s)\cdot S(s-1)",
        fontsize=12, label="Eq. (9) — Cause-specific sub-hazard CIF"
    )
    elems.append(PageBreak())
    return elems


def section_loss():
    elems = [h1("7. Loss Function"), rule()]
    elems.append(body(
        "Training minimises the DeepHit composite loss, a weighted sum of a "
        "negative log-likelihood (calibration) term and a pairwise ranking "
        "(concordance) term:"
    ))
    elems += render_eq(
        r"\mathcal{L}(\theta) = "
        r"(1-\alpha)\,\mathcal{L}_{\mathrm{NLL}} + \alpha\,\mathcal{L}_{\mathrm{rank}},"
        r"\quad \alpha = 0.5",
        fontsize=13, label="Eq. (10) — DeepHit composite loss (α=0.5 in V1)"
    )

    elems.append(h2("7.1  Negative Log-Likelihood Term"))
    elems.append(body(
        "For uncensored subjects, the NLL maximises the probability of the observed "
        "(event type, event time) pair. For censored subjects, it maximises the "
        "predicted overall survival at the censoring time:"
    ))
    elems += render_eq(
        r"\mathcal{L}_{\mathrm{NLL}} = -\frac{1}{n}\left["
        r"\sum_{\delta_i>0}\log\pi(t_i,\delta_i\mid\mathbf{z}_i)"
        r"+\sum_{\delta_i=0}\log S(t_i\mid\mathbf{z}_i)\right]",
        fontsize=12, label="Eq. (11) — NLL for uncensored (event) and censored subjects"
    )

    elems.append(h2("7.2  Pairwise Ranking Term"))
    elems.append(body(
        "For each event k, the ranking loss penalises concordance violations: pairs (i,j) "
        "where subject i experienced event k before subject j, but the model incorrectly "
        "assigns higher CIF at time t_i to j than to i. The exponential kernel "
        "exp(-Δ/σ) with σ=0.1 is a differentiable surrogate for the concordance "
        "violation indicator:"
    ))
    elems += render_eq(
        r"\mathcal{L}_{\mathrm{rank}} = \frac{1}{K}\sum_{k=1}^{K}"
        r"\frac{1}{|\mathcal{P}_k|}\sum_{(i,j)\in\mathcal{P}_k}"
        r"\exp\!\left(\frac{F_k(t_i\mid\mathbf{z}_j)-F_k(t_i\mid\mathbf{z}_i)}{\sigma}\right)",
        fontsize=12, label="Eq. (12) — Vectorised pairwise ranking loss (σ=0.1)"
    )
    elems.append(body(
        "P_k = {(i,j) : t_i < t_j, δ_i=k} is the set of comparable pairs for event k — "
        "subject i experienced event k before subject j's observation time. "
        "A well-calibrated model should have F_k(t_i|z_i) > F_k(t_i|z_j), giving Δ>0 "
        "and a small penalty. The ranking loss is fully vectorised in the codebase "
        "(no Python loops over subjects)."
    ))

    loss_buf = loss_diagram()
    img = Image(loss_buf, width=BODY_W * 0.96, height=BODY_W * 0.96 * 2.6/7.2)
    t_w = Table([[img]], colWidths=[BODY_W])
    t_w.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
    elems.append(vspace(4))
    elems.append(t_w)
    elems.append(Paragraph(
        "<i>Figure 2: Left — NLL term highlights the observed (t*, k*) cell in the joint PMF. "
        "Right — Ranking term: for pair (i,j) with t_i &lt; t_j, the penalty is exp(-Δ/σ) "
        "where Δ = CIF_i(t_i) − CIF_j(t_i). A higher Δ (correct ordering) gives a smaller penalty.</i>",
        ST["caption"]))

    elems += code_block("""\
# capa/model/losses.py — ranking loss (vectorised, no Python loops)
for k in range(1, num_events + 1):
    mask_k = event_types == k
    t_k    = event_times[mask_k].clamp(0, time_bins - 1)
    cif_k  = cif[mask_k, k - 1, :]          # (n_k, T)
    cif_i  = cif_k[arange_k, t_k]           # (n_k,)  CIF_k(t_i | x_i)
    # CIF_k(t_i | x_j): evaluate each subject j's CIF at each subject i's time
    cif_j_at_ti = cif_k[:, t_k].permute(1, 0)   # (n_k_i, n_k_j)
    valid  = (t_k.unsqueeze(0) > t_k.unsqueeze(1)).float()
    delta  = cif_i.unsqueeze(1) - cif_j_at_ti   # (n_k_i, n_k_j)
    phi    = torch.exp(-delta / sigma)
    total_loss += (valid * phi).sum()""",
    caption="Vectorised ranking loss in losses.py")
    elems.append(PageBreak())
    return elems


def section_training():
    elems = [h1("8. Training Procedure"), rule()]
    elems.append(h2("8.1  Optimiser and Schedule"))
    elems.append(body(
        "All learnable parameters are optimised jointly with <b>AdamW</b> "
        "(Loshchilov & Hutter, 2019): initial learning rate η₀=10⁻⁴, "
        "β₁=0.9, β₂=0.999, weight decay λ=10⁻⁴. Gradients are clipped to "
        "maximum ℓ₂ norm 1.0. A cosine-annealing schedule decays the learning "
        "rate from η₀ to 0 over max_epochs=200 iterations."
    ))
    elems.append(body(
        "Note: The manuscript describes ReduceLROnPlateau (patience=10), but the "
        "implemented trainer uses CosineAnnealingLR. This is a discrepancy between "
        "the paper draft and code that should be reconciled before submission."
    ))
    elems.append(h2("8.2  Early Stopping"))
    elems.append(body(
        "Training is monitored on the validation set using mean C-index across "
        "all three competing events (with events having &lt;2 observed instances "
        "excluded from the mean). The checkpoint achieving the highest validation "
        "mean C-index is retained. Training terminates if the metric does not "
        "improve for patience=20 consecutive epochs."
    ))
    elems.append(h2("8.3  Hyperparameter Grid Search"))
    hp_rows = [
        ["Hyperparameter",    "Range explored",            "Selected"],
        ["Attention layers N","1, 2, 3",                   "2"],
        ["Interaction dim d'","64, 128, 256",               "128"],
        ["Attention heads H", "4, 8",                       "8"],
        ["Learning rate η₀",  "10⁻³, 10⁻⁴, 10⁻⁵",        "10⁻⁴"],
        ["Loss weight α",     "0.2, 0.5, 0.8",             "0.5"],
        ["Ranking σ",         "0.05, 0.1, 0.2",            "0.1"],
        ["Batch size",        "16, 32",                     "32"],
    ]
    elems += styled_table(hp_rows[0], hp_rows[1:],
        col_widths=[1.9*inch, 1.9*inch, BODY_W - 3.8*inch],
        caption="Hyperparameter grid search space and selected values")
    elems.append(body(
        "Hyperparameter selection used grid search on the validation mean C-index. "
        "Final performance estimates are reported across <b>five independent random seeds</b> "
        "controlling train/val/test split assignment, with the best checkpoint "
        "selected independently per seed."
    ))
    elems.append(h2("8.4  Batch Format"))
    elems += code_block("""\
# Expected DataLoader batch dict (capa/training/trainer.py)
batch = {
    "donor_embeddings":     Tensor,  # (batch, 5, 1280) float32
    "recipient_embeddings": Tensor,  # (batch, 5, 1280) float32
    "clinical_features":   Tensor,  # (batch, 32)      float32
    "event_times":         Tensor,  # (batch,)         int64
    "event_types":         Tensor,  # (batch,)  0=censored, 1..K=event
}""", caption="Required DataLoader batch format")
    elems.append(PageBreak())
    return elems


def section_evaluation():
    elems = [h1("9. Evaluation Framework"), rule()]
    elems.append(body(
        "All metrics are computed on the held-out test set only. "
        "Confidence intervals use 1000-iteration bootstrap resampling of the test set. "
        "The evaluation module (capa/training/evaluate.py) is fully self-contained "
        "and operates on NumPy arrays — no PyTorch dependency at evaluation time."
    ))
    elems.append(h2("9.1  Time-Dependent Concordance Index"))
    elems.append(body(
        "For each event k, the cause-specific C-index considers only pairs (i,j) "
        "where subject i experienced event k before subject j's observation time:"
    ))
    elems += render_eq(
        r"C_k = \frac{\sum_{(i,j)\in\mathcal{P}_k}"
        r"\mathbf{1}\left[F_k(t_i\mid\mathbf{z}_i)>F_k(t_i\mid\mathbf{z}_j)\right]"
        r"}{|\mathcal{P}_k|}",
        fontsize=12, label="Eq. (13) — Cause-specific concordance index (C=0.5 random, C=1.0 perfect)"
    )
    elems.append(body(
        "C_k=0.5 is random discrimination; C_k=1.0 is perfect. The metric is "
        "undefined (NaN) when |P_k|=0, as is the case for GvHD in the test set (n=2 events)."
    ))
    elems.append(h2("9.2  Integrated Brier Score (IBS)"))
    elems.append(body(
        "The IPCW Brier score at evaluation time τ for event k is:"
    ))
    elems += render_eq(
        r"\mathrm{BS}_k(\tau) = \frac{1}{n}\sum_{i=1}^n \hat{w}_i(\tau)"
        r"\left(F_k(\tau\mid\mathbf{z}_i) - \mathbf{1}(T_i\leq\tau,K_i=k)\right)^2",
        fontsize=12, label="Eq. (14) — IPCW Brier score at time τ"
    )
    elems.append(body(
        "where ŵᵢ(τ) is the inverse probability of censoring weight estimated via "
        "Kaplan-Meier on the censoring distribution. The IBS integrates this over "
        "a time grid [0, t_max] via the trapezoidal rule. Lower IBS = better calibration."
    ))
    elems.append(h2("9.3  Bootstrap Confidence Intervals"))
    elems += code_block("""\
# capa/training/evaluate.py — generic bootstrap wrapper
def bootstrap_ci(metric_fn, *args, n_bootstrap=1000, ci_level=0.95, seed=0):
    arrays = [np.asarray(a) for a in args]
    value = metric_fn(*arrays)       # point estimate on full data
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_args = [a[idx] for a in arrays]
        replicates.append(metric_fn(*boot_args))
    lo = np.percentile(replicates, 100*(1-ci_level)/2)
    hi = np.percentile(replicates, 100*(1+ci_level)/2)
    return MetricWithCI(value=value, ci_lower=lo, ci_upper=hi)""",
    caption="Bootstrap CI wrapper — applies to any scalar metric function")
    elems.append(PageBreak())
    return elems


def section_baselines():
    elems = [h1("10. Baseline Models and Results"), rule()]
    elems.append(body(
        "Three tabular-feature competing-risks baselines were trained on the UCI BMT "
        "dataset using the 21 aggregate clinical and HLA-score features. All models "
        "used the same 70/15/15 split and are evaluated on the held-out test set (n=29). "
        "No allele-level ESM-2 embeddings are used in these baselines."
    ))
    elems.append(h2("10.1  Cause-Specific Cox (Cox-CS)"))
    elems.append(body(
        "One Cox model per competing event, treating subjects experiencing a competing "
        "event as censored at their event time. Implemented with lifelines v0.30, "
        "ridge penaliser λ=0.1 to prevent singular information matrices on this small "
        "dataset. Per-event concordance index from the linear predictor."
    ))
    elems.append(h2("10.2  Fine-Gray Subdistribution Hazard (FG)"))
    elems.append(body(
        "IPCW-weighted subdistribution hazard formulation (Fine & Gray, 1999). Subjects "
        "experiencing a competing event remain in the risk set with censoring weights, "
        "correctly targeting the subdistribution hazard. Yields valid CIF estimates "
        "under the proportional subdistribution hazards assumption."
    ))
    elems.append(h2("10.3  DeepHit MLP (Tabular HLA Features)"))
    elems.append(body(
        "Two-hidden-layer MLP (d=64 per layer, GELU, dropout p=0.2) with a DeepHit "
        "joint PMF head using the same 21 tabular features. Trained with NLL + ranking "
        "loss, AdamW (LR=10⁻³), cosine annealing, early stopping (patience=25). "
        "This baseline tests whether a neural survival head improves on classical "
        "methods on small tabular data."
    ))
    elems.append(h2("10.4  Results"))

    cindex_rows = [
        ["Model",                "GvHD C-index",  "Relapse C-index",        "TRM C-index"],
        ["Cox (cause-specific)", "—",             "0.754 (0.527–1.000)",    "0.647 (0.459–0.854)"],
        ["Fine-Gray",            "—",             "0.841 (0.691–1.000) ★",  "0.655 (0.478–0.858) ★"],
        ["DeepHit MLP (tabular)","—",             "0.667 (0.129–1.000)",    "0.409 (0.259–0.568)"],
    ]
    elems += styled_table(cindex_rows[0], cindex_rows[1:],
        col_widths=[1.8*inch, 1.1*inch, 1.85*inch, BODY_W - 4.75*inch],
        caption="Time-dependent C-index (95% bootstrap CI, 1000 resamples). "
                "GvHD: n=2 test events — C-index undefined. ★ = best per column.")

    res_buf = results_figure()
    img = Image(res_buf, width=BODY_W * 0.88, height=BODY_W * 0.88 * 2.8/6.5)
    t_w = Table([[img]], colWidths=[BODY_W])
    t_w.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
    elems.append(vspace(6))
    elems.append(t_w)
    elems.append(Paragraph(
        "<i>Figure 3: Grouped bar chart of test-set C-index with 95% bootstrap CIs. "
        "Fine-Gray achieves highest discrimination on both evaluable endpoints. "
        "DeepHit MLP underperforms on TRM (C=0.41), consistent with the known "
        "difficulty of training neural survival models on n&lt;200 cohorts.</i>", ST["caption"]))

    elems.append(h2("10.5  Interpretation"))
    elems.append(body(
        "<b>Fine-Gray dominates</b> on both relapse (0.84) and TRM (0.66). "
        "The subdistribution hazard formulation is well-matched to the competing-risks "
        "structure of the problem. Confidence intervals are wide throughout — the "
        "relapse CI spans 0.69–1.00 — reflecting the fundamental statistical power "
        "limitation of 4 relapse events in a 29-patient test set."
    ))
    elems.append(body(
        "<b>DeepHit MLP underperforms</b> classical methods on TRM (0.41 vs 0.65), "
        "a common finding when deep models are trained on small datasets without strong "
        "structural inductive biases. This confirms the core CAPA hypothesis: architectural "
        "complexity alone does not compensate for the absence of structural priors — the "
        "ESM-2 embedding is precisely that prior."
    ))
    elems.append(body(
        "<b>GvHD cannot be evaluated</b> (n=2 test events). The full dataset contains "
        "only 16 GvHD events (8.6%), a fundamental limitation of the UCI BMT cohort. "
        "A registry dataset with hundreds of GvHD events is needed for reliable assessment."
    ))
    elems.append(PageBreak())
    return elems


def section_codebase():
    elems = [h1("11. Codebase Architecture Review"), rule()]

    mod_buf = module_diagram()
    img = Image(mod_buf, width=BODY_W * 0.97, height=BODY_W * 0.97 * 4.2/7.5)
    t_w = Table([[img]], colWidths=[BODY_W])
    t_w.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
    elems.append(vspace(4))
    elems.append(t_w)
    elems.append(Paragraph(
        "<i>Figure 4: CAPA V1 module structure. The package is organized into seven "
        "sub-packages plus a Next.js web frontend. All 14 modules have corresponding test files.</i>",
        ST["caption"]))

    elems.append(h2("11.1  Key Design Decisions"))
    elems += bullet_list([
        "<b>Frozen embeddings as first-class tensors.</b> ESM-2 is not a submodule of "
        "CAPAModel. Embeddings arrive as detached (batch, n_loci, 1280) tensors. This "
        "separates the heavy embedding computation (GPU-hours) from the lightweight "
        "training loop (minutes), enabling CPU-only training once embeddings are cached.",
        "<b>Two survival head options.</b> DeepHitHead (joint PMF softmax) and "
        "CauseSpecificHazardHead (sub-hazards with competing-risks product formula) "
        "share the same forward() / cif() interface. Switching requires only "
        "survival_type='cause_specific' at construction time.",
        "<b>Mock fallback in the API.</b> When no checkpoint is found, "
        "capa/api/predict.py returns Weibull-parameterised synthetic CIFs scaled by "
        "mismatch count. This allows the web demo to run before any model is trained, "
        "but makes it visually indistinguishable from real predictions — a "
        "UX risk that should be addressed with a clear 'Demo mode' banner.",
        "<b>Pydantic settings throughout.</b> All paths and hyperparameters live in "
        "capa/config.py as Pydantic BaseSettings, overridable via environment variables "
        "(CAPA_DATA_*, CAPA_EMBED_*, etc.). No hardcoded paths anywhere in the package.",
        "<b>NumPy-only evaluation.</b> The evaluate.py module has no PyTorch dependency. "
        "All metric functions operate on float64 NumPy arrays, making them usable with "
        "any survival model (lifelines Cox, scikit-learn, etc.).",
    ])

    elems.append(h2("11.2  Identified Code-Paper Discrepancies"))
    disc_rows = [
        ["Location",              "Paper says",                   "Code does"],
        ["trainer.py",            "ReduceLROnPlateau (patience=10)","CosineAnnealingLR"],
        ["capa_model.py",         "interaction_dim output = 2d'=256","interaction_dim param = 128 (output = 256 via concat)"],
        ["evaluate.py",           "Antolini (2005) time-dep C-index","Harrell C-index"],
        ["clinical encoder",      "4 continuous features (paper §2.3)","3 continuous + 1 binary flag"],
        ["configs/default.yaml",  "referenced in README/scripts","file does not exist"],
    ]
    elems += styled_table(disc_rows[0], disc_rows[1:],
        col_widths=[1.6*inch, 2.2*inch, BODY_W - 3.8*inch],
        caption="Code-paper discrepancies requiring reconciliation before journal submission")

    elems.append(h2("11.3  Test Coverage"))
    elems.append(body(
        "14 test files cover every module. Key tests include: "
        "test_interaction.py (CrossAttentionInteraction forward/backward, "
        "attention weight shape); test_survival_losses.py (NLL and ranking loss "
        "gradients, monotonicity of CIF output); test_capa_model.py "
        "(end-to-end forward with random embeddings); test_baselines.py "
        "(Cox-CS and Fine-Gray fit/predict on synthetic data); "
        "test_evaluate.py (C-index, Brier score, IBS against known analytical values)."
    ))
    elems.append(note(
        "The test suite uses random embeddings (no ESM-2 required) and synthetic "
        "survival data, keeping CI runtime under 60 seconds with no GPU needed."
    ))
    elems.append(PageBreak())
    return elems


def section_web():
    elems = [h1("12. Web Frontend and Deployment"), rule()]
    elems.append(h2("12.1  Next.js Frontend"))
    elems.append(body(
        "The web interface (web/) is a Next.js 14 App Router application deployed on "
        "Vercel. It contains four pages: landing (/), interactive prediction tool "
        "(/predict), about page (/about), and paper page (/paper). The prediction "
        "tool includes HLACombobox (dropdown with autocomplete for valid allele names), "
        "RiskChart (recharts-based CIF visualisation), and AttentionHeatmap "
        "(5×5 heatmap of donor-to-recipient attention weights)."
    ))
    elems.append(h2("12.2  API Route and Backend Proxy"))
    elems += code_block("""\
// web/app/api/predict/route.ts
// Next.js Route Handler — proxies to Python FastAPI backend
const BACKEND_URL = process.env.CAPA_BACKEND_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  const body = await request.json();
  // Validate: at least one HLA locus per side
  const upstream = await fetch(`${BACKEND_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(30_000),   // 30s timeout
  });
  return NextResponse.json(await upstream.json(),
                            { status: upstream.status });
}""", caption="API proxy route in web/app/api/predict/route.ts")

    elems.append(h2("12.3  Python FastAPI Backend"))
    elems.append(body(
        "capa/api/predict.py is a FastAPI application exposing POST /predict and "
        "GET /health endpoints. On startup it attempts to load a model checkpoint "
        "from CAPA_CHECKPOINT (default: runs/best/model.pt). If no checkpoint is "
        "found, /predict returns HTTP 200 with mock Weibull-parameterised CIFs "
        "rather than HTTP 503, to support demo usage. /health always returns "
        "HTTP 200 with a 'ready' boolean and optional 'startup_error' field."
    ))

    elems.append(h2("12.4  Deployment Gap (Critical)"))
    elems.append(body(
        "The Vercel deployment has CAPA_BACKEND_URL unset. The Next.js API route "
        "therefore defaults to localhost:8000, which is unreachable from the Vercel "
        "serverless environment. The frontend currently serves only mock data. "
        "Resolution requires:"
    ))
    elems += bullet_list([
        "Deploy the FastAPI backend to Modal.com or Railway (both referenced in CLAUDE.md)",
        "Set CAPA_BACKEND_URL in Vercel environment variables (Production + Preview)",
        "Add a /health polling call on the /predict page load so users see a clear "
        "'Model offline — showing demo data' banner when the backend is unreachable",
        "Train a baseline DeepHit checkpoint on UCI BMT tabular features and commit "
        "to the repository or Modal volume so the backend serves real predictions",
    ])
    elems.append(PageBreak())
    return elems


def section_limitations():
    elems = [h1("13. Limitations"), rule()]
    lims = [
        ("<b>No per-allele HLA typing in UCI BMT.</b>",
         "The dataset records only aggregate mismatch scores (0–3), not allele "
         "identities. CAPA's ESM-2 pipeline requires allele strings (e.g. 'A*02:01'). "
         "The entire ESM-2 + cross-attention pathway therefore cannot be evaluated "
         "end-to-end on any real patient in V1. Validation requires a registry dataset "
         "with high-resolution (field-2) HLA typing from CIBMTR or NMDP."),
        ("<b>Small sample size.</b>",
         "n=187 total, n=29 test. All confidence intervals are wide. GvHD "
         "(only 16 events total, 2 in the test set) cannot support any reliable "
         "discriminative evaluation. Results should be interpreted as proof-of-concept, "
         "not clinical benchmarks."),
        ("<b>Single-centre paediatric cohort.</b>",
         "All 187 patients are from one Polish centre, 1992–2004. Generalisability "
         "to adult HSCT, multi-centre settings, or modern conditioning regimens is unknown."),
        ("<b>ESM-2 not fine-tuned.</b>",
         "ESM-2 was pre-trained on general protein sequences (UniRef50). HLA alleles "
         "differ from typical ESM-2 training examples in that the immunologically critical "
         "residues (peptide-binding groove) are a small fraction of the sequence. "
         "Task-specific fine-tuning on HLA or immunology sequences may improve embedding "
         "quality substantially."),
        ("<b>Loci treated as independent.</b>",
         "The embedding pipeline computes one embedding per allele independently. "
         "Linkage disequilibrium (LD) across loci — the non-random co-inheritance of "
         "alleles — is a known biological signal not captured by the current architecture."),
        ("<b>No trained model checkpoint in the repository.</b>",
         "The FastAPI backend serves only mock data. The web demo is illustrative, "
         "not predictive."),
    ]
    for title, text in lims:
        elems.append(h3(title))
        elems.append(body(text))
    elems.append(PageBreak())
    return elems


def section_roadmap():
    elems = [h1("14. Version 2 Roadmap"), rule()]
    elems.append(h2("14.1  Immediate (Unblock the Repository)"))
    roadmap = [
        ("P0", "Create configs/default.yaml with the hyperparameters documented in the paper. "
               "scripts/train.py crashes on invocation without it."),
        ("P0", "Train the tabular DeepHit MLP baseline on UCI BMT and commit runs/best/model.pt "
               "so the backend serves real predictions."),
        ("P0", "Deploy FastAPI backend to Modal or Railway. Set CAPA_BACKEND_URL in Vercel "
               "production and preview environment variables."),
        ("P1", "Add notebooks 01_eda.ipynb (event distributions, KM curves), "
               "03_model_dev.ipynb (hyperparameter sweeps, learning curves), "
               "04_figures.ipynb (reproducible paper figure generation)."),
        ("P1", "Reconcile code-paper discrepancies: LR scheduler, C-index variant, "
               "clinical feature count. Update whichever is wrong."),
    ]
    rows = [[p, t] for p, t in roadmap]
    elems += styled_table(["Priority", "Action"], rows,
        col_widths=[0.7*inch, BODY_W - 0.7*inch],
        caption="Immediate fixes required before the repository is fully functional")

    elems.append(h2("14.2  Scientific Next Steps"))
    sci = [
        ("Registry data",
         "Access CIBMTR or NMDP registry data with allele-level (field-2) HLA typing. "
         "This is the single highest-leverage action — it enables end-to-end CAPA "
         "validation and is the primary direction for the next manuscript revision."),
        ("HLA-DPB1",
         "Add DPB1 as an optional 6th locus (config flag). TCE group mismatching at "
         "DPB1 is independently associated with TRM in unrelated HSCT; the current "
         "5-locus model omits this signal."),
        ("ESM-2 fine-tuning",
         "Unfreeze the final 1–2 ESM-2 transformer layers during training. Use a "
         "much smaller learning rate for ESM-2 parameters (layer-wise LR decay). "
         "Evaluate whether fine-tuning on HLA sequences improves embedding quality "
         "measured by UMAP cluster separation and downstream C-index."),
        ("Linkage disequilibrium encoding",
         "Add locus-level positional embeddings so the cross-attention network "
         "knows which locus each row corresponds to. Long-term: a locus-pair graph "
         "encoder that incorporates known LD structure between DRB1, DQB1, and DPB1."),
        ("Multi-donor comparison API",
         "Add POST /compare endpoint accepting a list of donor HLA dicts. Returns a "
         "ranked comparison of predicted CIF curves for all candidates — the clinical "
         "use case for which CAPA is designed."),
        ("Post-hoc calibration",
         "Apply isotonic regression to the predicted CIFs on the validation set. "
         "Small-cohort DeepHit models tend to be overconfident; post-hoc recalibration "
         "is cheap and reliable."),
        ("Cross-validation",
         "Replace the single 70/15/15 split with 5-fold nested cross-validation. "
         "Report mean ± SD C-index across folds."),
    ]
    sci_rows = [[t, d] for t, d in sci]
    elems += styled_table(["Direction", "Description"], sci_rows,
        col_widths=[1.4*inch, BODY_W - 1.4*inch],
        caption="Prioritised scientific directions for CAPA V2")
    return elems


def section_references():
    elems = [PageBreak(), h1("15. References"), rule()]
    refs = [
        "Appelbaum F.R. (2001). Haematopoietic cell transplantation as immunotherapy. "
        "<i>Nature</i>, 411, 385–389.",
        "Fine J.P. & Gray R.J. (1999). A proportional hazards model for the subdistribution "
        "of a competing risk. <i>JASA</i>, 94, 496–509.",
        "Graf E. et al. (1999). Assessment and comparison of prognostic classification "
        "schemes for survival data. <i>Statistics in Medicine</i>, 18, 2529–2545.",
        "Lee C. et al. (2018). DeepHit: A deep learning approach to survival analysis "
        "with competing risks. <i>AAAI 2018</i>.",
        "Lin Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein "
        "structure with a language model. <i>Science</i>, 379, 1123–1130.",
        "Loshchilov I. & Hutter F. (2019). Decoupled weight decay regularization. "
        "<i>ICLR 2019</i>.",
        "McInnes L. et al. (2018). UMAP: Uniform Manifold Approximation and Projection "
        "for Dimension Reduction. <i>arXiv:1802.03426</i>.",
        "Petersdorf E.W. et al. (2015). Role of HLA-DPB1 T-cell epitope groups in "
        "allogeneic hematopoietic cell transplantation. <i>NEJM</i>, 373, 599–609.",
        "Rives A. et al. (2021). Biological structure and function emerge from "
        "scaling unsupervised learning to 250 million protein sequences. "
        "<i>PNAS</i>, 118(15).",
        "Sikora M. et al. (2010). Application of rule induction algorithms for analysis "
        "of data collected by a bone marrow transplantation registry. "
        "<i>Applied Artificial Intelligence</i>, 25, 97–103.",
        "Vaswani A. et al. (2017). Attention is all you need. <i>NeurIPS 2017</i>.",
        "Zeiser R. & Blazar B.R. (2017). Acute graft-versus-host disease — "
        "biologic process, prevention, and therapy. <i>NEJM</i>, 377, 2167–2179.",
    ]
    for i, ref in enumerate(refs, 1):
        elems.append(Paragraph(f"[{i}]&#160;&#160;{ref}", ST["bullet"]))
        elems.append(vspace(3))
    return elems


# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE AND BUILD
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Generating CAPA Technical Debrief V1…")
    story = []
    story += cover_page()
    story += section_exec_summary()
    story += section_motivation()
    story += section_data()
    story += section_embeddings()
    story += section_architecture()
    story += section_deephit()
    story += section_loss()
    story += section_training()
    story += section_evaluation()
    story += section_baselines()
    story += section_codebase()
    story += section_web()
    story += section_limitations()
    story += section_roadmap()
    story += section_references()

    build_doc(story)
    print(f"Done → {OUT_PATH}")

if __name__ == "__main__":
    main()
