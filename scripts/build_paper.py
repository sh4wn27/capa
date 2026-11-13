"""Compile the manuscript and publish it to the website.

Runs pdflatex/bibtex on ``paper/main.tex``, copies the result to
``web/capa-paper.pdf`` and stamps the build date and page count into
``web/paper.html`` so the site always serves and labels the newest edition.

Usage::

    uv run python scripts/build_paper.py
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
WEB_PDF = ROOT / "web" / "capa-paper.pdf"
PAPER_HTML = ROOT / "web" / "paper.html"
TEX_BIN = Path("/Library/TeX/texbin")

logger = logging.getLogger(__name__)


def run_latex() -> None:
    """Run the full pdflatex/bibtex cycle on ``paper/main.tex``.

    Raises
    ------
    subprocess.CalledProcessError
        If any LaTeX or BibTeX pass fails.
    """
    env = os.environ.copy()
    if TEX_BIN.is_dir():
        env["PATH"] = f"{TEX_BIN}{os.pathsep}{env['PATH']}"
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"]
    for cmd in (latex, ["bibtex", "main"], latex, latex):
        logger.info("running %s", " ".join(cmd))
        subprocess.run(cmd, cwd=PAPER_DIR, env=env, check=True, capture_output=True)


def page_count() -> int:
    """Read the page count of the last build from ``paper/main.log``.

    Returns
    -------
    int
        Number of pages in ``paper/main.pdf``.
    """
    log_text = (PAPER_DIR / "main.log").read_text(errors="replace")
    match = re.search(r"Output written on main\.pdf \((\d+) pages?", log_text)
    if match is None:
        msg = "could not find page count in paper/main.log"
        raise RuntimeError(msg)
    return int(match.group(1))


def stamp_html(date: dt.date, pages: int) -> None:
    """Write the edition label and a cache-busting PDF link into ``paper.html``.

    Parameters
    ----------
    date : datetime.date
        Build date shown on the page.
    pages : int
        Page count shown on the page.
    """
    html = PAPER_HTML.read_text()
    html, n_label = re.subn(
        r'(<span class="pm-v" id="pdf-edition">)[^<]*(</span>)',
        rf"\g<1>{date.isoformat()} · {pages} pp\g<2>",
        html,
    )
    html, n_link = re.subn(
        r'(id="pdf-link" href="capa-paper\.pdf)(\?v=\d+)?(")',
        rf"\g<1>?v={date:%Y%m%d}\g<3>",
        html,
    )
    if n_label != 1 or n_link != 1:
        msg = "paper.html is missing the #pdf-edition or #pdf-link markers"
        raise RuntimeError(msg)
    PAPER_HTML.write_text(html)


def main() -> None:
    """Build the paper and publish it to ``web/``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_latex()
    except subprocess.CalledProcessError as err:
        logger.error("LaTeX build failed (%s); see paper/main.log", " ".join(err.cmd))
        sys.exit(1)
    pages = page_count()
    shutil.copyfile(PAPER_DIR / "main.pdf", WEB_PDF)
    stamp_html(dt.date.today(), pages)
    logger.info("published %d-page PDF to %s", pages, WEB_PDF.relative_to(ROOT))


if __name__ == "__main__":
    main()
