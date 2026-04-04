"""ESM-2 HLA allele sequence → embedding vector encoder.

Design notes
------------
* **Lazy loading** — the ESM-2 weights (~1.2 GB) are only downloaded and loaded
  into memory on the first call to :meth:`ESMEmbedder.embed` or
  :meth:`ESMEmbedder.embed_one`.  This keeps import time near-zero and lets the
  module be imported on machines without a GPU or internet access.

* **Device auto-detection** — :func:`detect_device` checks for CUDA, then Apple
  Silicon MPS, then falls back to CPU.  Pass an explicit ``device`` string to
  :class:`ESMEmbedder` to override.

* **Mean-pooling** — ESM-2 outputs a hidden state for every token, including
  special ``[CLS]`` / ``[EOS]`` tokens and padding tokens.  We exclude padding
  via the attention mask and average over the remaining positions to get a
  fixed-size representation regardless of sequence length.

* **Batch progress** — :meth:`ESMEmbedder.embed` shows a ``tqdm`` progress bar
  whenever more than one batch is processed, so long-running embedding jobs
  (e.g. all 44 000 IMGT/HLA alleles) are observable.

* **No-gradient context** — the ``@torch.no_grad()`` decorator on :meth:`embed`
  halves memory usage compared to the default gradient-tracking mode.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default ESM-2 model used throughout the project
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
# Embedding dimensionality for the default model (33 transformer layers, 650M params)
EMBEDDING_DIM = 1280


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Return the best available PyTorch device string.

    Priority: CUDA GPU → Apple Silicon MPS → CPU.

    Returns
    -------
    str
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if torch.cuda.is_available():
        logger.info("CUDA GPU detected — using cuda")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS detected — using mps")
        return "mps"
    logger.info("No accelerator detected — using cpu")
    return "cpu"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class ESMEmbedder:
    """Encode HLA amino acid sequences using ESM-2.

    The model weights are loaded **once** and kept frozen in ``eval()`` mode.
    Embeddings are mean-pooled over non-padding sequence positions to produce a
    fixed-size :data:`EMBEDDING_DIM`-dimensional vector per allele.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
        Defaults to ``"facebook/esm2_t33_650M_UR50D"`` (650 M parameters,
        1280-dimensional hidden states).
    device : str | None
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        ``None`` (default) triggers :func:`detect_device` auto-detection.
    batch_size : int
        Number of sequences processed per forward pass.  Reduce if you run out
        of GPU memory; increase for throughput on high-VRAM cards.
    max_length : int
        Maximum token length passed to the tokenizer.  HLA proteins are
        typically 180–380 AA; 512 comfortably covers all known alleles.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._device = torch.device(device if device is not None else detect_device())
        self._batch_size = batch_size
        self._max_length = max_length
        # Populated lazily on first call to embed()
        self._model: object | None = None
        self._tokenizer: object | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The torch device the model runs on."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """``True`` once the model weights have been loaded into memory."""
        return self._model is not None

    @torch.no_grad()  # type: ignore[misc]
    def embed(
        self,
        sequences: list[str],
        *,
        show_progress: bool | None = None,
    ) -> npt.NDArray[np.float32]:
        """Embed a list of amino acid sequences via mean-pooled ESM-2 representations.

        Parameters
        ----------
        sequences : list[str]
            Amino acid sequences in single-letter IUPAC code.  ``X`` (unknown)
            is handled by the ESM-2 tokenizer.
        show_progress : bool | None
            Display a ``tqdm`` progress bar over batches.
            ``None`` (default) shows the bar only when there are multiple batches.

        Returns
        -------
        npt.NDArray[np.float32]
            Array of shape ``(len(sequences), EMBEDDING_DIM)`` where each row is
            the mean-pooled representation of the corresponding sequence.

        Raises
        ------
        ValueError
            If *sequences* is empty.
        """
        if not sequences:
            raise ValueError("sequences must be non-empty")

        self._load()

        n_batches = (len(sequences) + self._batch_size - 1) // self._batch_size
        if show_progress is None:
            show_progress = n_batches > 1

        all_embeddings: list[npt.NDArray[np.float32]] = []

        batch_iter = range(0, len(sequences), self._batch_size)
        for i in tqdm(
            batch_iter,
            desc="Embedding batches",
            unit="batch",
            total=n_batches,
            disable=not show_progress,
        ):
            batch = sequences[i : i + self._batch_size]
            inputs = self._tokenizer(  # type: ignore[call-arg, operator]
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)  # type: ignore[operator]

            # Mean-pool over sequence positions, masking out padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
            hidden = outputs.last_hidden_state * mask              # (B, L, D)
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, D)

            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

        result = np.concatenate(all_embeddings, axis=0)
        logger.debug("Embedded %d sequences → shape %s", len(sequences), result.shape)
        return result

    def embed_one(self, sequence: str) -> npt.NDArray[np.float32]:
        """Embed a single amino acid sequence.

        Parameters
        ----------
        sequence : str
            Amino acid sequence in single-letter IUPAC code.

        Returns
        -------
        npt.NDArray[np.float32]
            Embedding vector of shape ``(EMBEDDING_DIM,)``.
        """
        return self.embed([sequence], show_progress=False)[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load the ESM-2 model and tokenizer on first use.

        Idempotent — subsequent calls are no-ops.
        """
        if self._model is not None:
            return

        # Import inside method so that the module is importable without
        # `transformers` installed (tests can inject mock model/tokenizer).
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

        logger.info(
            "Loading ESM-2 weights: %s (device=%s)", self._model_name, self._device
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()  # type: ignore[union-attr]
        self._model.to(self._device)  # type: ignore[union-attr]
        logger.info("ESM-2 ready on %s", self._device)
