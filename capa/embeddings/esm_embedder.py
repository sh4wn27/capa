"""ESM-2 HLA allele sequence → embedding vector encoder."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import torch

logger = logging.getLogger(__name__)


class ESMEmbedder:
    """Encode HLA amino acid sequences using ESM-2.

    The model weights are loaded once and kept frozen. Embeddings are mean-pooled
    over sequence positions to produce a fixed-size vector per allele.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"facebook/esm2_t33_650M_UR50D"``.
    device : str
        Torch device string, e.g. ``"cpu"``, ``"cuda"``, or ``"mps"``.
    batch_size : int
        Number of sequences to embed per forward pass.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        self._model_name = model_name
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._model: object | None = None
        self._tokenizer: object | None = None

    def _load(self) -> None:
        """Lazy-load the ESM-2 model and tokenizer on first use."""
        if self._model is not None:
            return
        # Import inside method to allow import of this module without transformers installed
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

        logger.info("Loading ESM-2 model: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()  # type: ignore[union-attr]
        self._model.to(self._device)  # type: ignore[union-attr]
        logger.info("ESM-2 loaded on device: %s", self._device)

    @torch.no_grad()  # type: ignore[misc]
    def embed(self, sequences: list[str]) -> npt.NDArray[np.float32]:
        """Embed a list of amino acid sequences.

        Parameters
        ----------
        sequences : list[str]
            Amino acid sequences in single-letter code.

        Returns
        -------
        npt.NDArray[np.float32]
            Array of shape ``(len(sequences), embedding_dim)`` with mean-pooled
            ESM-2 representations.
        """
        self._load()
        all_embeddings: list[npt.NDArray[np.float32]] = []

        for i in range(0, len(sequences), self._batch_size):
            batch = sequences[i : i + self._batch_size]
            inputs = self._tokenizer(  # type: ignore[call-arg, operator]
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)  # type: ignore[operator]
            # Mean-pool over sequence positions (excluding padding via attention mask)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = outputs.last_hidden_state * mask
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def embed_one(self, sequence: str) -> npt.NDArray[np.float32]:
        """Embed a single sequence.

        Parameters
        ----------
        sequence : str
            Amino acid sequence in single-letter code.

        Returns
        -------
        npt.NDArray[np.float32]
            Embedding vector of shape ``(embedding_dim,)``.
        """
        return self.embed([sequence])[0]
