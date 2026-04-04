"""HDF5-backed embedding cache to avoid recomputing ESM-2 embeddings.

Usage pattern
-------------
The typical workflow is::

    from capa.embeddings.cache import EmbeddingCache
    from capa.embeddings.esm_embedder import ESMEmbedder

    embedder = ESMEmbedder()
    cache = EmbeddingCache(Path("data/processed/hla_embeddings.h5"))

    allele_sequences = {"A*02:01": "MAVMAPRTL...", "B*07:02": "MRVMAPRTL..."}
    embeddings = cache.embed_alleles(allele_sequences, embedder)

:meth:`EmbeddingCache.embed_alleles` checks the cache before calling the
embedder, so repeated runs only compute embeddings for new alleles.  The HDF5
file persists across Python sessions.

Thread / process safety
-----------------------
HDF5 files opened in append mode (``"a"``) are **not** safe for concurrent
writes from multiple processes.  Embed one process at a time, or use the SWMR
feature of HDF5 if parallel reads are needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import h5py
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder protocol (structural typing — no hard dependency on ESMEmbedder)
# ---------------------------------------------------------------------------

@runtime_checkable
class SequenceEmbedder(Protocol):
    """Minimal interface required by :meth:`EmbeddingCache.embed_alleles`.

    Any object with an ``embed(sequences: list[str]) -> ndarray`` method
    satisfies this protocol, including :class:`~capa.embeddings.esm_embedder.ESMEmbedder`
    and lightweight mock/stub objects used in tests.
    """

    def embed(
        self, sequences: list[str], **kwargs: Any
    ) -> npt.NDArray[np.float32]: ...


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Read/write numpy embedding vectors keyed by allele name in an HDF5 file.

    Each allele is stored as a separate HDF5 dataset, gzip-compressed.  The
    file is opened and closed on every operation to avoid holding a file
    descriptor open across long-running jobs.

    Parameters
    ----------
    cache_path : Path
        Path to the ``.h5`` cache file.  The file and its parent directories
        are created automatically on the first :meth:`put` call.
    """

    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Low-level read/write
    # ------------------------------------------------------------------

    def contains(self, allele: str) -> bool:
        """Return ``True`` if an embedding for *allele* is already cached.

        Parameters
        ----------
        allele : str
            HLA allele name used as the HDF5 dataset key, e.g. ``"A*02:01"``.

        Returns
        -------
        bool
        """
        if not self._path.exists():
            return False
        with h5py.File(self._path, "r") as f:
            return allele in f

    def get(self, allele: str) -> npt.NDArray[np.float32]:
        """Retrieve a cached embedding vector.

        Parameters
        ----------
        allele : str
            HLA allele name.

        Returns
        -------
        npt.NDArray[np.float32]
            Embedding vector of shape ``(embedding_dim,)``.

        Raises
        ------
        KeyError
            If the allele is not in the cache.
        """
        if not self._path.exists():
            raise KeyError(f"Embedding not cached for allele: {allele!r}")
        with h5py.File(self._path, "r") as f:
            if allele not in f:
                raise KeyError(f"Embedding not cached for allele: {allele!r}")
            return f[allele][:]  # type: ignore[index]

    def put(self, allele: str, embedding: npt.NDArray[np.float32]) -> None:
        """Store an embedding vector in the cache.

        Overwrites any existing entry for *allele*.

        Parameters
        ----------
        allele : str
            HLA allele name.
        embedding : npt.NDArray[np.float32]
            Embedding vector (any shape, typically ``(embedding_dim,)``).
        """
        with h5py.File(self._path, "a") as f:
            if allele in f:
                del f[allele]
            f.create_dataset(allele, data=embedding, compression="gzip")
        logger.debug("Cached embedding for %s", allele)

    def __len__(self) -> int:
        """Return the number of alleles currently stored in the cache."""
        if not self._path.exists():
            return 0
        with h5py.File(self._path, "r") as f:
            return len(f)

    # ------------------------------------------------------------------
    # High-level: cache-aware batch embedding
    # ------------------------------------------------------------------

    def embed_alleles(
        self,
        allele_sequences: dict[str, str],
        embedder: SequenceEmbedder,
        *,
        show_progress: bool = True,
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Return embeddings for *allele_sequences*, using the cache where possible.

        Alleles already present in the HDF5 cache are returned without calling
        the embedder.  Missing alleles are embedded in a **single batched call**
        (respecting the embedder's internal batch size), stored in the cache,
        and merged with the cached results.

        Parameters
        ----------
        allele_sequences : dict[str, str]
            Mapping of allele name → amino acid sequence for every allele that
            needs an embedding.
        embedder : SequenceEmbedder
            An object with an ``embed(sequences: list[str])`` method, typically
            an :class:`~capa.embeddings.esm_embedder.ESMEmbedder` instance.
        show_progress : bool
            Show a ``tqdm`` progress bar while writing new embeddings to disk.
            Defaults to ``True``.

        Returns
        -------
        dict[str, npt.NDArray[np.float32]]
            Mapping of allele name → embedding vector for every allele in
            *allele_sequences*.
        """
        results: dict[str, npt.NDArray[np.float32]] = {}

        # Partition into cache hits and misses
        cached_alleles: list[str] = []
        missing_alleles: list[str] = []
        for allele in allele_sequences:
            if self.contains(allele):
                cached_alleles.append(allele)
            else:
                missing_alleles.append(allele)

        logger.info(
            "embed_alleles: %d cached, %d to compute",
            len(cached_alleles),
            len(missing_alleles),
        )

        # Fetch cache hits
        for allele in cached_alleles:
            results[allele] = self.get(allele)

        # Compute and store misses
        if missing_alleles:
            sequences = [allele_sequences[a] for a in missing_alleles]
            embeddings = embedder.embed(sequences)

            for allele, vec in tqdm(
                zip(missing_alleles, embeddings),
                total=len(missing_alleles),
                desc="Writing to cache",
                unit="allele",
                disable=not show_progress,
            ):
                self.put(allele, vec)
                results[allele] = vec

        return results

    def cached_alleles(self) -> list[str]:
        """Return a sorted list of allele names currently in the cache.

        Returns
        -------
        list[str]
        """
        if not self._path.exists():
            return []
        with h5py.File(self._path, "r") as f:
            return sorted(f.keys())
