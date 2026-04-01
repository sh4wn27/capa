"""HDF5-backed embedding cache to avoid recomputing ESM-2 embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Read/write numpy embedding vectors keyed by allele name in an HDF5 file.

    Parameters
    ----------
    cache_path : Path
        Path to the ``.h5`` cache file (created if it does not exist).
    """

    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    def contains(self, allele: str) -> bool:
        """Return ``True`` if an embedding for *allele* is cached.

        Parameters
        ----------
        allele : str
            HLA allele name used as the HDF5 dataset key.

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
        with h5py.File(self._path, "r") as f:
            if allele not in f:
                raise KeyError(f"Embedding not cached for allele: {allele!r}")
            return f[allele][:]  # type: ignore[index]

    def put(self, allele: str, embedding: npt.NDArray[np.float32]) -> None:
        """Store an embedding vector in the cache.

        Parameters
        ----------
        allele : str
            HLA allele name.
        embedding : npt.NDArray[np.float32]
            Embedding vector.
        """
        with h5py.File(self._path, "a") as f:
            if allele in f:
                del f[allele]
            f.create_dataset(allele, data=embedding, compression="gzip")
        logger.debug("Cached embedding for %s", allele)

    def __len__(self) -> int:
        if not self._path.exists():
            return 0
        with h5py.File(self._path, "r") as f:
            return len(f)
