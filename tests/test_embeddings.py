"""Tests for capa.embeddings (cache and sequence DB)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from capa.embeddings.cache import EmbeddingCache
from capa.embeddings.hla_sequences import HLASequenceDB


class TestEmbeddingCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        vec = np.random.rand(1280).astype(np.float32)
        cache.put("A*02:01", vec)
        assert cache.contains("A*02:01")
        np.testing.assert_array_almost_equal(cache.get("A*02:01"), vec)

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        with pytest.raises(KeyError):
            cache.get("A*99:99")

    def test_len(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        assert len(cache) == 0
        cache.put("A*02:01", np.zeros(10, dtype=np.float32))
        cache.put("B*07:02", np.zeros(10, dtype=np.float32))
        assert len(cache) == 2

    def test_overwrite(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        v1 = np.ones(5, dtype=np.float32)
        v2 = np.zeros(5, dtype=np.float32)
        cache.put("A*02:01", v1)
        cache.put("A*02:01", v2)
        np.testing.assert_array_equal(cache.get("A*02:01"), v2)

    def test_not_exists_contains_false(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "missing.h5")
        assert not cache.contains("A*02:01")


class TestHLASequenceDB:
    def _make_db(self, tmp_path: Path, data: dict[str, str]) -> HLASequenceDB:
        p = tmp_path / "seqs.json"
        p.write_text(json.dumps(data))
        return HLASequenceDB(p)

    def test_exact_lookup(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path, {"A*02:01": "MAAAA"})
        assert db.get("A*02:01") == "MAAAA"

    def test_fallback_two_field(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path, {"A*02:01": "MAAAA"})
        assert db.get("A*02:01:01") == "MAAAA"

    def test_missing_raises(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path, {})
        with pytest.raises(KeyError):
            db.get("A*99:99")

    def test_len(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path, {"A*02:01": "M", "B*07:02": "M"})
        assert len(db) == 2
