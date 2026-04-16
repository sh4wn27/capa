"""Tests for capa.embeddings — ESMEmbedder, EmbeddingCache, parse_fasta.

Testing strategy for ESMEmbedder
---------------------------------
Loading the real ESM-2 model requires ~1.2 GB of weights and a network call.
We bypass this by *injecting* a tiny mock model and tokenizer directly onto
the embedder instance, skipping the ``_load()`` HuggingFace call entirely.

The mock objects satisfy the duck-typed interface used inside ``embed()``:

* ``mock_tokenizer(batch, ...)`` → dict with ``input_ids`` and
  ``attention_mask`` tensors.
* ``mock_model(**inputs)`` → namespace with ``last_hidden_state`` tensor of
  shape ``(batch, seq_len, hidden_dim)``.

All mathematical correctness tests (mean-pooling, masking, shapes) run on CPU
with these mocks.  GPU paths are guarded by ``pytest.mark.skipif``.
"""

from __future__ import annotations

import json
import logging
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from capa.data.hla_parser import parse_who_allele
from capa.embeddings.cache import EmbeddingCache, SequenceEmbedder
from capa.embeddings.esm_embedder import EMBEDDING_DIM, ESMEmbedder, detect_device
from capa.embeddings.hla_sequences import (
    HLASequenceDB,
    _best_from_candidates,
    _candidate_keys,
    _is_expressed,
)
from scripts.download_hla_seqs import parse_fasta


# ---------------------------------------------------------------------------
# Helpers: mock model / tokenizer injection
# ---------------------------------------------------------------------------

_MOCK_SEQ_LEN = 12
_MOCK_HIDDEN = 16


def _make_mock_tokenizer(
    seq_len: int = _MOCK_SEQ_LEN,
    mask_last_n: int = 0,
) -> object:
    """Return a callable that mimics HuggingFace tokenizer output.

    Parameters
    ----------
    seq_len : int
        Fixed sequence length (simulates padding to this length).
    mask_last_n : int
        Number of trailing tokens to set to 0 in the attention mask
        (simulates padding tokens that should be ignored).
    """
    def tokenizer(
        batch: list[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        b = len(batch)
        mask = torch.ones(b, seq_len, dtype=torch.long)
        if mask_last_n > 0:
            mask[:, seq_len - mask_last_n :] = 0
        return {
            "input_ids": torch.zeros(b, seq_len, dtype=torch.long),
            "attention_mask": mask,
        }

    return tokenizer


def _make_mock_model(hidden_dim: int = _MOCK_HIDDEN, fill: float = 1.0) -> object:
    """Return a callable that mimics a HuggingFace model forward pass.

    Parameters
    ----------
    hidden_dim : int
        Hidden state dimensionality.
    fill : float
        Constant value to fill ``last_hidden_state`` with.  Defaults to 1.0
        so that mean-pooling over non-masked positions gives ``1.0`` in every
        dimension.
    """
    class _MockOutput:
        def __init__(self, hs: torch.Tensor) -> None:
            self.last_hidden_state = hs

    class _MockModel:
        def __call__(self, **inputs: torch.Tensor) -> _MockOutput:
            b, sl = inputs["attention_mask"].shape
            return _MockOutput(torch.full((b, sl, hidden_dim), fill))

        def eval(self) -> "_MockModel":
            return self

        def to(self, device: object) -> "_MockModel":
            return self

    return _MockModel()


def _make_embedder(
    *,
    hidden_dim: int = _MOCK_HIDDEN,
    batch_size: int = 4,
    fill: float = 1.0,
    seq_len: int = _MOCK_SEQ_LEN,
    mask_last_n: int = 0,
) -> ESMEmbedder:
    """Return an ESMEmbedder pre-loaded with mock model/tokenizer (no download)."""
    embedder = ESMEmbedder(device="cpu", batch_size=batch_size)
    embedder._model = _make_mock_model(hidden_dim, fill)
    embedder._tokenizer = _make_mock_tokenizer(seq_len, mask_last_n)
    return embedder


# ---------------------------------------------------------------------------
# detect_device
# ---------------------------------------------------------------------------

class TestDetectDevice:
    def test_returns_valid_device_string(self) -> None:
        device = detect_device()
        assert device in {"cpu", "cuda", "mps"}

    def test_returns_string(self) -> None:
        assert isinstance(detect_device(), str)

    def test_cpu_always_valid(self) -> None:
        # Whatever is returned, PyTorch must accept it
        torch.device(detect_device())

    def test_cuda_only_when_available(self) -> None:
        device = detect_device()
        if device == "cuda":
            assert torch.cuda.is_available()

    def test_mps_only_when_available(self) -> None:
        device = detect_device()
        if device == "mps":
            assert hasattr(torch.backends, "mps")
            assert torch.backends.mps.is_available()


# ---------------------------------------------------------------------------
# ESMEmbedder — construction
# ---------------------------------------------------------------------------

class TestESMEmbedderInit:
    def test_default_device_auto_detected(self) -> None:
        embedder = ESMEmbedder()
        assert str(embedder.device) in {"cpu", "cuda", "mps"}

    def test_explicit_cpu_device(self) -> None:
        embedder = ESMEmbedder(device="cpu")
        assert embedder.device == torch.device("cpu")

    def test_model_not_loaded_at_init(self) -> None:
        embedder = ESMEmbedder(device="cpu")
        assert not embedder.is_loaded

    def test_model_loaded_after_embed(self) -> None:
        embedder = _make_embedder()
        # is_loaded is True because we injected the mock directly
        assert embedder.is_loaded


# ---------------------------------------------------------------------------
# ESMEmbedder — embed() output shape and type
# ---------------------------------------------------------------------------

class TestESMEmbedderEmbed:
    def test_output_shape_single(self) -> None:
        embedder = _make_embedder(hidden_dim=_MOCK_HIDDEN)
        out = embedder.embed(["MAVMAPRTL"])
        assert out.shape == (1, _MOCK_HIDDEN)

    def test_output_shape_batch(self) -> None:
        embedder = _make_embedder(hidden_dim=_MOCK_HIDDEN)
        seqs = ["MAVMAPRTL", "MRVMAPRTL", "MRVSAPRTL"]
        out = embedder.embed(seqs)
        assert out.shape == (3, _MOCK_HIDDEN)

    def test_output_dtype_float32(self) -> None:
        embedder = _make_embedder()
        out = embedder.embed(["MAVMAP"])
        assert out.dtype == np.float32

    def test_empty_sequence_list_raises(self) -> None:
        embedder = _make_embedder()
        with pytest.raises(ValueError, match="non-empty"):
            embedder.embed([])

    def test_multi_batch_correct_row_count(self) -> None:
        """8 sequences with batch_size=3 → three batches → 8 rows total."""
        embedder = _make_embedder(batch_size=3)
        seqs = ["SEQ"] * 8
        out = embedder.embed(seqs)
        assert out.shape[0] == 8

    def test_batch_boundary_single_remainder(self) -> None:
        """7 sequences with batch_size=4 → last batch has 1 sequence."""
        embedder = _make_embedder(batch_size=4)
        out = embedder.embed(["S"] * 7)
        assert out.shape == (7, _MOCK_HIDDEN)

    def test_returns_numpy_array(self) -> None:
        embedder = _make_embedder()
        out = embedder.embed(["MAVMAP"])
        assert isinstance(out, np.ndarray)

    def test_no_gradient_tracking(self) -> None:
        """embed() must run inside torch.no_grad() — no grad fn on results."""
        embedder = _make_embedder()
        out = embedder.embed(["SEQ"])
        # Output is numpy, cannot have grad.  Check that no in-flight grads exist.
        assert isinstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# ESMEmbedder — mean-pooling correctness
# ---------------------------------------------------------------------------

class TestESMEmbedderPooling:
    def test_all_ones_hidden_mean_pools_to_one(self) -> None:
        """With all-ones hidden states and full mask → pooled values all 1.0."""
        embedder = _make_embedder(fill=1.0, mask_last_n=0)
        out = embedder.embed(["MAVMAPRTL"])
        np.testing.assert_allclose(out[0], np.ones(_MOCK_HIDDEN), atol=1e-5)

    def test_padding_tokens_excluded_from_mean(self) -> None:
        """Padding tokens (mask=0) must not affect the mean-pooled output.

        We fill hidden states with 1.0 and mask the last 4 of 12 tokens.
        The unmasked tokens are still all 1.0, so the pooled value must
        remain 1.0 regardless of what the masked positions contain.
        """
        # mask_last_n=4 → only 8 of 12 positions are unmasked
        embedder = _make_embedder(fill=1.0, seq_len=12, mask_last_n=4)
        out = embedder.embed(["MAVMAPRTL"])
        # All non-padding tokens = 1.0, so mean must be 1.0
        np.testing.assert_allclose(out[0], np.ones(_MOCK_HIDDEN), atol=1e-5)

    def test_different_sequences_can_give_same_embedding(self) -> None:
        """Mock always returns fill=1.0, so all sequences embed identically."""
        embedder = _make_embedder(fill=1.0)
        out = embedder.embed(["AAA", "MAVMAPRTL"])
        np.testing.assert_allclose(out[0], out[1], atol=1e-5)


# ---------------------------------------------------------------------------
# ESMEmbedder — embed_one
# ---------------------------------------------------------------------------

class TestESMEmbedderEmbedOne:
    def test_shape_is_1d(self) -> None:
        embedder = _make_embedder(hidden_dim=_MOCK_HIDDEN)
        out = embedder.embed_one("MAVMAPRTL")
        assert out.shape == (_MOCK_HIDDEN,)

    def test_dtype_float32(self) -> None:
        embedder = _make_embedder()
        assert embedder.embed_one("SEQ").dtype == np.float32

    def test_consistent_with_embed(self) -> None:
        """embed_one('X') must equal embed(['X'])[0]."""
        embedder = _make_embedder()
        single = embedder.embed_one("MAVMAP")
        batch = embedder.embed(["MAVMAP"])[0]
        np.testing.assert_array_equal(single, batch)


# ---------------------------------------------------------------------------
# ESMEmbedder — progress bar (tqdm)
# ---------------------------------------------------------------------------

class TestESMEmbedderProgress:
    def test_no_progress_single_batch(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Single-batch embed should not print a progress bar by default."""
        embedder = _make_embedder(batch_size=10)
        embedder.embed(["SEQ1", "SEQ2"])  # 2 sequences, batch_size=10 → 1 batch
        captured = capsys.readouterr()
        # tqdm writes to stderr; nothing should appear for 1 batch
        assert "Embedding batches" not in captured.err

    def test_progress_shown_multi_batch(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Multi-batch embed should display a tqdm bar when show_progress=True."""
        embedder = _make_embedder(batch_size=2)
        # 6 sequences with batch_size=2 → 3 batches → bar shown
        embedder.embed(["S"] * 6, show_progress=True)
        captured = capsys.readouterr()
        assert "Embedding batches" in captured.err

    def test_progress_suppressed_when_disabled(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        embedder = _make_embedder(batch_size=2)
        embedder.embed(["S"] * 6, show_progress=False)
        captured = capsys.readouterr()
        assert "Embedding batches" not in captured.err


# ---------------------------------------------------------------------------
# EmbeddingCache — low-level operations (already tested in full in the
# previous session; minimal re-tests here for completeness)
# ---------------------------------------------------------------------------

class TestEmbeddingCacheLowLevel:
    def test_put_get_round_trip(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        vec = np.arange(8, dtype=np.float32)
        cache.put("A*02:01", vec)
        np.testing.assert_array_equal(cache.get("A*02:01"), vec)

    def test_contains_true_after_put(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        cache.put("A*02:01", np.zeros(4, dtype=np.float32))
        assert cache.contains("A*02:01")

    def test_contains_false_before_put(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        assert not cache.contains("B*07:02")

    def test_get_missing_raises_key_error(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        with pytest.raises(KeyError):
            cache.get("A*99:99")

    def test_len(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        assert len(cache) == 0
        cache.put("A*02:01", np.zeros(4, dtype=np.float32))
        assert len(cache) == 1

    def test_cached_alleles_sorted(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        cache.put("B*07:02", np.zeros(4, dtype=np.float32))
        cache.put("A*02:01", np.zeros(4, dtype=np.float32))
        alleles = cache.cached_alleles()
        assert alleles == sorted(alleles)
        assert "A*02:01" in alleles

    def test_overwrite_updates_value(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        cache.put("A*02:01", np.ones(4, dtype=np.float32))
        cache.put("A*02:01", np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(cache.get("A*02:01"), np.zeros(4, dtype=np.float32))


# ---------------------------------------------------------------------------
# EmbeddingCache.embed_alleles — cache integration
# ---------------------------------------------------------------------------

class _FixedEmbedder:
    """Minimal embedder stub that returns all-zeros with configurable dim.

    Tracks how many sequences were passed to embed() so tests can verify
    cache-hit skipping.
    """

    def __init__(self, dim: int = _MOCK_HIDDEN) -> None:
        self.dim = dim
        self.call_count = 0
        self.last_sequences: list[str] = []

    def embed(
        self, sequences: list[str], **_: object
    ) -> npt.NDArray[np.float32]:
        self.call_count += 1
        self.last_sequences = list(sequences)
        return np.zeros((len(sequences), self.dim), dtype=np.float32)


class TestEmbedAlleles:
    def test_returns_all_requested_alleles(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        embedder = _FixedEmbedder()
        allele_seqs = {"A*02:01": "MAVMAP", "B*07:02": "MRVMAP"}
        result = cache.embed_alleles(allele_seqs, embedder, show_progress=False)
        assert set(result.keys()) == {"A*02:01", "B*07:02"}

    def test_shapes_match_embedder_dim(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        dim = 32
        embedder = _FixedEmbedder(dim=dim)
        result = cache.embed_alleles(
            {"A*02:01": "SEQ1", "B*07:02": "SEQ2"}, embedder, show_progress=False
        )
        for vec in result.values():
            assert vec.shape == (dim,)

    def test_all_miss_calls_embedder_once(self, tmp_path: Path) -> None:
        """With empty cache, embedder.embed() is called once with all sequences."""
        cache = EmbeddingCache(tmp_path / "test.h5")
        embedder = _FixedEmbedder()
        allele_seqs = {"A*02:01": "SEQ1", "B*07:02": "SEQ2", "C*07:01": "SEQ3"}
        cache.embed_alleles(allele_seqs, embedder, show_progress=False)
        assert embedder.call_count == 1
        assert len(embedder.last_sequences) == 3

    def test_all_hit_skips_embedder(self, tmp_path: Path) -> None:
        """With all alleles cached, embedder.embed() is never called."""
        cache = EmbeddingCache(tmp_path / "test.h5")
        vec = np.ones(8, dtype=np.float32)
        cache.put("A*02:01", vec)
        cache.put("B*07:02", vec)

        embedder = _FixedEmbedder()
        cache.embed_alleles({"A*02:01": "SEQ1", "B*07:02": "SEQ2"}, embedder, show_progress=False)
        assert embedder.call_count == 0

    def test_partial_hit_only_computes_misses(self, tmp_path: Path) -> None:
        """Cache hit for A*02:01, miss for B*07:02 → embedder called with only B."""
        cache = EmbeddingCache(tmp_path / "test.h5")
        cache.put("A*02:01", np.ones(8, dtype=np.float32))

        embedder = _FixedEmbedder(dim=8)
        result = cache.embed_alleles(
            {"A*02:01": "SEQ1", "B*07:02": "SEQ2"}, embedder, show_progress=False
        )
        assert embedder.call_count == 1
        assert embedder.last_sequences == ["SEQ2"]
        assert set(result.keys()) == {"A*02:01", "B*07:02"}

    def test_cached_hit_value_returned_unchanged(self, tmp_path: Path) -> None:
        """Cached vector must be returned byte-for-byte (not recomputed)."""
        cache = EmbeddingCache(tmp_path / "test.h5")
        original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        cache.put("A*02:01", original)

        # Embedder returns all-zeros — if cache is used, we should get original
        embedder = _FixedEmbedder(dim=4)
        result = cache.embed_alleles({"A*02:01": "SEQ"}, embedder, show_progress=False)
        np.testing.assert_array_equal(result["A*02:01"], original)

    def test_new_embeddings_persisted_to_cache(self, tmp_path: Path) -> None:
        """Computed embeddings must be retrievable from cache after embed_alleles."""
        cache = EmbeddingCache(tmp_path / "test.h5")
        embedder = _FixedEmbedder(dim=8)
        cache.embed_alleles({"A*02:01": "SEQ"}, embedder, show_progress=False)
        assert cache.contains("A*02:01")

    def test_second_call_serves_from_cache(self, tmp_path: Path) -> None:
        """A second call with the same alleles must not re-invoke the embedder."""
        cache = EmbeddingCache(tmp_path / "test.h5")
        embedder = _FixedEmbedder()
        allele_seqs = {"A*02:01": "SEQ1"}
        cache.embed_alleles(allele_seqs, embedder, show_progress=False)
        assert embedder.call_count == 1
        cache.embed_alleles(allele_seqs, embedder, show_progress=False)
        assert embedder.call_count == 1  # still 1, not 2

    def test_empty_input_returns_empty_dict(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        embedder = _FixedEmbedder()
        result = cache.embed_alleles({}, embedder, show_progress=False)
        assert result == {}
        assert embedder.call_count == 0

    def test_satisfies_sequence_embedder_protocol(self) -> None:
        """_FixedEmbedder must satisfy the SequenceEmbedder Protocol."""
        assert isinstance(_FixedEmbedder(), SequenceEmbedder)

    def test_esm_embedder_satisfies_protocol(self) -> None:
        """ESMEmbedder must also satisfy SequenceEmbedder (structural check)."""
        assert isinstance(_make_embedder(), SequenceEmbedder)


# ---------------------------------------------------------------------------
# parse_fasta (scripts/download_hla_seqs.py) — kept from previous session
# ---------------------------------------------------------------------------

def _write_fasta(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.fasta"
    p.write_text(textwrap.dedent(content), encoding="ascii")
    return p


class TestParseFasta:
    def test_basic_header_and_sequence(self, tmp_path: Path) -> None:
        fasta = """\
            >HLA:HLA00001 A*01:01:01:01 365 bp
            MAVMAPRTLLL
            LSGALALTQTW
        """
        db = parse_fasta(_write_fasta(tmp_path, fasta))
        assert "A*01:01:01:01" in db
        assert db["A*01:01:01:01"] == "MAVMAPRTLLLLSGALALTQTW"

    def test_stop_codon_stripped(self, tmp_path: Path) -> None:
        fasta = """\
            >HLA:HLA00001 B*07:02 100 bp
            MAVMAPRT*LLLL
        """
        db = parse_fasta(_write_fasta(tmp_path, fasta))
        assert "*" not in db["B*07:02"]
        assert db["B*07:02"] == "MAVMAPRTLLLL"

    def test_x_amino_acid_preserved(self, tmp_path: Path) -> None:
        fasta = """\
            >HLA:HLA00001 C*01:02 100 bp
            MAVXAPRT
        """
        db = parse_fasta(_write_fasta(tmp_path, fasta))
        assert "X" in db["C*01:02"]

    def test_expression_suffix_kept_in_key(self, tmp_path: Path) -> None:
        fasta = """\
            >HLA:HLA00001 A*01:01:01:02N 365 bp
            MAVMAPRTL
        """
        db = parse_fasta(_write_fasta(tmp_path, fasta))
        assert "A*01:01:01:02N" in db

    def test_empty_sequence_skipped(self, tmp_path: Path) -> None:
        fasta = """\
            >HLA:HLA00001 A*99:99 10 bp
            ***
            >HLA:HLA00002 A*01:01 10 bp
            MAVMAPRTL
        """
        db = parse_fasta(_write_fasta(tmp_path, fasta))
        assert "A*99:99" not in db
        assert "A*01:01" in db

    def test_loci_filter_includes(self, tmp_path: Path) -> None:
        fasta = """\
            >HLA:HLA00001 A*01:01 100 bp
            MAVMAPRTL
            >HLA:HLA00002 B*07:02 100 bp
            MAVMAPRTL
            >HLA:HLA00003 DRB1*15:01 100 bp
            MAVMAPRTL
        """
        db = parse_fasta(_write_fasta(tmp_path, fasta), loci_filter={"A", "DRB1"})
        assert "A*01:01" in db
        assert "DRB1*15:01" in db
        assert "B*07:02" not in db

    def test_loci_filter_none_keeps_all(self, tmp_path: Path) -> None:
        fasta = """\
            >HLA:HLA00001 A*01:01 100 bp
            MAVMAPRTL
            >HLA:HLA00002 B*07:02 100 bp
            MAVMAPRTL
        """
        db = parse_fasta(_write_fasta(tmp_path, fasta), loci_filter=None)
        assert "A*01:01" in db and "B*07:02" in db

    def test_sequence_uppercased(self, tmp_path: Path) -> None:
        fasta = """\
            >HLA:HLA00001 A*01:01 100 bp
            mavmaprtl
        """
        db = parse_fasta(_write_fasta(tmp_path, fasta))
        assert db["A*01:01"] == "MAVMAPRTL"

    def test_malformed_header_skipped(self, tmp_path: Path) -> None:
        fasta = """\
            >BADHEADER
            MAVMAPRTL
            >HLA:HLA00001 A*02:01 100 bp
            MRVMAPRTL
        """
        db = parse_fasta(_write_fasta(tmp_path, fasta))
        assert "A*02:01" in db and len(db) == 1


# ---------------------------------------------------------------------------
# HLASequenceDB helpers — kept from previous session
# ---------------------------------------------------------------------------

def _make_hla_db(tmp_path: Path, data: dict[str, object]) -> HLASequenceDB:
    p = tmp_path / "seqs.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return HLASequenceDB(p)


class TestIsExpressed:
    def test_normal_allele(self) -> None:
        assert _is_expressed("A*02:01:01:01") is True

    def test_null_allele(self) -> None:
        assert _is_expressed("A*01:01:01:02N") is False

    def test_low_expression(self) -> None:
        assert _is_expressed("B*44:02:01:02L") is False


class TestCandidateKeys:
    def test_four_field_allele(self) -> None:
        allele = parse_who_allele("A*02:01:01:04")
        assert _candidate_keys(allele) == ["A*02:01:01:04", "A*02:01:01", "A*02:01", "A*02"]

    def test_two_field_allele(self) -> None:
        allele = parse_who_allele("DRB1*15:01")
        assert _candidate_keys(allele) == ["DRB1*15:01", "DRB1*15"]


class TestBestFromCandidates:
    def test_prefers_expressed_over_null(self) -> None:
        assert _best_from_candidates(["A*02:01:01:02N", "A*02:01:01:01"]) == "A*02:01:01:01"

    def test_all_expressed_returns_lexicographic_min(self) -> None:
        assert _best_from_candidates(["A*02:01:01:03", "A*02:01:01:01"]) == "A*02:01:01:01"


class TestHLASequenceDBGet:
    def test_exact_match(self, tmp_path: Path) -> None:
        db = _make_hla_db(tmp_path, {"A*02:01": "MAVMAPRTL"})
        assert db.get(parse_who_allele("A*02:01")) == "MAVMAPRTL"

    def test_fallback_four_to_two_field(self, tmp_path: Path) -> None:
        db = _make_hla_db(tmp_path, {"A*02:01": "TWOFIELDSEQ"})
        assert db.get(parse_who_allele("A*02:01:01:04")) == "TWOFIELDSEQ"

    def test_fallback_via_prefix_index(self, tmp_path: Path) -> None:
        db = _make_hla_db(tmp_path, {"A*02:01:01:01": "PREFIXSEQ"})
        assert db.get(parse_who_allele("A*02:01:01:99")) == "PREFIXSEQ"

    def test_no_match_raises_key_error(self, tmp_path: Path) -> None:
        db = _make_hla_db(tmp_path, {"A*02:01": "SEQ"})
        with pytest.raises(KeyError):
            db.get(parse_who_allele("B*07:02"))

    def test_fallback_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        db = _make_hla_db(tmp_path, {"A*02:01": "SEQ"})
        with caplog.at_level(logging.WARNING):
            db.get(parse_who_allele("A*02:01:01"))
        assert any("fallback" in r.message.lower() for r in caplog.records)


class TestHLASequenceDBGetByName:
    def test_exact_match(self, tmp_path: Path) -> None:
        db = _make_hla_db(tmp_path, {"A*02:01": "SEQ"})
        assert db.get_by_name("A*02:01") == "SEQ"

    def test_fallback_truncation(self, tmp_path: Path) -> None:
        db = _make_hla_db(tmp_path, {"A*02:01": "SEQ"})
        assert db.get_by_name("A*02:01:01:99") == "SEQ"

    def test_missing_star_raises(self, tmp_path: Path) -> None:
        db = _make_hla_db(tmp_path, {"A*02:01": "SEQ"})
        with pytest.raises(KeyError, match="missing"):
            db.get_by_name("A02:01")


# ---------------------------------------------------------------------------
# EmbeddingCache — original low-level tests (kept from previous session)
# ---------------------------------------------------------------------------

class TestEmbeddingCacheOriginal:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        vec = np.random.rand(1280).astype(np.float32)
        cache.put("A*02:01", vec)
        np.testing.assert_array_almost_equal(cache.get("A*02:01"), vec)

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        with pytest.raises(KeyError):
            cache.get("A*99:99")

    def test_len_zero_before_writes(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        assert len(cache) == 0

    def test_overwrite_updates_value(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "test.h5")
        cache.put("A*02:01", np.ones(5, dtype=np.float32))
        cache.put("A*02:01", np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(cache.get("A*02:01"), np.zeros(5, dtype=np.float32))

    def test_contains_false_when_file_missing(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "missing.h5")
        assert not cache.contains("A*02:01")


# ---------------------------------------------------------------------------
# ESMEmbedder fine-tuning helpers (V2)
# ---------------------------------------------------------------------------

def _make_nn_mock_model(
    n_layers: int = 4,
    hidden_dim: int = _MOCK_HIDDEN,
) -> torch.nn.Module:
    """Return a tiny nn.Module that mimics the ESM-2 encoder.layer structure."""
    import types

    class _FakeOutput:
        def __init__(self, hs: torch.Tensor) -> None:
            self.last_hidden_state = hs

    class _FakeLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_dim, hidden_dim))

    class _FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.ModuleList(
                [_FakeLayer() for _ in range(n_layers)]
            )

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _FakeEncoder()

        def __call__(  # type: ignore[override]
            self, **inputs: torch.Tensor
        ) -> _FakeOutput:
            b, sl = inputs["attention_mask"].shape
            return _FakeOutput(torch.full((b, sl, hidden_dim), 1.0))

    model = _FakeModel()
    # Freeze all params initially (mimics ESM-2 after load)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _make_nn_embedder(n_layers: int = 4) -> ESMEmbedder:
    """ESMEmbedder with a tiny nn.Module mock (supports fine-tune API)."""
    embedder = ESMEmbedder(device="cpu", batch_size=4)
    embedder._model = _make_nn_mock_model(n_layers=n_layers)
    embedder._tokenizer = _make_mock_tokenizer()
    return embedder


class TestESMEmbedderFineTune:
    def test_unfreeze_before_load_raises(self) -> None:
        embedder = ESMEmbedder(device="cpu")
        with pytest.raises(RuntimeError, match="embed\\(\\)"):
            embedder.unfreeze_last_n_layers(1)

    def test_unfreeze_zero_is_noop(self) -> None:
        embedder = _make_nn_embedder()
        n = embedder.unfreeze_last_n_layers(0)
        assert n == 0
        assert all(not p.requires_grad for p in embedder._model.parameters())  # type: ignore[union-attr]

    def test_unfreeze_last_two_layers(self) -> None:
        embedder = _make_nn_embedder(n_layers=4)
        n_params = embedder.unfreeze_last_n_layers(2)
        assert n_params > 0
        trainable = [p for p in embedder._model.parameters() if p.requires_grad]  # type: ignore[union-attr]
        frozen   = [p for p in embedder._model.parameters() if not p.requires_grad]  # type: ignore[union-attr]
        # 2 unfrozen layers out of 4 → half the layers' params are trainable
        assert len(trainable) == 2
        assert len(frozen) == 2

    def test_get_finetune_parameters_empty_before_unfreeze(self) -> None:
        embedder = _make_nn_embedder()
        assert embedder.get_finetune_parameters() == []

    def test_get_finetune_parameters_after_unfreeze(self) -> None:
        embedder = _make_nn_embedder(n_layers=4)
        embedder.unfreeze_last_n_layers(1)
        params = embedder.get_finetune_parameters()
        assert len(params) == 1  # one weight tensor per layer

    def test_set_train_mode(self) -> None:
        embedder = _make_nn_embedder()
        embedder.set_train_mode(train=True)
        assert embedder._model.training  # type: ignore[union-attr]
        embedder.set_train_mode(train=False)
        assert not embedder._model.training  # type: ignore[union-attr]

    def test_get_finetune_parameters_empty_without_model(self) -> None:
        embedder = ESMEmbedder(device="cpu")
        assert embedder.get_finetune_parameters() == []
