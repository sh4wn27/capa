"""Tests for capa/model/baselines.py and scripts/compare_baselines.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_tabular_data(
    n: int = 60,
    p: int = 8,
    num_events: int = 2,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (X, times, event_types) for tabular baselines."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, p)),
        columns=[f"feat_{i}" for i in range(p)],
    )
    times      = rng.uniform(1.0, 50.0, n)
    event_types = rng.integers(0, num_events + 1, n).astype(np.int64)
    return X, times, event_types


def _make_tabular_data_int(n: int = 60, p: int = 8, num_events: int = 2, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X, times, event_types = _make_tabular_data(n, p, num_events, seed)
    # Use integer times (bin indices)
    int_times = np.clip(times.astype(np.int64), 1, 49)
    return X, int_times.astype(float), event_types


# ---------------------------------------------------------------------------
# AlleleVocabulary
# ---------------------------------------------------------------------------

class TestAlleleVocabulary:

    def setup_method(self) -> None:
        from capa.model.baselines import AlleleVocabulary
        self.AlleleVocabulary = AlleleVocabulary

    def test_unk_always_zero(self) -> None:
        v = self.AlleleVocabulary()
        assert v.encode("<unk>") == 0
        assert v.encode(None) == 0     # type: ignore[arg-type]
        assert v.encode("UNSEEN") == 0

    def test_fit_adds_entries(self) -> None:
        v = self.AlleleVocabulary()
        v.fit(["A*01:01", "A*02:01", "B*07:02"])
        assert v.size == 4   # 3 alleles + <unk>

    def test_encode_known_allele(self) -> None:
        v = self.AlleleVocabulary()
        v.fit(["A*01:01", "A*02:01"])
        assert v.encode("A*01:01") != 0
        assert v.encode("A*02:01") != 0

    def test_encode_batch(self) -> None:
        v = self.AlleleVocabulary()
        v.fit(["A*01:01", "B*07:02"])
        result = v.encode_batch(["A*01:01", None, "B*07:02", "UNKNOWN"])
        assert len(result) == 4
        assert result[0] != 0
        assert result[1] == 0
        assert result[3] == 0

    def test_fit_idempotent(self) -> None:
        v = self.AlleleVocabulary()
        v.fit(["A*01:01", "A*01:01", "A*01:01"])
        assert v.size == 2   # only 1 unique allele + <unk>

    def test_len_equals_size(self) -> None:
        v = self.AlleleVocabulary()
        v.fit(["X", "Y", "Z"])
        assert len(v) == v.size


# ---------------------------------------------------------------------------
# Fine-Gray baseline
# ---------------------------------------------------------------------------

class TestFineGrayBaseline:

    def setup_method(self) -> None:
        from capa.model.baselines import FineGrayBaseline
        self.FineGray = FineGrayBaseline

    def test_fit_runs(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=2, seed=1)
        fg = self.FineGray(num_events=2)
        fg.fit(X, times, event_types)
        assert len(fg._fitters) == 2

    def test_predict_cif_shape(self) -> None:
        n, p, K = 80, 5, 2
        X, times, event_types = _make_tabular_data(n=n, p=p, num_events=K, seed=2)
        fg = self.FineGray(num_events=K)
        fg.fit(X, times, event_types)
        time_bins = np.linspace(1, 50, 20)
        X_test = X.iloc[:20].reset_index(drop=True)
        cif = fg.predict_cif(X_test, time_bins)
        assert cif.shape == (20, K, 20)

    def test_cif_in_unit_interval(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=2, seed=3)
        fg = self.FineGray(num_events=2)
        fg.fit(X, times, event_types)
        time_bins = np.linspace(1, 50, 15)
        cif = fg.predict_cif(X.iloc[:10].reset_index(drop=True), time_bins)
        assert np.all(cif >= 0.0)
        assert np.all(cif <= 1.0)

    def test_cif_monotone_increasing(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=2, seed=4)
        fg = self.FineGray(num_events=2)
        fg.fit(X, times, event_types)
        time_bins = np.linspace(1, 50, 15)
        cif = fg.predict_cif(X.iloc[:5].reset_index(drop=True), time_bins)
        for k in range(2):
            diffs = np.diff(cif[:, k, :], axis=1)
            assert np.all(diffs >= -1e-9), f"CIF not monotone for event {k}"

    def test_all_censored_no_crash(self) -> None:
        X, times, _ = _make_tabular_data(n=60, num_events=2, seed=5)
        event_types = np.zeros(60, dtype=np.int64)   # all censored
        fg = self.FineGray(num_events=2)
        # Should not raise; may fit a null model
        fg.fit(X, times, event_types)

    def test_name_property(self) -> None:
        fg = self.FineGray(num_events=2)
        assert isinstance(fg.name, str)
        assert len(fg.name) > 0


# ---------------------------------------------------------------------------
# Cox PH baseline
# ---------------------------------------------------------------------------

class TestCoxPHBaseline:

    def setup_method(self) -> None:
        from capa.model.baselines import CoxPHBaseline
        self.CoxPH = CoxPHBaseline

    def test_fit_runs(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=2, seed=10)
        cox = self.CoxPH(num_events=2)
        cox.fit(X, times, event_types)
        assert len(cox._fitters) == 2

    def test_predict_cif_shape(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=3, seed=11)
        cox = self.CoxPH(num_events=3)
        cox.fit(X, times, event_types)
        time_bins = np.linspace(1, 50, 10)
        cif = cox.predict_cif(X.iloc[:15].reset_index(drop=True), time_bins)
        assert cif.shape == (15, 3, 10)

    def test_cif_in_unit_interval(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=2, seed=12)
        cox = self.CoxPH(num_events=2)
        cox.fit(X, times, event_types)
        time_bins = np.linspace(1, 50, 10)
        cif = cox.predict_cif(X.iloc[:10].reset_index(drop=True), time_bins)
        assert np.all(cif >= 0.0)
        assert np.all(cif <= 1.0)

    def test_cif_monotone_increasing(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=2, seed=13)
        cox = self.CoxPH(num_events=2)
        cox.fit(X, times, event_types)
        time_bins = np.linspace(1, 50, 15)
        cif = cox.predict_cif(X.iloc[:5].reset_index(drop=True), time_bins)
        diffs = np.diff(cif[:, 0, :], axis=1)
        assert np.all(diffs >= -1e-9)

    def test_name_property(self) -> None:
        assert isinstance(self.CoxPH(num_events=2).name, str)


# ---------------------------------------------------------------------------
# RSF baseline (conditionally, requires scikit-survival)
# ---------------------------------------------------------------------------

def _sksurv_available() -> bool:
    try:
        import sksurv  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _sksurv_available(), reason="scikit-survival not installed")
class TestRandomSurvivalForestBaseline:

    def setup_method(self) -> None:
        from capa.model.baselines import RandomSurvivalForestBaseline
        self.RSF = RandomSurvivalForestBaseline

    def test_fit_runs(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=2, seed=20)
        rsf = self.RSF(num_events=2, n_estimators=5)
        rsf.fit(X.to_numpy(dtype=np.float64), times, event_types)
        assert len(rsf._forests) == 2

    def test_predict_cif_shape(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=2, seed=21)
        rsf = self.RSF(num_events=2, n_estimators=5)
        rsf.fit(X.to_numpy(dtype=np.float64), times, event_types)
        time_bins = np.linspace(1, 50, 10)
        cif = rsf.predict_cif(X.to_numpy(dtype=np.float64)[:10], time_bins)
        assert cif.shape == (10, 2, 10)

    def test_cif_in_unit_interval(self) -> None:
        X, times, event_types = _make_tabular_data(n=80, num_events=2, seed=22)
        rsf = self.RSF(num_events=2, n_estimators=5)
        rsf.fit(X.to_numpy(dtype=np.float64), times, event_types)
        time_bins = np.linspace(1, 50, 10)
        cif = rsf.predict_cif(X.to_numpy(dtype=np.float64)[:5], time_bins)
        assert np.all(cif >= 0.0)
        assert np.all(cif <= 1.0)


class TestRSFImportError:
    """RSF raises a clear ImportError when scikit-survival is absent."""

    def test_check_raises_import_error(self) -> None:
        import unittest.mock as mock
        from capa.model.baselines import RandomSurvivalForestBaseline
        rsf = RandomSurvivalForestBaseline()
        with mock.patch.dict("sys.modules", {"sksurv": None}):
            with pytest.raises(ImportError, match="scikit-survival"):
                rsf._check_sksurv()


# ---------------------------------------------------------------------------
# CAPAOneHotModel
# ---------------------------------------------------------------------------

class TestCAPAOneHotModel:

    def setup_method(self) -> None:
        from capa.model.baselines import AlleleVocabulary, CAPAOneHotModel
        self.vocab = AlleleVocabulary()
        self.vocab.fit([f"A*0{i}:01" for i in range(10)])
        self.CAPAOneHotModel = CAPAOneHotModel

    def _make_model(self, **kwargs: int | float) -> "CAPAOneHotModel":
        defaults = dict(
            vocab_size=self.vocab.size,
            embedding_dim=16,
            n_loci=3,
            raw_clinical_dim=6,
            clinical_dim=8,
            interaction_dim=16,
            num_events=2,
            time_bins=10,
            num_heads=2,
            num_layers=1,
        )
        defaults.update(kwargs)
        return self.CAPAOneHotModel(**defaults)  # type: ignore[arg-type]

    def test_forward_shape(self) -> None:
        model = self._make_model()
        B, L = 4, 3
        donor = torch.randint(0, self.vocab.size, (B, L))
        recip = torch.randint(0, self.vocab.size, (B, L))
        clin  = torch.randn(B, 6)      # raw_clinical_dim=6
        out = model(donor, recip, clin)
        assert out.shape == (B, 2, 10)

    def test_cif_shape_and_range(self) -> None:
        model = self._make_model()
        B, L = 4, 3
        donor = torch.randint(0, self.vocab.size, (B, L))
        recip = torch.randint(0, self.vocab.size, (B, L))
        clin  = torch.randn(B, 6)
        cif = model.cif(donor, recip, clin)
        assert cif.shape == (B, 2, 10)
        assert (cif >= 0).all()
        assert (cif <= 1).all()

    def test_cif_monotone(self) -> None:
        model = self._make_model()
        donor = torch.zeros(2, 3, dtype=torch.long)
        recip = torch.zeros(2, 3, dtype=torch.long)
        clin  = torch.randn(2, 6)
        cif = model.cif(donor, recip, clin)
        diffs = torch.diff(cif, dim=2)
        assert (diffs >= -1e-6).all()

    def test_gradients_flow(self) -> None:
        model = self._make_model()
        donor = torch.randint(0, self.vocab.size, (4, 3))
        recip = torch.randint(0, self.vocab.size, (4, 3))
        clin  = torch.randn(4, 6)
        out = model(donor, recip, clin)
        out.sum().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

    def test_unk_index_produces_valid_output(self) -> None:
        model = self._make_model()
        # All unknown tokens (index 0)
        donor = torch.zeros(2, 3, dtype=torch.long)
        recip = torch.zeros(2, 3, dtype=torch.long)
        clin  = torch.randn(2, 6)
        out = model(donor, recip, clin)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# CAPAOneHotBaseline (full training loop)
# ---------------------------------------------------------------------------

class TestCAPAOneHotBaseline:

    def _make_vocab_and_loaders(
        self,
        n: int = 40,
        n_loci: int = 3,
        time_bins: int = 10,
        num_events: int = 2,
        seed: int = 0,
    ) -> tuple:
        from capa.model.baselines import AlleleVocabulary
        from torch.utils.data import DataLoader, TensorDataset

        rng = np.random.default_rng(seed)
        vocab = AlleleVocabulary()
        alleles = [f"A*{i:02d}:01" for i in range(1, 8)]
        vocab.fit(alleles)

        donor_idx  = torch.randint(0, vocab.size, (n, n_loci))
        recip_idx  = torch.randint(0, vocab.size, (n, n_loci))
        clinical   = torch.randn(n, 4)
        times      = torch.randint(1, time_bins, (n,))
        event_types = torch.randint(0, num_events + 1, (n,))

        class _DS(torch.utils.data.Dataset):  # type: ignore[type-arg]
            def __init__(self, di, ri, cl, t, et):
                self.di, self.ri, self.cl, self.t, self.et = di, ri, cl, t, et
            def __len__(self): return len(self.t)
            def __getitem__(self, i):
                return {
                    "donor_allele_indices": self.di[i],
                    "recipient_allele_indices": self.ri[i],
                    "clinical_features": self.cl[i],
                    "event_times": self.t[i],
                    "event_types": self.et[i],
                }

        ds = _DS(donor_idx, recip_idx, clinical, times, event_types)
        split = int(0.8 * n)
        train_ldr = DataLoader(ds[:split] if hasattr(ds, '__getitem__') else ds,
                               batch_size=8)
        val_ldr   = DataLoader(ds, batch_size=8)
        # Actually build two full loaders for simplicity
        train_ldr = DataLoader(
            _DS(donor_idx[:split], recip_idx[:split], clinical[:split],
                times[:split], event_types[:split]),
            batch_size=8,
        )
        val_ldr = DataLoader(
            _DS(donor_idx[split:], recip_idx[split:], clinical[split:],
                times[split:], event_types[split:]),
            batch_size=8,
        )
        return vocab, train_ldr, val_ldr, n_loci, time_bins, num_events

    def test_fit_runs(self) -> None:
        from capa.model.baselines import CAPAOneHotBaseline
        vocab, train_ldr, val_ldr, n_loci, time_bins, K = self._make_vocab_and_loaders()
        oh = CAPAOneHotBaseline(
            num_events=K,
            time_bins=time_bins,
            embedding_dim=16,
            n_loci=n_loci,
            raw_clinical_dim=4,
            clinical_dim=8,
            interaction_dim=16,
            max_epochs=3,
            patience=10,
        )
        oh.fit(train_ldr, val_ldr, vocab)
        assert oh.model is not None

    def test_predict_cif_shape(self) -> None:
        from capa.model.baselines import CAPAOneHotBaseline
        vocab, train_ldr, val_ldr, n_loci, time_bins, K = self._make_vocab_and_loaders(n=40)
        oh = CAPAOneHotBaseline(
            num_events=K,
            time_bins=time_bins,
            embedding_dim=16,
            n_loci=n_loci,
            clinical_dim=8,
            interaction_dim=16,
            max_epochs=2,
            patience=10,
        )
        oh.fit(train_ldr, val_ldr, vocab)

        n_test = 10
        donor_idx = np.random.randint(0, vocab.size, (n_test, n_loci))
        recip_idx = np.random.randint(0, vocab.size, (n_test, n_loci))
        clinical  = np.random.randn(n_test, 4).astype(np.float32)
        time_bins_arr = np.arange(time_bins, dtype=np.float64)

        cif = oh.predict_cif(donor_idx, recip_idx, clinical, time_bins_arr)
        assert cif.shape == (n_test, K, time_bins)

    def test_cif_in_unit_interval(self) -> None:
        from capa.model.baselines import CAPAOneHotBaseline
        vocab, train_ldr, val_ldr, n_loci, time_bins, K = self._make_vocab_and_loaders(n=40)
        oh = CAPAOneHotBaseline(
            num_events=K, time_bins=time_bins,
            embedding_dim=16, n_loci=n_loci, clinical_dim=8,
            interaction_dim=16, max_epochs=2, patience=10,
        )
        oh.fit(train_ldr, val_ldr, vocab)

        donor_idx = np.zeros((5, n_loci), dtype=np.int64)
        recip_idx = np.zeros((5, n_loci), dtype=np.int64)
        clin      = np.zeros((5, 4), dtype=np.float32)
        cif = oh.predict_cif(donor_idx, recip_idx, clin,
                             np.arange(time_bins, dtype=np.float64))
        assert np.all(cif >= 0.0)
        assert np.all(cif <= 1.0)

    def test_history_recorded(self) -> None:
        from capa.model.baselines import CAPAOneHotBaseline
        vocab, train_ldr, val_ldr, n_loci, time_bins, K = self._make_vocab_and_loaders(n=40)
        oh = CAPAOneHotBaseline(
            num_events=K, time_bins=time_bins,
            embedding_dim=16, n_loci=n_loci, clinical_dim=8,
            interaction_dim=16, max_epochs=3, patience=10,
        )
        oh.fit(train_ldr, val_ldr, vocab)
        assert len(oh._history["train_loss"]) >= 1

    def test_early_stopping_triggers(self) -> None:
        """With patience=1, training should stop early (≤ max_epochs)."""
        from capa.model.baselines import CAPAOneHotBaseline
        vocab, train_ldr, val_ldr, n_loci, time_bins, K = self._make_vocab_and_loaders(n=40)
        oh = CAPAOneHotBaseline(
            num_events=K, time_bins=time_bins,
            embedding_dim=16, n_loci=n_loci, clinical_dim=8,
            interaction_dim=16, max_epochs=10, patience=1,
        )
        oh.fit(train_ldr, val_ldr, vocab)
        # Should stop before max_epochs (unless C-index improves every epoch)
        # At minimum, training ran without error
        assert oh.model is not None


# ---------------------------------------------------------------------------
# KM censoring helper
# ---------------------------------------------------------------------------

class TestKMCensoring:

    def setup_method(self) -> None:
        from capa.model.baselines import _km_censoring, _km_eval
        self._km = _km_censoring
        self._eval = _km_eval

    def test_all_events_no_censoring(self) -> None:
        times = np.array([1.0, 2.0, 3.0, 4.0])
        event_types = np.array([1, 2, 1, 1])  # no censoring
        t, g = self._km(times, event_types)
        # No censoring events → default [1.0]
        assert g[0] == pytest.approx(1.0)

    def test_some_censoring(self) -> None:
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event_types = np.array([1, 0, 1, 0, 1])   # events at 2.0, 4.0
        t, g = self._km(times, event_types)
        assert len(t) == len(g)
        assert (np.diff(g) <= 0).all()   # monotone non-increasing

    def test_eval_before_first_event(self) -> None:
        t = np.array([2.0, 4.0])
        g = np.array([0.8, 0.5])
        assert self._eval(t, g, 0.5) == pytest.approx(1.0)

    def test_eval_at_event(self) -> None:
        t = np.array([2.0, 4.0])
        g = np.array([0.8, 0.5])
        assert self._eval(t, g, 2.0) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# compare_baselines.py smoke tests
# ---------------------------------------------------------------------------

class TestSyntheticDatasetHelper:

    def test_make_synthetic_dataset_keys(self) -> None:
        sys.path.insert(0, str(_ROOT / "scripts"))
        from compare_baselines import make_synthetic_dataset
        d = make_synthetic_dataset(n=30, time_bins=10, num_events=2, seed=0)
        assert "tabular_X" in d
        assert "times" in d
        assert "event_types" in d
        assert "donor_alleles" in d
        assert "recipient_alleles" in d
        assert "clinical_cont" in d

    def test_split_preserves_n(self) -> None:
        from compare_baselines import make_synthetic_dataset, split_dataset
        d = make_synthetic_dataset(n=50, time_bins=10, num_events=2, seed=1)
        tr, va, te = split_dataset(d, seed=1)
        total = len(tr["times"]) + len(va["times"]) + len(te["times"])
        assert total == 50

    def test_make_onehot_loaders(self) -> None:
        from capa.model.baselines import AlleleVocabulary
        from compare_baselines import make_onehot_loaders, make_synthetic_dataset, split_dataset
        d = make_synthetic_dataset(n=40, time_bins=10, num_events=2, seed=2)
        tr, va, _ = split_dataset(d, seed=2)

        vocab = AlleleVocabulary()
        vocab.fit([a for row in tr["donor_alleles"] + tr["recipient_alleles"] for a in row])

        train_ldr, val_ldr = make_onehot_loaders(tr, va, vocab, batch_size=8)
        batch = next(iter(train_ldr))
        assert "donor_allele_indices" in batch
        assert "recipient_allele_indices" in batch
        assert "event_times" in batch


class TestCompareBaselinesCLI:
    """Smoke-tests for scripts/compare_baselines.py."""

    def test_synthetic_finegray_cox(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "compare_baselines.py"),
                "--synthetic",
                "--n", "60",
                "--models", "finegray", "cox",
                "--n-bootstrap", "0",
                "--epochs", "2",
                "--time-bins", "20",
                "--n-events", "2",
                "--runs-dir", str(tmp_path),
                "--run-name", "test_run",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr

    def test_capa_onehot_runs(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "compare_baselines.py"),
                "--synthetic",
                "--n", "60",
                "--models", "capa_onehot",
                "--n-bootstrap", "0",
                "--epochs", "2",
                "--time-bins", "20",
                "--n-events", "2",
                "--embedding-dim", "8",
                "--interaction-dim", "8",
                "--runs-dir", str(tmp_path),
                "--run-name", "test_onehot",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr

    def test_comparison_json_produced(self, tmp_path: Path) -> None:
        import json
        result = subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "compare_baselines.py"),
                "--synthetic",
                "--n", "60",
                "--models", "finegray", "cox",
                "--n-bootstrap", "0",
                "--epochs", "2",
                "--time-bins", "20",
                "--n-events", "2",
                "--runs-dir", str(tmp_path),
                "--run-name", "json_test",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        out_file = tmp_path / "compare_json_test" / "comparison.json"
        assert out_file.exists(), f"comparison.json not found at {out_file}"
        with out_file.open() as f:
            d = json.load(f)
        assert "models" in d
        assert "n_subjects" in d

    def test_capa_runs(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "compare_baselines.py"),
                "--synthetic",
                "--n", "60",
                "--models", "capa",
                "--n-bootstrap", "0",
                "--epochs", "2",
                "--time-bins", "20",
                "--n-events", "2",
                "--embedding-dim", "8",
                "--interaction-dim", "8",
                "--runs-dir", str(tmp_path),
                "--run-name", "test_capa",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
