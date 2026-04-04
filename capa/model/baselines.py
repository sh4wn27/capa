"""Baseline competing-risks survival models for comparison with CAPA.

Four baselines
--------------
1. **Fine-Gray** — subdistribution hazard regression (one lifelines
   CoxPHFitter per event, fitted on the IPCW-modified risk set that treats
   competing events as censored at max follow-up with inverse-probability
   weights).
2. **Random Survival Forest** — cause-specific RSF via scikit-survival (one
   forest per event type).  Falls back gracefully with an ImportError
   message if scikit-survival is not installed.
3. **Cause-specific Cox PH** — separate lifelines CoxPHFitter per event,
   standard censoring; CIF computed from cause-specific hazards.
4. **CAPA-OneHot** — the full CAPA cross-attention architecture trained from
   scratch with *learned* allele embeddings (no ESM-2 pretraining).  This
   ablation isolates the contribution of structural language-model embeddings.

Shared interface
----------------
Every tabular baseline (Fine-Gray, RSF, Cox) exposes::

    baseline.fit(X_train, times_train, event_types_train)
    cif = baseline.predict_cif(X_test, time_bins)   # → (n, K, T)

The deep baseline (CAPA-OneHot) exposes::

    baseline.fit(train_loader, val_loader, ...)
    cif = baseline.predict_cif(donor_idx, recip_idx, clin, time_bins)

Use :func:`make_tabular_features` and :func:`make_onehot_loaders` (defined in
``scripts/compare_baselines.py``) to convert a common synthetic or real
dataset into the right format for each model.

Fine-Gray implementation note
------------------------------
Lifelines does not natively implement the Fine-Gray subdistribution hazard
model.  We apply the standard IPCW-weighted Cox trick (Geskus 2011, Stat Med):

* Event-k subjects: kept in dataset, ``event=True``, ``weight=1``.
* Censored subjects: kept, ``event=False``, ``weight=1``.
* Competing-event subjects: set ``duration=max_time``, ``event=False``,
  ``weight=G(t_j)`` where G is the Kaplan-Meier censoring distribution
  evaluated at their event time.  This corrects for the artificially extended
  follow-up while retaining them in the subdistribution risk set.

References
----------
* Fine & Gray (1999), J. Am. Stat. Assoc. — subdistribution hazard model.
* Geskus (2011), Stat. Med. — IPCW weighted Cox for Fine-Gray.
* Ishwaran et al. (2008), Ann. Appl. Stat. — Random Survival Forests.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from capa.model.interaction import CrossAttentionInteraction
from capa.model.losses import deephit_loss
from capa.model.survival import DeepHitHead
from capa.training.evaluate import concordance_index

logger = logging.getLogger(__name__)

# Type aliases
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


# ---------------------------------------------------------------------------
# Kaplan-Meier helper (censoring distribution for IPCW)
# ---------------------------------------------------------------------------

def _km_censoring(times: F64, event_types: I64) -> tuple[F64, F64]:
    """Kaplan-Meier estimate of the censoring survival function G(t).

    Parameters
    ----------
    times : F64, shape (n,)
    event_types : I64, shape (n,)
        0 = censored, >0 = any event.

    Returns
    -------
    km_times : F64
        Unique censoring times (sorted).
    km_G : F64
        G(t) — probability of *not* being censored by time t.
    """
    censored_indicator = (event_types == 0).astype(bool)
    order = np.argsort(times)
    t_s = times[order]
    c_s = censored_indicator[order]
    n = len(times)
    G = 1.0
    km_t: list[float] = []
    km_g: list[float] = []
    for i, (t, c) in enumerate(zip(t_s, c_s)):
        if c:
            n_at_risk = n - i
            G *= 1.0 - 1.0 / n_at_risk
            km_t.append(float(t))
            km_g.append(G)
    if not km_t:
        return np.array([0.0]), np.array([1.0])
    return np.array(km_t), np.array(km_g)


def _km_eval(km_t: F64, km_g: F64, t: float) -> float:
    """Evaluate KM step function at t (left-continuous)."""
    if t < km_t[0]:
        return 1.0
    idx = int(np.searchsorted(km_t, t, side="right")) - 1
    return float(km_g[idx])


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaselineModel(ABC):
    """Abstract base class for all CAPA baseline models.

    All baselines share a common ``name`` property that identifies the model
    in comparison tables and JSON results.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable model name."""
        ...


# ---------------------------------------------------------------------------
# 1. Fine-Gray baseline (IPCW-weighted Cox via lifelines)
# ---------------------------------------------------------------------------

class FineGrayBaseline(BaselineModel):
    """Subdistribution hazard regression (Fine & Gray 1999).

    One lifelines CoxPHFitter is trained per competing event using the
    IPCW-modified risk set (Geskus 2011).

    Parameters
    ----------
    num_events : int
        Number of competing events (indices 1..num_events in ``event_types``).
    penalizer : float
        Ridge penalizer passed to lifelines CoxPHFitter (prevents singular
        information matrix on small datasets).
    """

    def __init__(self, num_events: int = 3, penalizer: float = 0.1) -> None:
        self._num_events = num_events
        self._penalizer = penalizer
        self._fitters: list[Any] = []   # one CoxPHFitter per event

    @property
    def name(self) -> str:
        return "Fine-Gray"

    def fit(
        self,
        X: pd.DataFrame,
        times: F64,
        event_types: I64,
    ) -> None:
        """Fit one subdistribution Cox model per event.

        Parameters
        ----------
        X : pd.DataFrame, shape (n, p)
            Covariate matrix.  Column names are preserved for prediction.
        times : F64, shape (n,)
            Observed times (real-valued or bin indices).
        event_types : I64, shape (n,)
            0 = censored, 1..K = event type.
        """
        from lifelines import CoxPHFitter

        max_time = float(times.max()) + 1.0
        km_t, km_g = _km_censoring(times, event_types)

        self._fitters = []
        self._feature_cols = list(X.columns)

        for k in range(1, self._num_events + 1):
            durations: list[float] = []
            observed: list[int] = []
            weights: list[float] = []

            for i in range(len(times)):
                etype = int(event_types[i])
                t_i = float(times[i])
                if etype == k:
                    durations.append(t_i)
                    observed.append(1)
                    weights.append(1.0)
                elif etype == 0:
                    durations.append(t_i)
                    observed.append(0)
                    weights.append(1.0)
                else:
                    # Competing event: extend to max_time, IPCW weight
                    g_ti = _km_eval(km_t, km_g, t_i)
                    w = max(g_ti, 1e-6)
                    durations.append(max_time)
                    observed.append(0)
                    weights.append(w)

            df = X.copy()
            df["_duration"] = durations
            df["_event"] = observed
            df["_weight"] = weights

            fitter = CoxPHFitter(penalizer=self._penalizer)
            try:
                fitter.fit(
                    df,
                    duration_col="_duration",
                    event_col="_event",
                    weights_col="_weight",
                )
            except Exception as exc:
                logger.warning("Fine-Gray event %d fit failed: %s — using null model", k, exc)
                fitter = None  # type: ignore[assignment]
            self._fitters.append(fitter)

        logger.info("Fine-Gray: fitted %d cause-specific models", self._num_events)

    def predict_cif(
        self,
        X: pd.DataFrame,
        time_bins: F64,
    ) -> F64:
        """Predict CIF for all events at all time bins.

        Parameters
        ----------
        X : pd.DataFrame, shape (n, p)
        time_bins : F64, shape (T,)

        Returns
        -------
        F64, shape (n, K, T)
        """
        n = len(X)
        T = len(time_bins)
        K = self._num_events
        cif = np.zeros((n, K, T), dtype=np.float64)

        for k, fitter in enumerate(self._fitters):
            if fitter is None:
                continue
            # predict_survival_function returns a DataFrame: index=times, cols=subjects
            sf = fitter.predict_survival_function(X, times=time_bins)
            # sf.values: shape (T, n); CIF = 1 - S
            cif[:, k, :] = np.clip(1.0 - sf.values.T, 0.0, 1.0)

        return cif


# ---------------------------------------------------------------------------
# 2. Cause-specific Cox PH baseline
# ---------------------------------------------------------------------------

class CoxPHBaseline(BaselineModel):
    """Cause-specific Cox PH, one fitter per competing event.

    Competing events are treated as censored at their actual event time
    (standard cause-specific censoring).  CIF is computed by converting
    cause-specific cumulative hazards via the relationship::

        CIF_k(t) = 1 - exp(-H_k(t))   [marginal, ignoring other causes]

    This is an approximation — the proper formula integrates the overall
    survival against the cause-specific hazard — but it is exact when
    hazards for competing events are small.

    Parameters
    ----------
    num_events : int
    penalizer : float
    """

    def __init__(self, num_events: int = 3, penalizer: float = 0.1) -> None:
        self._num_events = num_events
        self._penalizer = penalizer
        self._fitters: list[Any] = []

    @property
    def name(self) -> str:
        return "Cox PH (cause-specific)"

    def fit(
        self,
        X: pd.DataFrame,
        times: F64,
        event_types: I64,
    ) -> None:
        """Fit one Cox PH per event, treating competing events as censored."""
        from lifelines import CoxPHFitter

        self._fitters = []
        self._feature_cols = list(X.columns)

        for k in range(1, self._num_events + 1):
            observed = (event_types == k).astype(int)
            df = X.copy()
            df["_duration"] = times.astype(float)
            df["_event"] = observed

            fitter = CoxPHFitter(penalizer=self._penalizer)
            try:
                fitter.fit(df, duration_col="_duration", event_col="_event")
            except Exception as exc:
                logger.warning("CoxPH event %d fit failed: %s — null model", k, exc)
                fitter = None  # type: ignore[assignment]
            self._fitters.append(fitter)

        logger.info("Cox PH: fitted %d cause-specific models", self._num_events)

    def predict_cif(
        self,
        X: pd.DataFrame,
        time_bins: F64,
    ) -> F64:
        """Return CIF (n, K, T) from cause-specific cumulative hazards."""
        n = len(X)
        T = len(time_bins)
        K = self._num_events
        cif = np.zeros((n, K, T), dtype=np.float64)

        for k, fitter in enumerate(self._fitters):
            if fitter is None:
                continue
            sf = fitter.predict_survival_function(X, times=time_bins)
            cif[:, k, :] = np.clip(1.0 - sf.values.T, 0.0, 1.0)

        return cif


# ---------------------------------------------------------------------------
# 3. Random Survival Forest baseline (scikit-survival)
# ---------------------------------------------------------------------------

class RandomSurvivalForestBaseline(BaselineModel):
    """Cause-specific Random Survival Forest using scikit-survival.

    Requires ``scikit-survival`` to be installed::

        uv add scikit-survival   # or: pip install scikit-survival

    One RSF is fitted per competing event (cause-specific censoring of other
    events, same as CoxPHBaseline).

    Parameters
    ----------
    num_events : int
    n_estimators : int
        Number of trees per forest.
    min_samples_leaf : int
        Minimum samples required to be at a leaf node.
    random_state : int
    """

    def __init__(
        self,
        num_events: int = 3,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self._num_events = num_events
        self._n_estimators = n_estimators
        self._min_samples_leaf = min_samples_leaf
        self._random_state = random_state
        self._forests: list[Any] = []
        self._train_times: list[F64] = []   # per-event, for interpolation grid

    @property
    def name(self) -> str:
        return "Random Survival Forest"

    @staticmethod
    def _check_sksurv() -> None:
        try:
            import sksurv  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "scikit-survival is required for RandomSurvivalForestBaseline.\n"
                "Install it with:  uv add scikit-survival\n"
                "or:               pip install scikit-survival"
            ) from e

    def fit(
        self,
        X: np.ndarray,
        times: F64,
        event_types: I64,
    ) -> None:
        """Fit one RSF per event.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Numerical feature matrix.
        times : F64, shape (n,)
        event_types : I64, shape (n,)
        """
        self._check_sksurv()
        from sksurv.ensemble import RandomSurvivalForest

        self._forests = []
        self._train_times = []

        for k in range(1, self._num_events + 1):
            event_k = (event_types == k).astype(bool)
            # scikit-survival structured array: (event: bool, time: float)
            y = np.array(
                list(zip(event_k, times.astype(float))),
                dtype=[("event", bool), ("time", float)],
            )
            rsf = RandomSurvivalForest(
                n_estimators=self._n_estimators,
                min_samples_leaf=self._min_samples_leaf,
                random_state=self._random_state,
                n_jobs=-1,
            )
            try:
                rsf.fit(X, y)
            except Exception as exc:
                logger.warning("RSF event %d fit failed: %s — null model", k, exc)
                rsf = None  # type: ignore[assignment]
            self._forests.append(rsf)
            self._train_times.append(times.copy())

        logger.info("RSF: fitted %d cause-specific forests", self._num_events)

    def predict_cif(
        self,
        X: np.ndarray,
        time_bins: F64,
    ) -> F64:
        """Predict CIF from RSF cumulative hazard functions.

        RSF survival functions are piecewise-constant and defined only at
        training times; values are interpolated to *time_bins*.
        """
        self._check_sksurv()
        n = len(X)
        T = len(time_bins)
        K = self._num_events
        cif = np.zeros((n, K, T), dtype=np.float64)

        for k, rsf in enumerate(self._forests):
            if rsf is None:
                continue
            surv_fns = rsf.predict_survival_function(X)
            for i, fn in enumerate(surv_fns):
                # fn is a StepFunction with .x (times) and .y (survival values)
                # interpolate to our time_bins grid
                s_vals = np.interp(time_bins, fn.x, fn.y, left=1.0, right=fn.y[-1])
                cif[i, k, :] = np.clip(1.0 - s_vals, 0.0, 1.0)

        return cif


# ---------------------------------------------------------------------------
# 4. CAPA-OneHot baseline (ablation: CAPA without ESM-2)
# ---------------------------------------------------------------------------

class AlleleVocabulary:
    """Maps allele strings to integer indices.

    Index 0 is reserved for the ``<unk>`` (unknown/missing) token.

    Parameters
    ----------
    None — build from data using :meth:`fit`.
    """

    UNK = "<unk>"
    UNK_IDX = 0

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {self.UNK: self.UNK_IDX}

    def fit(self, allele_strings: list[str]) -> "AlleleVocabulary":
        """Add all unique alleles to the vocabulary.

        Parameters
        ----------
        allele_strings : list[str]
            Flat list of allele name strings from training data.

        Returns
        -------
        self
        """
        for a in allele_strings:
            if a and a not in self._vocab:
                self._vocab[a] = len(self._vocab)
        return self

    def encode(self, allele_name: str | None) -> int:
        """Return the integer index for an allele (0 if unknown)."""
        if allele_name is None:
            return self.UNK_IDX
        return self._vocab.get(allele_name, self.UNK_IDX)

    def encode_batch(self, allele_names: list[str | None]) -> list[int]:
        """Encode a list of allele names to integer indices."""
        return [self.encode(a) for a in allele_names]

    @property
    def size(self) -> int:
        """Vocabulary size including the ``<unk>`` token."""
        return len(self._vocab)

    def __len__(self) -> int:
        return self.size


class CAPAOneHotModel(nn.Module):
    """CAPA with trainable allele embeddings in place of ESM-2.

    Architecture is identical to :class:`~capa.model.capa_model.CAPAModel`
    except that HLA allele strings are mapped through a learnable
    ``nn.Embedding`` table (one shared table for donor and recipient) rather
    than being looked up from a frozen ESM-2 cache.

    This isolates the contribution of structural priors in ESM-2 embeddings
    versus the cross-attention interaction architecture itself.

    Parameters
    ----------
    vocab_size : int
        Allele vocabulary size (from :class:`AlleleVocabulary`).
    embedding_dim : int
        Dimension of learnable allele embeddings.
    n_loci : int
        Number of HLA loci.
    raw_clinical_dim : int
        Dimensionality of the *raw* clinical feature vector (as provided by
        the DataLoader).  Projected to ``clinical_dim`` internally.
    clinical_dim : int
        Output dimensionality of the internal clinical projection.
    interaction_dim : int
        Output dim of the interaction network.
    num_events : int
    time_bins : int
    num_heads : int
    num_layers : int
    dropout : float
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        n_loci: int = 5,
        raw_clinical_dim: int = 4,
        clinical_dim: int = 32,
        interaction_dim: int = 128,
        num_events: int = 3,
        time_bins: int = 100,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._num_events = num_events
        self._time_bins = time_bins

        # Learnable allele embeddings — shared across donor and recipient
        self.allele_embed = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=AlleleVocabulary.UNK_IDX
        )

        self.interaction = CrossAttentionInteraction(
            embedding_dim=embedding_dim,
            interaction_dim=interaction_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Simple clinical projection: raw features → clinical_dim
        self.clinical_proj = nn.Sequential(
            nn.Linear(raw_clinical_dim, clinical_dim),
            nn.GELU(),
            nn.LayerNorm(clinical_dim),
        )

        combined_dim = interaction_dim + clinical_dim
        self.survival_head = DeepHitHead(
            input_dim=combined_dim,
            num_events=num_events,
            time_bins=time_bins,
            dropout=dropout,
        )

    def forward(
        self,
        donor_idx: Tensor,           # (batch, n_loci) int64
        recip_idx: Tensor,           # (batch, n_loci) int64
        raw_clinical: Tensor,        # (batch, raw_clinical_dim) float32
    ) -> Tensor:
        """Forward pass: allele indices + raw clinical → logits (B, K, T)."""
        donor_emb = self.allele_embed(donor_idx)   # (B, L, D)
        recip_emb = self.allele_embed(recip_idx)   # (B, L, D)
        interaction_feats = self.interaction(donor_emb, recip_emb)
        clin_feats = self.clinical_proj(raw_clinical)
        combined = torch.cat([interaction_feats, clin_feats], dim=-1)
        return self.survival_head(combined)

    def cif(
        self,
        donor_idx: Tensor,
        recip_idx: Tensor,
        raw_clinical: Tensor,
    ) -> Tensor:
        """Return CIF (batch, num_events, time_bins) in [0, 1]."""
        out = self.forward(donor_idx, recip_idx, raw_clinical)
        batch = out.shape[0]
        joint = F.softmax(out.view(batch, -1), dim=-1).view(
            batch, self._num_events, self._time_bins
        )
        return torch.cumsum(joint, dim=2)


class CAPAOneHotBaseline(BaselineModel):
    """CAPA ablation: identical architecture, learned allele embeddings.

    Removes the ESM-2 structural prior to isolate the value of protein
    language model embeddings over random initialisation.

    Parameters
    ----------
    num_events : int
    time_bins : int
    embedding_dim : int
        Embedding dimension for learned allele representations.  Smaller than
        ESM-2 1280 to keep the parameter count similar to other baselines.
    n_loci : int
    raw_clinical_dim : int
        Dimensionality of the raw clinical features in the DataLoader
        (``clinical_features`` column).  Default 4: age_recipient, age_donor,
        cd34_dose, sex_mismatch.
    clinical_dim : int
        Internal projection dimension for clinical features.
    interaction_dim : int
    max_epochs : int
    patience : int
    learning_rate : float
    batch_size : int
    device : str
    """

    def __init__(
        self,
        num_events: int = 3,
        time_bins: int = 100,
        embedding_dim: int = 64,
        n_loci: int = 5,
        raw_clinical_dim: int = 4,
        clinical_dim: int = 32,
        interaction_dim: int = 64,
        max_epochs: int = 50,
        patience: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha: float = 0.5,
        sigma: float = 0.1,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self._num_events = num_events
        self._time_bins = time_bins
        self._embedding_dim = embedding_dim
        self._n_loci = n_loci
        self._raw_clinical_dim = raw_clinical_dim
        self._clinical_dim = clinical_dim
        self._interaction_dim = interaction_dim
        self._max_epochs = max_epochs
        self._patience = patience
        self._lr = learning_rate
        self._wd = weight_decay
        self._alpha = alpha
        self._sigma = sigma
        self._batch_size = batch_size
        self._device = torch.device(device)

        self.vocab: AlleleVocabulary | None = None
        self.model: CAPAOneHotModel | None = None
        self._history: dict[str, list[float]] = {}

    @property
    def name(self) -> str:
        return "CAPA-OneHot (ablation)"

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        vocab: AlleleVocabulary,
    ) -> None:
        """Train the one-hot CAPA model.

        Parameters
        ----------
        train_loader : DataLoader
            Each batch must be a dict with keys:
            ``donor_allele_indices`` (B, L) int64,
            ``recipient_allele_indices`` (B, L) int64,
            ``clinical_features`` (B, C) float32,
            ``event_times`` (B,) int64,
            ``event_types`` (B,) int64.
        val_loader : DataLoader
            Same format.
        vocab : AlleleVocabulary
            Pre-built vocabulary for the training alleles.
        """
        self.vocab = vocab
        self.model = CAPAOneHotModel(
            vocab_size=vocab.size,
            embedding_dim=self._embedding_dim,
            n_loci=self._n_loci,
            raw_clinical_dim=self._raw_clinical_dim,
            clinical_dim=self._clinical_dim,
            interaction_dim=self._interaction_dim,
            num_events=self._num_events,
            time_bins=self._time_bins,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self._lr, weight_decay=self._wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._max_epochs, eta_min=self._lr * 1e-2
        )

        best_cindex = -1.0
        epochs_no_improve = 0
        history: dict[str, list[float]] = {
            "train_loss": [], "val_cindex": []
        }

        for epoch in range(1, self._max_epochs + 1):
            # --- Train ---
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = self._batch_loss(batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            train_loss = total_loss / max(n_batches, 1)

            # --- Validate ---
            val_cindex = self._val_cindex(val_loader)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_cindex"].append(val_cindex)

            logger.info(
                "CAPA-OneHot  epoch %d/%d  loss=%.4f  val_c=%.4f",
                epoch, self._max_epochs, train_loss, val_cindex,
            )

            if val_cindex > best_cindex:
                best_cindex = val_cindex
                epochs_no_improve = 0
                self._best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self._patience:
                    logger.info(
                        "CAPA-OneHot early stop at epoch %d (best c=%.4f)",
                        epoch, best_cindex,
                    )
                    break

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(
                {k: v.to(self._device) for k, v in self._best_state.items()}
            )
        self._history = history
        logger.info("CAPA-OneHot training complete (best val C-index=%.4f)", best_cindex)

    def _batch_loss(self, batch: dict[str, Tensor]) -> Tensor:
        donor_idx = batch["donor_allele_indices"].to(self._device)
        recip_idx = batch["recipient_allele_indices"].to(self._device)
        clinical  = batch["clinical_features"].to(self._device)
        times     = batch["event_times"].to(self._device)
        types     = batch["event_types"].to(self._device)
        assert self.model is not None
        logits = self.model(donor_idx, recip_idx, clinical)
        return deephit_loss(logits, times, types, alpha=self._alpha, sigma=self._sigma)

    @torch.no_grad()
    def _val_cindex(self, val_loader: DataLoader[Any]) -> float:
        assert self.model is not None
        self.model.eval()
        all_times: list[np.ndarray] = []
        all_risks: list[np.ndarray] = []
        all_obs:   list[np.ndarray] = []
        for batch in val_loader:
            donor_idx = batch["donor_allele_indices"].to(self._device)
            recip_idx = batch["recipient_allele_indices"].to(self._device)
            clinical  = batch["clinical_features"].to(self._device)
            cif_t = self.model.cif(donor_idx, recip_idx, clinical)
            med_bin = cif_t.shape[2] // 2
            risk = cif_t[:, 0, med_bin].cpu().numpy()
            types = batch["event_types"].numpy()
            all_times.append(batch["event_times"].numpy().astype(float))
            all_risks.append(risk)
            all_obs.append((types == 1).astype(bool))
        if not all_times:
            return 0.5
        t = np.concatenate(all_times)
        r = np.concatenate(all_risks)
        o = np.concatenate(all_obs)
        if o.sum() < 2:
            return 0.5
        c = concordance_index(t, r, o)
        return float(c) if not np.isnan(c) else 0.5

    @torch.no_grad()
    def predict_cif(
        self,
        donor_idx: np.ndarray,    # (n, n_loci) int
        recip_idx: np.ndarray,    # (n, n_loci) int
        clinical_features: np.ndarray,  # (n, C) float
        time_bins: F64,
    ) -> F64:
        """Return CIF array (n, K, T).

        Parameters
        ----------
        donor_idx : int array (n, n_loci)
            Allele indices from :class:`AlleleVocabulary`.
        recip_idx : int array (n, n_loci)
        clinical_features : float array (n, C)
            Pre-encoded clinical features (output of
            :meth:`~capa.model.capa_model.ClinicalEncoder.prepare_inputs`
            flattened to a matrix, or any ``clinical_dim``-dim features).
        time_bins : F64, shape (T,)
            Requested time bins; must be a subset of ``range(self._time_bins)``.

        Returns
        -------
        F64, shape (n, K, T)
        """
        assert self.model is not None
        self.model.eval()

        d_t = torch.tensor(donor_idx, dtype=torch.long, device=self._device)
        r_t = torch.tensor(recip_idx, dtype=torch.long, device=self._device)
        c_t = torch.tensor(clinical_features, dtype=torch.float32, device=self._device)

        cif_t = self.model.cif(d_t, r_t, c_t)   # (n, K, time_bins)
        cif_np = cif_t.cpu().numpy()              # (n, K, T_model)

        # Select requested time bin indices
        T_model = cif_np.shape[2]
        indices = np.clip(
            np.searchsorted(np.arange(T_model, dtype=np.float64), time_bins),
            0, T_model - 1,
        ).astype(int)
        return cif_np[:, :, indices]              # (n, K, T)
