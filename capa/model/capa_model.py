"""Full CAPA model: cached embeddings → interaction → survival.

Pipeline
--------
::

    donor HLA strings ──► HLA sequence lookup ──► embedding cache/ESM-2 ──┐
                                                                            ▼
    recipient HLA strings ──────────────────────────────────────────► CrossAttentionInteraction
                                                                            │
    clinical covariates ──► ClinicalEncoder ──────────────────────────── concat
                                                                            │
                                                                            ▼
                                                                    DeepHitHead (or CSH)
                                                                            │
                                                                            ▼
                                                                   CIF curves per event

Key design choices
------------------
* **Frozen embeddings**: ESM-2 is not a submodule of this model.  Embeddings
  arrive as ``(batch, n_loci, embedding_dim)`` tensors that are already
  detached from any ESM-2 graph.  All learnable parameters live in the
  interaction network, clinical encoder, and survival head.

* **Differentiable forward pass**: ``forward()`` is end-to-end differentiable
  through the learnable parts.  The embedding lookup in ``predict()`` uses
  ``torch.no_grad()`` since ESM-2 is frozen.

* **Prediction without ESM-2**: ``predict()`` accepts pre-computed embedding
  dicts (``allele_name → ndarray``) directly, so inference can run at test
  time without loading the 650 M-parameter model.

* **Attention weight access**: ``get_attention_weights()`` returns the
  ``AttentionWeights`` named tuple from the most recent forward pass via
  the interaction network's ``last_attn_weights`` property.

* **Survival head**: defaults to :class:`~capa.model.survival.DeepHitHead`
  (joint distribution).  Pass ``survival_type="cause_specific"`` to use
  :class:`~capa.model.survival.CauseSpecificHazardHead` instead.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from capa.model.interaction import AttentionWeights, CrossAttentionInteraction
from capa.model.survival import CauseSpecificHazardHead, DeepHitHead, hazards_to_cif

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clinical covariate encoder
# ---------------------------------------------------------------------------

# Ordered categories for each categorical covariate.
# Index 0 is always "unknown" — used for missing or unrecognised values.
DISEASE_CATEGORIES: list[str] = [
    "unknown", "ALL", "AML", "CML", "MDS", "NHL", "HD", "AA", "MM", "other",
]
CONDITIONING_CATEGORIES: list[str] = ["unknown", "MAC", "RIC", "NMA"]
DONOR_TYPE_CATEGORIES: list[str] = ["unknown", "MSD", "MUD", "MMUD", "haplo", "cord"]
STEM_CELL_SOURCE_CATEGORIES: list[str] = ["unknown", "BM", "PBSC", "cord"]

_CAT_SPECS: list[tuple[str, list[str]]] = [
    ("disease", DISEASE_CATEGORIES),
    ("conditioning", CONDITIONING_CATEGORIES),
    ("donor_type", DONOR_TYPE_CATEGORIES),
    ("stem_cell_source", STEM_CELL_SOURCE_CATEGORIES),
]

# Continuous features handled by ClinicalEncoder (in order)
_CONTINUOUS_KEYS: list[str] = ["age_recipient", "age_donor", "cd34_dose"]

# Build reverse-lookup dicts once at module load time
_CAT_MAPS: dict[str, dict[str, int]] = {
    name: {cat: i for i, cat in enumerate(cats)}
    for name, cats in _CAT_SPECS
}


class ClinicalEncoder(nn.Module):
    """Encode structured clinical covariates to a fixed-dim embedding vector.

    Handles three covariate types:

    * **Continuous** — ``age_recipient``, ``age_donor``, ``cd34_dose``:
      passed as normalised floats (see :meth:`prepare_inputs`).
    * **Binary** — ``sex_mismatch`` (0 or 1): concatenated with continuous.
    * **Categorical** — ``disease``, ``conditioning``, ``donor_type``,
      ``stem_cell_source``: mapped to learnable embeddings.

    All features are concatenated and projected to *output_dim* via a
    two-layer GELU MLP with LayerNorm.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the output clinical feature vector.
    cat_embed_dim : int
        Embedding dimension for each categorical variable.
    dropout : float
        Dropout probability in the MLP.
    """

    # age_recipient, age_donor, cd34_dose + sex_mismatch binary
    _N_CONT: int = len(_CONTINUOUS_KEYS) + 1

    def __init__(
        self,
        output_dim: int = 32,
        cat_embed_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._output_dim = output_dim

        # One embedding table per categorical variable.
        # padding_idx is intentionally omitted: the "unknown" class (index 0)
        # is learnable so its representation improves during training.
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(len(cats), cat_embed_dim)
             for _, cats in _CAT_SPECS]
        )

        mlp_input_dim = len(_CAT_SPECS) * cat_embed_dim + self._N_CONT
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    @property
    def output_dim(self) -> int:
        """Dimensionality of the output vector."""
        return self._output_dim

    def forward(self, cont: Tensor, cat_indices: Tensor) -> Tensor:
        """Encode clinical covariates.

        Parameters
        ----------
        cont : Tensor
            Continuous + binary features, shape ``(batch, N_CONT)`` where
            ``N_CONT == 4`` (age_recipient/100, age_donor/100, cd34_dose/10,
            sex_mismatch).
        cat_indices : Tensor
            Integer category indices, shape ``(batch, N_CAT)`` where
            ``N_CAT == 4`` (disease, conditioning, donor_type,
            stem_cell_source).  Index 0 = "unknown".

        Returns
        -------
        Tensor
            Clinical feature vector of shape ``(batch, output_dim)``.
        """
        embeds = [
            emb(cat_indices[:, i])
            for i, emb in enumerate(self.cat_embeddings)
        ]
        cat_concat = torch.cat(embeds, dim=-1)        # (batch, N_CAT * cat_embed_dim)
        combined = torch.cat([cont, cat_concat], dim=-1)
        return self.mlp(combined)

    @staticmethod
    def prepare_inputs(
        features: dict[str, Any] | None,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Convert a raw clinical feature dict to ``(cont, cat_indices)`` tensors.

        All missing or unrecognised values are imputed with zeros / "unknown".

        Parameters
        ----------
        features : dict[str, Any] | None
            Covariate mapping.  Any subset of keys is accepted; extras are
            silently ignored.  Recognised keys: ``age_recipient``,
            ``age_donor``, ``cd34_dose``, ``sex_mismatch``, ``disease``,
            ``conditioning``, ``donor_type``, ``stem_cell_source``.
        device : torch.device | str
            Target device for the returned tensors.

        Returns
        -------
        cont : Tensor
            Shape ``(1, 4)`` float32.
        cat_indices : Tensor
            Shape ``(1, 4)`` int64.
        """
        d = features or {}

        def _float(key: str, scale: float = 1.0) -> float:
            v = d.get(key)
            try:
                return float(v) / scale if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        cont_vals = [
            _float("age_recipient", scale=100.0),
            _float("age_donor",     scale=100.0),
            _float("cd34_dose",     scale=10.0),
            float(bool(d.get("sex_mismatch", 0))),
        ]
        cont = torch.tensor([cont_vals], dtype=torch.float32, device=device)

        def _cat_idx(name: str) -> int:
            val = d.get(name)
            if val is None:
                return 0
            return _CAT_MAPS[name].get(str(val), 0)

        cat_vals = [_cat_idx(name) for name, _ in _CAT_SPECS]
        cat_indices = torch.tensor([cat_vals], dtype=torch.long, device=device)

        return cont, cat_indices


# ---------------------------------------------------------------------------
# Full CAPA model
# ---------------------------------------------------------------------------

_LOCI_DEFAULT: list[str] = ["A", "B", "C", "DRB1", "DQB1"]
_EVENT_NAMES_DEFAULT: list[str] = ["gvhd", "relapse", "trm"]


class CAPAModel(nn.Module):
    """End-to-end CAPA model for competing-risks survival prediction.

    The three learnable components are:

    1. :class:`CrossAttentionInteraction` — bidirectional cross-attention
       between donor and recipient allele embeddings.
    2. :class:`ClinicalEncoder` — embeds structured clinical covariates.
    3. A survival head (:class:`~capa.model.survival.DeepHitHead` or
       :class:`~capa.model.survival.CauseSpecificHazardHead`).

    ESM-2 embeddings are **not** part of this module and are assumed to be
    pre-computed and detached (see :meth:`predict`).

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of ESM-2 HLA allele embeddings.
    loci : list[str] | None
        HLA loci to include.  Defaults to ``["A","B","C","DRB1","DQB1"]``.
    clinical_dim : int
        Output dimensionality of the :class:`ClinicalEncoder`.
    interaction_dim : int
        Output dimensionality of the interaction network.
    survival_type : str
        ``"deephit"`` (default) or ``"cause_specific"``.
    num_events : int
        Number of competing events.
    time_bins : int
        Number of discrete time bins.
    event_names : list[str] | None
        Human-readable names for each event.
    num_heads : int
        Number of cross-attention heads.
    num_layers : int
        Number of cross-attention layers.
    dropout : float
        Dropout probability throughout.
    cat_embed_dim : int
        Categorical embedding dimension in :class:`ClinicalEncoder`.
    n_loci : int | None
        Deprecated alias for ``len(loci)``; kept for backward compatibility.
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        loci: list[str] | None = None,
        clinical_dim: int = 32,
        interaction_dim: int = 128,
        survival_type: str = "deephit",
        num_events: int = 3,
        time_bins: int = 100,
        event_names: list[str] | None = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        cat_embed_dim: int = 8,
        # Backward-compatibility alias
        n_loci: int | None = None,
    ) -> None:
        super().__init__()

        self._loci: list[str] = loci if loci is not None else _LOCI_DEFAULT
        self._n_loci: int = n_loci if n_loci is not None else len(self._loci)
        self.embedding_dim = embedding_dim
        self._time_bins = time_bins
        self._num_events = num_events
        self._event_names: list[str] = (
            event_names if event_names is not None
            else _EVENT_NAMES_DEFAULT[:num_events]
        )
        if len(self._event_names) != num_events:
            raise ValueError(
                f"len(event_names)={len(self._event_names)} must equal "
                f"num_events={num_events}"
            )
        if survival_type not in {"deephit", "cause_specific"}:
            raise ValueError(
                f"Unknown survival_type {survival_type!r}. "
                "Choose 'deephit' or 'cause_specific'."
            )
        self._survival_type = survival_type

        # --- Sub-modules ---
        self.interaction = CrossAttentionInteraction(
            embedding_dim=embedding_dim,
            interaction_dim=interaction_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.clinical_encoder = ClinicalEncoder(
            output_dim=clinical_dim,
            cat_embed_dim=cat_embed_dim,
            dropout=dropout,
        )

        combined_dim = interaction_dim + clinical_dim
        if survival_type == "deephit":
            self.survival_head: nn.Module = DeepHitHead(
                input_dim=combined_dim,
                num_events=num_events,
                time_bins=time_bins,
                dropout=dropout,
            )
        else:
            self.survival_head = CauseSpecificHazardHead(
                input_dim=combined_dim,
                num_events=num_events,
                time_bins=time_bins,
                dropout=dropout,
            )

        # Optional inference infrastructure
        self._seq_db: Any | None = None
        self._cache: Any | None = None

    # ------------------------------------------------------------------
    # Inference infrastructure injection
    # ------------------------------------------------------------------

    def set_inference_components(
        self,
        seq_db: Any | None = None,
        cache: Any | None = None,
    ) -> None:
        """Attach an HLA sequence database and embedding cache for :meth:`predict`.

        Parameters
        ----------
        seq_db : HLASequenceDB | None
            Sequence lookup DB (used to resolve allele → protein sequence).
        cache : EmbeddingCache | None
            HDF5 embedding cache with ``contains(allele)`` and ``get(allele)``.
        """
        self._seq_db = seq_db
        self._cache = cache

    # ------------------------------------------------------------------
    # Forward (differentiable through learnable parameters)
    # ------------------------------------------------------------------

    def forward(
        self,
        donor_embeddings: Tensor,
        recipient_embeddings: Tensor,
        clinical_features: Tensor,
    ) -> Tensor:
        """Differentiable forward pass through all learnable components.

        ESM-2 embeddings should be detached before being passed here.

        Parameters
        ----------
        donor_embeddings : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.
        recipient_embeddings : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.
        clinical_features : Tensor
            Encoded clinical features, shape ``(batch, clinical_dim)``.
            Typically the output of :meth:`ClinicalEncoder.forward`.

        Returns
        -------
        Tensor
            Raw logits / sub-hazards, shape ``(batch, num_events, time_bins)``.
        """
        interaction_feats = self.interaction(donor_embeddings, recipient_embeddings)
        combined = torch.cat([interaction_feats, clinical_features], dim=-1)
        return self.survival_head(combined)

    def forward_from_dict(
        self,
        donor_embeddings: Tensor,
        recipient_embeddings: Tensor,
        clinical_dict: dict[str, Any] | None = None,
    ) -> Tensor:
        """Forward pass that encodes a raw clinical dict internally.

        Parameters
        ----------
        donor_embeddings : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.
        recipient_embeddings : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.
        clinical_dict : dict[str, Any] | None
            Raw clinical features dict; missing values are imputed.

        Returns
        -------
        Tensor
            Raw logits / sub-hazards, shape ``(batch, num_events, time_bins)``.
        """
        device = donor_embeddings.device
        batch = donor_embeddings.shape[0]
        cont, cats = ClinicalEncoder.prepare_inputs(clinical_dict, device=device)
        if batch > 1:
            cont = cont.expand(batch, -1)
            cats = cats.expand(batch, -1)
        clin_feats = self.clinical_encoder(cont, cats)
        return self.forward(donor_embeddings, recipient_embeddings, clin_feats)

    # ------------------------------------------------------------------
    # CIF output
    # ------------------------------------------------------------------

    def cif(
        self,
        donor_embeddings: Tensor,
        recipient_embeddings: Tensor,
        clinical_features: Tensor,
    ) -> Tensor:
        """Return cumulative incidence functions.

        Parameters
        ----------
        donor_embeddings : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.
        recipient_embeddings : Tensor
            Shape ``(batch, n_loci, embedding_dim)``.
        clinical_features : Tensor
            Shape ``(batch, clinical_dim)``.

        Returns
        -------
        Tensor
            CIF values in ``[0, 1]``, shape ``(batch, num_events, time_bins)``.
            Monotone non-decreasing along the time dimension.
        """
        out = self.forward(donor_embeddings, recipient_embeddings, clinical_features)
        if self._survival_type == "deephit":
            batch = out.shape[0]
            joint = F.softmax(out.view(batch, -1), dim=-1).view(
                batch, self._num_events, self._time_bins
            )
            return torch.cumsum(joint, dim=2)
        else:
            return hazards_to_cif(out)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def get_attention_weights(self) -> AttentionWeights | None:
        """Attention weights from the most recent forward pass.

        Returns
        -------
        AttentionWeights | None
            Named tuple with ``donor_to_recip`` and ``recip_to_donor`` —
            each a list of ``(batch, n_loci_q, n_loci_kv)`` tensors, one
            per cross-attention layer.  ``None`` before the first forward.
        """
        return self.interaction.last_attn_weights

    # ------------------------------------------------------------------
    # High-level inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        donor_hla: dict[str, str],
        recipient_hla: dict[str, str],
        clinical: dict[str, Any] | None = None,
        *,
        donor_embeddings: dict[str, npt.NDArray[np.float32]] | None = None,
        recipient_embeddings: dict[str, npt.NDArray[np.float32]] | None = None,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Predict competing-risk curves for one donor-recipient pair.

        Embedding resolution priority (for each locus):

        1. **Provided embeddings** dict ``{allele_name: ndarray}`` (test-friendly,
           no cache required).
        2. **Cache lookup** via the cache attached with
           :meth:`set_inference_components`.
        3. **Zero fallback** — a warning is logged and a zero vector is used.

        Parameters
        ----------
        donor_hla : dict[str, str]
            Mapping ``locus → allele_name``.
        recipient_hla : dict[str, str]
            Same format.
        clinical : dict[str, Any] | None
            Clinical covariates.  Missing keys are imputed.  Recognised keys:
            ``age_recipient``, ``age_donor``, ``cd34_dose``, ``sex_mismatch``,
            ``disease``, ``conditioning``, ``donor_type``, ``stem_cell_source``.
        donor_embeddings : dict[str, ndarray] | None
            Pre-computed embeddings keyed by allele name.
        recipient_embeddings : dict[str, ndarray] | None
            Pre-computed embeddings keyed by allele name.
        device : str | torch.device | None
            Inference device.  Defaults to the device of the first model
            parameter, or CPU.

        Returns
        -------
        dict[str, Any]
            * One key per event (``"gvhd"``, ``"relapse"``, ``"trm"``, …):
              ``{"cif": list[float], "risk_score": float}``.
            * ``"attention_weights"``: ``{"donor_to_recip": ...,
              "recip_to_donor": ...}`` (last-layer, averaged over heads) or
              ``None``.
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        device = torch.device(device)

        def _resolve(
            hla_dict: dict[str, str],
            provided: dict[str, npt.NDArray[np.float32]] | None,
        ) -> Tensor:
            vecs: list[npt.NDArray[np.float32]] = []
            for locus in self._loci:
                allele = hla_dict.get(locus)
                vec: npt.NDArray[np.float32] | None = None

                if provided is not None and allele is not None:
                    vec = provided.get(allele)

                if vec is None and allele is not None and self._cache is not None:
                    if self._cache.contains(allele):
                        vec = self._cache.get(allele)

                if vec is None:
                    if allele:
                        logger.warning(
                            "No embedding for locus %s allele %s — using zeros",
                            locus, allele,
                        )
                    vec = np.zeros(self.embedding_dim, dtype=np.float32)

                vecs.append(vec)

            return torch.from_numpy(np.stack(vecs, axis=0)).unsqueeze(0).to(device)

        donor_t   = _resolve(donor_hla, donor_embeddings)
        recip_t   = _resolve(recipient_hla, recipient_embeddings)

        cont, cats = ClinicalEncoder.prepare_inputs(clinical, device=device)
        clin_feats = self.clinical_encoder(cont, cats)

        self.eval()
        cif_t = self.cif(donor_t, recip_t, clin_feats)  # (1, num_events, time_bins)
        cif_np = cif_t[0].cpu().numpy()

        result: dict[str, Any] = {}
        for k, name in enumerate(self._event_names):
            curve = cif_np[k].tolist()
            result[name] = {"cif": curve, "risk_score": float(curve[-1])}

        attn = self.get_attention_weights()
        if attn is not None:
            result["attention_weights"] = {
                "donor_to_recip": attn.donor_to_recip[-1][0].cpu().tolist(),
                "recip_to_donor": attn.recip_to_donor[-1][0].cpu().tolist(),
            }
        else:
            result["attention_weights"] = None

        return result
