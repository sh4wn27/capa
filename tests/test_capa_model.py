"""Integration tests for capa/model/capa_model.py.

Tests cover:
  ClinicalEncoder
    - prepare_inputs: all-missing, all-present, partial, unknown categoricals
    - forward: shape, dtype, no nan, gradients
    - output_dim property

  CAPAModel — construction
    - default construction
    - survival_type="cause_specific"
    - custom loci list
    - n_loci backward-compat kwarg
    - invalid survival_type raises ValueError
    - mismatched event_names raises ValueError

  CAPAModel — forward / cif
    - forward() shape (batch, num_events, time_bins)
    - forward() no nan/inf
    - cif() values in [0,1], monotone, row-sum ≤ 1
    - full gradient chain: input embeddings → loss.backward()
    - forward_from_dict() matches forward() for identical inputs
    - CauseSpecificHazardHead variant produces valid CIF

  CAPAModel — get_attention_weights
    - None before first forward
    - returns AttentionWeights after forward
    - shapes match n_loci
    - refreshed on second forward

  CAPAModel — predict()
    - returns correct keys for all events
    - CIF is list[float] of length time_bins
    - risk_score in [0,1]
    - CIF is monotone non-decreasing
    - CIF sum ≤ 1 at each time step across events
    - attention_weights returned and correct shape
    - works with pre-computed embedding dicts (no ESM-2 or cache needed)
    - zero-fallback when allele not in provided dict (logs warning)
    - clinical dict with all fields
    - clinical dict with missing fields
    - set_inference_components: cache hit is used
    - predict() does not require grad (no_grad context)
    - different inputs → different CIF curves

  End-to-end training step
    - forward → deephit_loss → backward → optimizer step
    - parameters update after optimizer step
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from capa.model.capa_model import (
    CONDITIONING_CATEGORIES,
    DISEASE_CATEGORIES,
    DONOR_TYPE_CATEGORIES,
    STEM_CELL_SOURCE_CATEGORIES,
    CAPAModel,
    ClinicalEncoder,
)
from capa.model.interaction import AttentionWeights
from capa.model.losses import deephit_loss

# ---------------------------------------------------------------------------
# Shared constants (small dims for fast CPU tests)
# ---------------------------------------------------------------------------
EMB = 32
N_LOCI = 3
LOCI = ["A", "B", "C"]
CLIN_DIM = 16
INT_DIM = 24
N_EVENTS = 3
T_BINS = 10
N_HEADS = 4
N_LAYERS = 1
BATCH = 4
EVENT_NAMES = ["gvhd", "relapse", "trm"]

# Dummy allele names for each locus
DONOR_HLA = {"A": "A*02:01", "B": "B*07:02", "C": "C*07:02"}
RECIP_HLA  = {"A": "A*01:01", "B": "B*08:01", "C": "C*07:01"}

# Pre-computed dummy embeddings for each allele in use
def _dummy_emb_dict() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    alleles = list(DONOR_HLA.values()) + list(RECIP_HLA.values())
    return {a: rng.standard_normal(EMB).astype(np.float32) for a in alleles}


def _make_model(survival_type: str = "deephit") -> CAPAModel:
    return CAPAModel(
        embedding_dim=EMB,
        loci=LOCI,
        clinical_dim=CLIN_DIM,
        interaction_dim=INT_DIM,
        survival_type=survival_type,
        num_events=N_EVENTS,
        time_bins=T_BINS,
        event_names=EVENT_NAMES,
        num_heads=N_HEADS,
        num_layers=N_LAYERS,
        dropout=0.0,
    )


def _rand_emb(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, N_LOCI, EMB)


def _rand_clin(batch: int = BATCH) -> torch.Tensor:
    """Already-encoded clinical features of shape (batch, CLIN_DIM)."""
    return torch.randn(batch, CLIN_DIM)


# ===========================================================================
# ClinicalEncoder
# ===========================================================================

class TestClinicalEncoderPrepareInputs:
    def test_all_missing_returns_zeros(self) -> None:
        cont, cats = ClinicalEncoder.prepare_inputs(None)
        assert cont.shape == (1, ClinicalEncoder._N_CONT)
        assert cats.shape == (1, 4)
        assert (cont == 0).all()
        assert (cats == 0).all()

    def test_empty_dict_returns_zeros(self) -> None:
        cont, cats = ClinicalEncoder.prepare_inputs({})
        assert (cont == 0).all()
        assert (cats == 0).all()

    def test_continuous_fields_scaled(self) -> None:
        cont, _ = ClinicalEncoder.prepare_inputs({"age_recipient": 50.0})
        assert abs(cont[0, 0].item() - 0.5) < 1e-5  # 50/100

    def test_age_donor_scaled(self) -> None:
        cont, _ = ClinicalEncoder.prepare_inputs({"age_donor": 30.0})
        assert abs(cont[0, 1].item() - 0.3) < 1e-5

    def test_cd34_dose_scaled(self) -> None:
        cont, _ = ClinicalEncoder.prepare_inputs({"cd34_dose": 5.0})
        assert abs(cont[0, 2].item() - 0.5) < 1e-5

    def test_sex_mismatch_binary(self) -> None:
        cont_yes, _ = ClinicalEncoder.prepare_inputs({"sex_mismatch": True})
        cont_no,  _ = ClinicalEncoder.prepare_inputs({"sex_mismatch": False})
        assert cont_yes[0, 3].item() == pytest.approx(1.0)
        assert cont_no[0, 3].item()  == pytest.approx(0.0)

    def test_known_disease_category(self) -> None:
        _, cats = ClinicalEncoder.prepare_inputs({"disease": "ALL"})
        expected = DISEASE_CATEGORIES.index("ALL")
        assert cats[0, 0].item() == expected

    def test_unknown_disease_maps_to_zero(self) -> None:
        _, cats = ClinicalEncoder.prepare_inputs({"disease": "XYZZY"})
        assert cats[0, 0].item() == 0

    def test_known_conditioning(self) -> None:
        _, cats = ClinicalEncoder.prepare_inputs({"conditioning": "MAC"})
        assert cats[0, 1].item() == CONDITIONING_CATEGORIES.index("MAC")

    def test_known_donor_type(self) -> None:
        _, cats = ClinicalEncoder.prepare_inputs({"donor_type": "MUD"})
        assert cats[0, 2].item() == DONOR_TYPE_CATEGORIES.index("MUD")

    def test_known_stem_cell_source(self) -> None:
        _, cats = ClinicalEncoder.prepare_inputs({"stem_cell_source": "PBSC"})
        assert cats[0, 3].item() == STEM_CELL_SOURCE_CATEGORIES.index("PBSC")

    def test_none_value_maps_to_zero(self) -> None:
        _, cats = ClinicalEncoder.prepare_inputs({"disease": None})
        assert cats[0, 0].item() == 0

    def test_device_respected(self) -> None:
        cont, cats = ClinicalEncoder.prepare_inputs({}, device="cpu")
        assert cont.device.type == "cpu"
        assert cats.device.type == "cpu"


class TestClinicalEncoderForward:
    @pytest.fixture()
    def enc(self) -> ClinicalEncoder:
        return ClinicalEncoder(output_dim=CLIN_DIM, cat_embed_dim=4, dropout=0.0)

    def _inputs(self, batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
        cont = torch.zeros(batch, ClinicalEncoder._N_CONT)
        cats = torch.zeros(batch, 4, dtype=torch.long)
        return cont, cats

    def test_output_dim_property(self, enc: ClinicalEncoder) -> None:
        assert enc.output_dim == CLIN_DIM

    def test_shape(self, enc: ClinicalEncoder) -> None:
        cont, cats = self._inputs()
        assert enc(cont, cats).shape == (BATCH, CLIN_DIM)

    def test_dtype_float32(self, enc: ClinicalEncoder) -> None:
        cont, cats = self._inputs()
        assert enc(cont, cats).dtype == torch.float32

    def test_no_nan(self, enc: ClinicalEncoder) -> None:
        cont, cats = self._inputs()
        assert not torch.isnan(enc(cont, cats)).any()

    def test_batch_one(self, enc: ClinicalEncoder) -> None:
        cont, cats = self._inputs(1)
        assert enc(cont, cats).shape == (1, CLIN_DIM)

    def test_grad_flows_to_cont(self, enc: ClinicalEncoder) -> None:
        cont = torch.randn(BATCH, ClinicalEncoder._N_CONT, requires_grad=True)
        cats = torch.zeros(BATCH, 4, dtype=torch.long)
        # Use .norm() instead of .sum(): sum(LayerNorm(x)) == 0 identically,
        # giving zero gradient; norm() does not have this degeneracy.
        enc(cont, cats).norm().backward()
        assert cont.grad is not None and not torch.all(cont.grad == 0)

    def test_grad_flows_to_mlp_weights(self, enc: ClinicalEncoder) -> None:
        cont, cats = self._inputs()
        cont.requires_grad_(True)
        enc(cont, cats).sum().backward()
        for name, p in enc.mlp.named_parameters():
            assert p.grad is not None, f"mlp.{name} has no grad"


# ===========================================================================
# CAPAModel — construction
# ===========================================================================

class TestCAPAModelConstruction:
    def test_default_construction(self) -> None:
        m = _make_model()
        assert isinstance(m, CAPAModel)

    def test_cause_specific_variant(self) -> None:
        m = _make_model("cause_specific")
        assert m._survival_type == "cause_specific"

    def test_custom_loci(self) -> None:
        m = CAPAModel(embedding_dim=EMB, loci=["A", "DRB1"], num_heads=2)
        assert m._loci == ["A", "DRB1"]
        assert m._n_loci == 2

    def test_n_loci_backward_compat(self) -> None:
        """n_loci kwarg is still accepted for legacy code."""
        m = CAPAModel(embedding_dim=EMB, n_loci=N_LOCI, num_heads=4)
        assert m._n_loci == N_LOCI

    def test_invalid_survival_type_raises(self) -> None:
        with pytest.raises(ValueError, match="survival_type"):
            CAPAModel(embedding_dim=EMB, survival_type="foo")

    def test_mismatched_event_names_raises(self) -> None:
        with pytest.raises(ValueError, match="event_names"):
            CAPAModel(embedding_dim=EMB, num_events=3, event_names=["a", "b"])

    def test_has_interaction_submodule(self) -> None:
        from capa.model.interaction import CrossAttentionInteraction
        m = _make_model()
        assert isinstance(m.interaction, CrossAttentionInteraction)

    def test_has_clinical_encoder_submodule(self) -> None:
        m = _make_model()
        assert isinstance(m.clinical_encoder, ClinicalEncoder)

    def test_is_nn_module(self) -> None:
        assert isinstance(_make_model(), torch.nn.Module)


# ===========================================================================
# CAPAModel — forward / cif
# ===========================================================================

class TestCAPAModelForward:
    def test_shape(self) -> None:
        m = _make_model()
        out = m(_rand_emb(), _rand_emb(), _rand_clin())
        assert out.shape == (BATCH, N_EVENTS, T_BINS)

    def test_dtype_float32(self) -> None:
        out = _make_model()(_rand_emb(), _rand_emb(), _rand_clin())
        assert out.dtype == torch.float32

    def test_no_nan(self) -> None:
        out = _make_model()(_rand_emb(), _rand_emb(), _rand_clin())
        assert not torch.isnan(out).any()

    def test_no_inf(self) -> None:
        out = _make_model()(_rand_emb(), _rand_emb(), _rand_clin())
        assert not torch.isinf(out).any()

    def test_cause_specific_shape(self) -> None:
        m = _make_model("cause_specific")
        out = m(_rand_emb(), _rand_emb(), _rand_clin())
        assert out.shape == (BATCH, N_EVENTS, T_BINS)

    def test_batch_one(self) -> None:
        m = _make_model()
        out = m(_rand_emb(1), _rand_emb(1), _rand_clin(1))
        assert out.shape == (1, N_EVENTS, T_BINS)

    def test_forward_from_dict_shape(self) -> None:
        m = _make_model()
        out = m.forward_from_dict(_rand_emb(), _rand_emb(), {"age_recipient": 10.0})
        assert out.shape == (BATCH, N_EVENTS, T_BINS)

    def test_forward_from_dict_no_nan(self) -> None:
        m = _make_model()
        out = m.forward_from_dict(_rand_emb(), _rand_emb())
        assert not torch.isnan(out).any()


class TestCAPAModelCIF:
    def test_cif_shape(self) -> None:
        m = _make_model()
        c = m.cif(_rand_emb(), _rand_emb(), _rand_clin())
        assert c.shape == (BATCH, N_EVENTS, T_BINS)

    def test_cif_in_01(self) -> None:
        m = _make_model()
        c = m.cif(_rand_emb(), _rand_emb(), _rand_clin())
        assert (c >= 0).all() and (c <= 1 + 1e-5).all()

    def test_cif_monotone(self) -> None:
        m = _make_model()
        c = m.cif(_rand_emb(), _rand_emb(), _rand_clin())
        diffs = c[:, :, 1:] - c[:, :, :-1]
        assert (diffs >= -1e-6).all()

    def test_cif_row_sum_le_one(self) -> None:
        m = _make_model()
        c = m.cif(_rand_emb(), _rand_emb(), _rand_clin())
        assert (c.sum(dim=1) <= 1.0 + 1e-5).all()

    def test_cif_cause_specific_valid(self) -> None:
        m = _make_model("cause_specific")
        c = m.cif(_rand_emb(), _rand_emb(), _rand_clin())
        assert (c >= 0).all() and (c <= 1 + 1e-5).all()
        diffs = c[:, :, 1:] - c[:, :, :-1]
        assert (diffs >= -1e-6).all()


class TestCAPAModelGradients:
    def test_grad_to_clinical_features(self) -> None:
        m = _make_model()
        clin = _rand_clin().requires_grad_(True)
        m(_rand_emb(), _rand_emb(), clin).sum().backward()
        assert clin.grad is not None and not torch.all(clin.grad == 0)

    def test_grad_flows_through_interaction(self) -> None:
        m = _make_model()
        d = _rand_emb().requires_grad_(True)
        r = _rand_emb().requires_grad_(True)
        m(d, r, _rand_clin()).sum().backward()
        assert d.grad is not None and not torch.all(d.grad == 0)
        assert r.grad is not None and not torch.all(r.grad == 0)

    def test_all_params_receive_grad(self) -> None:
        """All params get grad when forward_from_dict routes through ClinicalEncoder."""
        m = _make_model()
        clinical = {"disease": "ALL", "conditioning": "MAC",
                    "donor_type": "MUD", "stem_cell_source": "PBSC",
                    "age_recipient": 10.0, "sex_mismatch": True}
        m.forward_from_dict(_rand_emb(), _rand_emb(), clinical).sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"{name} has no grad"

    def test_grad_with_deephit_loss(self) -> None:
        """All params get grad through deephit_loss when using forward_from_dict."""
        m = _make_model()
        clinical = {"disease": "AML", "conditioning": "RIC",
                    "donor_type": "MSD", "stem_cell_source": "BM",
                    "age_recipient": 5.0}
        logits = m.forward_from_dict(_rand_emb(), _rand_emb(), clinical)
        t = torch.randint(0, T_BINS, (BATCH,))
        k = torch.randint(0, N_EVENTS + 1, (BATCH,))
        loss = deephit_loss(logits, t, k)
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"{name} has no grad after loss.backward()"


# ===========================================================================
# CAPAModel — attention weights
# ===========================================================================

class TestCAPAModelAttentionWeights:
    def test_none_before_forward(self) -> None:
        m = _make_model()
        assert m.get_attention_weights() is None

    def test_not_none_after_forward(self) -> None:
        m = _make_model()
        m(_rand_emb(), _rand_emb(), _rand_clin())
        assert m.get_attention_weights() is not None

    def test_returns_attention_weights_type(self) -> None:
        m = _make_model()
        m(_rand_emb(), _rand_emb(), _rand_clin())
        assert isinstance(m.get_attention_weights(), AttentionWeights)

    def test_d2r_shape(self) -> None:
        m = _make_model()
        m(_rand_emb(), _rand_emb(), _rand_clin())
        w = m.get_attention_weights()
        assert w is not None
        for layer_w in w.donor_to_recip:
            assert layer_w.shape == (BATCH, N_LOCI, N_LOCI)

    def test_r2d_shape(self) -> None:
        m = _make_model()
        m(_rand_emb(), _rand_emb(), _rand_clin())
        w = m.get_attention_weights()
        assert w is not None
        for layer_w in w.recip_to_donor:
            assert layer_w.shape == (BATCH, N_LOCI, N_LOCI)

    def test_num_layers_matches(self) -> None:
        m = _make_model()
        m(_rand_emb(), _rand_emb(), _rand_clin())
        w = m.get_attention_weights()
        assert w is not None
        assert len(w.donor_to_recip) == N_LAYERS
        assert len(w.recip_to_donor) == N_LAYERS

    def test_weights_refreshed_on_second_forward(self) -> None:
        m = _make_model()
        m(_rand_emb(), _rand_emb(), _rand_clin())
        w1 = m.get_attention_weights()
        m(_rand_emb(), _rand_emb(), _rand_clin())
        w2 = m.get_attention_weights()
        assert w1 is not w2


# ===========================================================================
# CAPAModel — predict()
# ===========================================================================

class TestCAPAModelPredict:
    @pytest.fixture()
    def model(self) -> CAPAModel:
        return _make_model()

    @pytest.fixture()
    def emb_dict(self) -> dict[str, np.ndarray]:
        return _dummy_emb_dict()

    def test_returns_event_keys(self, model: CAPAModel, emb_dict: dict) -> None:
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            donor_embeddings=emb_dict,
            recipient_embeddings=emb_dict,
        )
        for name in EVENT_NAMES:
            assert name in result

    def test_cif_is_list_of_floats(self, model: CAPAModel, emb_dict: dict) -> None:
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        for name in EVENT_NAMES:
            cif = result[name]["cif"]
            assert isinstance(cif, list)
            assert len(cif) == T_BINS
            assert all(isinstance(v, float) for v in cif)

    def test_risk_score_in_01(self, model: CAPAModel, emb_dict: dict) -> None:
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        for name in EVENT_NAMES:
            rs = result[name]["risk_score"]
            assert 0.0 <= rs <= 1.0

    def test_cif_monotone(self, model: CAPAModel, emb_dict: dict) -> None:
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        for name in EVENT_NAMES:
            cif = result[name]["cif"]
            for i in range(len(cif) - 1):
                assert cif[i + 1] >= cif[i] - 1e-6, f"{name} CIF is not monotone at t={i}"

    def test_cif_sum_le_one(self, model: CAPAModel, emb_dict: dict) -> None:
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        for t in range(T_BINS):
            total = sum(result[name]["cif"][t] for name in EVENT_NAMES)
            assert total <= 1.0 + 1e-5, f"CIF sum > 1 at t={t}: {total}"

    def test_attention_weights_returned(self, model: CAPAModel, emb_dict: dict) -> None:
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        assert "attention_weights" in result
        aw = result["attention_weights"]
        assert aw is not None
        assert "donor_to_recip" in aw
        assert "recip_to_donor" in aw

    def test_attention_weights_shape(self, model: CAPAModel, emb_dict: dict) -> None:
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        aw = result["attention_weights"]
        assert aw is not None
        d2r = aw["donor_to_recip"]
        assert len(d2r) == N_LOCI
        assert all(len(row) == N_LOCI for row in d2r)

    def test_zero_fallback_for_missing_allele(
        self, model: CAPAModel, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An allele not in the embedding dict triggers a zero-vector fallback."""
        with caplog.at_level(logging.WARNING, logger="capa.model.capa_model"):
            result = model.predict(
                DONOR_HLA, RECIP_HLA,
                donor_embeddings={},  # intentionally empty
                recipient_embeddings=_dummy_emb_dict(),
            )
        assert any("zeros" in msg for msg in caplog.messages)
        # Output should still be valid
        for name in EVENT_NAMES:
            cif = result[name]["cif"]
            assert all(0.0 <= v <= 1.0 + 1e-5 for v in cif)

    def test_clinical_dict_full(self, model: CAPAModel, emb_dict: dict) -> None:
        clinical = {
            "age_recipient": 8.0,
            "age_donor": 32.0,
            "cd34_dose": 6.5,
            "sex_mismatch": True,
            "disease": "ALL",
            "conditioning": "MAC",
            "donor_type": "MUD",
            "stem_cell_source": "PBSC",
        }
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            clinical=clinical,
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        for name in EVENT_NAMES:
            assert 0.0 <= result[name]["risk_score"] <= 1.0

    def test_clinical_dict_empty(self, model: CAPAModel, emb_dict: dict) -> None:
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            clinical={},
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        for name in EVENT_NAMES:
            assert 0.0 <= result[name]["risk_score"] <= 1.0

    def test_different_inputs_different_outputs(
        self, model: CAPAModel, emb_dict: dict
    ) -> None:
        rng = np.random.default_rng(99)
        other_embs = {a: rng.standard_normal(EMB).astype(np.float32)
                      for a in list(DONOR_HLA.values()) + list(RECIP_HLA.values())}
        r1 = model.predict(DONOR_HLA, RECIP_HLA,
                           donor_embeddings=emb_dict, recipient_embeddings=emb_dict)
        r2 = model.predict(DONOR_HLA, RECIP_HLA,
                           donor_embeddings=other_embs, recipient_embeddings=other_embs)
        # CIF curves should differ
        assert r1["gvhd"]["cif"] != r2["gvhd"]["cif"]

    def test_predict_uses_cache(self, model: CAPAModel, emb_dict: dict) -> None:
        """If a cache is attached and contains an allele, it is used."""
        mock_cache = MagicMock()
        mock_cache.contains.return_value = True
        mock_cache.get.return_value = np.zeros(EMB, dtype=np.float32)
        model.set_inference_components(cache=mock_cache)

        result = model.predict(DONOR_HLA, RECIP_HLA)  # no provided embeddings
        assert mock_cache.contains.called
        assert mock_cache.get.called
        for name in EVENT_NAMES:
            assert 0.0 <= result[name]["risk_score"] <= 1.0

    def test_predict_no_grad(self, model: CAPAModel, emb_dict: dict) -> None:
        """predict() should not accumulate gradients."""
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        for name in EVENT_NAMES:
            # result values are Python floats — no grad tracking
            assert isinstance(result[name]["risk_score"], float)

    def test_cause_specific_predict(self, emb_dict: dict) -> None:
        model = _make_model("cause_specific")
        result = model.predict(
            DONOR_HLA, RECIP_HLA,
            donor_embeddings=emb_dict, recipient_embeddings=emb_dict,
        )
        for name in EVENT_NAMES:
            cif = result[name]["cif"]
            assert len(cif) == T_BINS
            assert all(0.0 <= v <= 1.0 + 1e-5 for v in cif)


# ===========================================================================
# End-to-end training step
# ===========================================================================

class TestEndToEndTraining:
    def test_training_step(self) -> None:
        """A full forward → loss → backward → optimizer step runs without error."""
        model = _make_model()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        d = _rand_emb()
        r = _rand_emb()
        clin = _rand_clin()
        t = torch.randint(0, T_BINS, (BATCH,))
        k = torch.randint(0, N_EVENTS + 1, (BATCH,))

        model.train()
        optim.zero_grad()
        logits = model(d, r, clin)
        loss = deephit_loss(logits, t, k)
        loss.backward()
        optim.step()

        assert torch.isfinite(loss)

    def test_params_update_after_step(self) -> None:
        model = _make_model()
        optim = torch.optim.SGD(model.parameters(), lr=1.0)

        # Snapshot parameters before
        before = {n: p.clone() for n, p in model.named_parameters()}

        d, r, clin = _rand_emb(), _rand_emb(), _rand_clin()
        t = torch.randint(0, T_BINS, (BATCH,))
        k = torch.randint(0, N_EVENTS + 1, (BATCH,))

        model.train()
        optim.zero_grad()
        deephit_loss(model(d, r, clin), t, k).backward()
        optim.step()

        changed = sum(
            1 for n, p in model.named_parameters()
            if not torch.allclose(p, before[n])
        )
        assert changed > 0, "No parameters changed after optimizer step"

    def test_cause_specific_training_step(self) -> None:
        from capa.model.losses import cause_specific_loss
        model = _make_model("cause_specific")
        clinical = {"disease": "CML", "conditioning": "NMA",
                    "donor_type": "haplo", "stem_cell_source": "cord",
                    "age_recipient": 12.0}
        d, r = _rand_emb(), _rand_emb()
        t = torch.randint(0, T_BINS, (BATCH,))
        k = torch.randint(0, N_EVENTS + 1, (BATCH,))

        model.train()
        hazards = model.forward_from_dict(d, r, clinical)
        loss = cause_specific_loss(hazards, t, k)
        loss.backward()
        assert torch.isfinite(loss)
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
