"""Comprehensive tests for capa.data.hla_parser.

Coverage:
- WHO allele parsing (field counts, gene normalisation, edge cases)
- Serological antigen parsing (all loci, broad antigens, case)
- Unified parse_hla_string dispatcher
- HLAAllele properties (two_field, resolution, common_allele, __str__)
- HLAProfile construction (single, two-allele, slash-separated, mixed notation)
- parse_uci_hla_columns (all 93+41+ combinations observed in real data)
- Error cases (malformed, unknown antigens)
- Consistency checks (tables match, no orphans)
"""

from __future__ import annotations

import pytest

from capa.data.hla_parser import (
    ANTIGEN_TO_COMMON_ALLELE,
    ANTIGEN_TO_FIELD1,
    CANONICAL_GENE,
    MISMATCH_TYPE_LABELS,
    HLAAllele,
    HLAMismatchSummary,
    HLAProfile,
    STANDARD_LOCI,
    normalize_gene,
    parse_hla_string,
    parse_hla_typing,
    parse_serological_allele,
    parse_uci_hla_columns,
    parse_who_allele,
)


# ===========================================================================
# WHO allele parsing — parse_who_allele
# ===========================================================================


class TestParseWhoAllele:
    # --- Happy path ---

    def test_two_field(self) -> None:
        a = parse_who_allele("A*02:01")
        assert a.gene == "A"
        assert a.field1 == "02"
        assert a.field2 == "01"
        assert a.field3 is None
        assert a.field4 is None

    def test_three_field(self) -> None:
        a = parse_who_allele("DRB1*15:01:01")
        assert a.gene == "DRB1"
        assert a.field1 == "15"
        assert a.field2 == "01"
        assert a.field3 == "01"
        assert a.field4 is None

    def test_four_field(self) -> None:
        a = parse_who_allele("DRB1*15:01:01:02")
        assert a.field4 == "02"

    def test_three_digit_field1(self) -> None:
        # Some alleles have three-digit first fields (e.g. B*135:01)
        a = parse_who_allele("B*135:01")
        assert a.field1 == "135"

    def test_hla_prefix_stripped(self) -> None:
        a = parse_who_allele("HLA-A*02:01")
        assert a.gene == "A"
        assert a.field1 == "02"

    def test_whitespace_stripped(self) -> None:
        a = parse_who_allele("  B*07:02  ")
        assert a.gene == "B"
        assert a.field1 == "07"

    def test_gene_normalised_c(self) -> None:
        # HLA-C stored as "C" in the parser (no normalisation needed for C directly)
        a = parse_who_allele("C*07:01")
        assert a.gene == "C"

    def test_gene_normalised_drb1(self) -> None:
        a = parse_who_allele("DRB1*03:01")
        assert a.gene == "DRB1"

    def test_gene_normalised_dqb1(self) -> None:
        a = parse_who_allele("DQB1*06:02")
        assert a.gene == "DQB1"

    def test_source_notation_is_who(self) -> None:
        assert parse_who_allele("A*02:01").source_notation == "who"

    def test_resolution_is_allele(self) -> None:
        assert parse_who_allele("A*02:01").resolution == "allele"

    def test_common_allele_is_none_for_who(self) -> None:
        # WHO-notation alleles are already fully resolved; no fallback needed
        assert parse_who_allele("A*02:01").common_allele is None

    # --- two_field property ---

    def test_two_field_from_two_field(self) -> None:
        assert parse_who_allele("A*02:01").two_field == "A*02:01"

    def test_two_field_from_four_field(self) -> None:
        assert parse_who_allele("DRB1*15:01:01:02").two_field == "DRB1*15:01"

    # --- __str__ ---

    def test_str_two_field(self) -> None:
        a = parse_who_allele("B*07:02")
        assert str(a) == "B*07:02"

    def test_str_four_field(self) -> None:
        a = parse_who_allele("DRB1*15:01:01:02")
        assert str(a) == "DRB1*15:01:01:02"

    # --- Error cases ---

    def test_missing_asterisk_raises(self) -> None:
        with pytest.raises(ValueError, match="WHO allele"):
            parse_who_allele("A0201")

    def test_serological_string_raises(self) -> None:
        with pytest.raises(ValueError, match="WHO allele"):
            parse_who_allele("A2")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_who_allele("")

    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_who_allele("not_an_allele")

    def test_single_field_only_is_valid(self) -> None:
        # "A*02" = first-field-only resolution; used in low-resolution typing — must succeed
        a = parse_who_allele("A*02")
        assert a.field1 == "02"
        assert a.field2 is None


# ===========================================================================
# Serological allele parsing — parse_serological_allele
# ===========================================================================


class TestParseSerologicalAllele:
    # --- A locus ---

    def test_a2(self) -> None:
        a = parse_serological_allele("A2")
        assert a.gene == "A"
        assert a.field1 == "02"
        assert a.field2 is None
        assert a.source_notation == "serological"

    def test_a11(self) -> None:
        a = parse_serological_allele("A11")
        assert a.field1 == "11"

    def test_a24(self) -> None:
        a = parse_serological_allele("A24")
        assert a.gene == "A"
        assert a.field1 == "24"

    def test_a_broad_antigen_a9(self) -> None:
        a = parse_serological_allele("A9")
        assert a.gene == "A"
        assert a.field1 == "09"

    # --- B locus ---

    def test_b7(self) -> None:
        a = parse_serological_allele("B7")
        assert a.gene == "B"
        assert a.field1 == "07"

    def test_b27(self) -> None:
        a = parse_serological_allele("B27")
        assert a.gene == "B"
        assert a.field1 == "27"

    def test_b44(self) -> None:
        a = parse_serological_allele("B44")
        assert a.field1 == "44"

    def test_b_split_b60(self) -> None:
        # B60 = B*40:01; maps to field1="40"
        a = parse_serological_allele("B60")
        assert a.gene == "B"
        assert a.field1 == "40"

    def test_b_split_b64(self) -> None:
        # B64 = B*14:01; maps to field1="14"
        a = parse_serological_allele("B64")
        assert a.field1 == "14"

    # --- C locus (Cw prefix) ---

    def test_cw7(self) -> None:
        a = parse_serological_allele("Cw7")
        assert a.gene == "C"
        assert a.field1 == "07"

    def test_cw7_lowercase(self) -> None:
        a = parse_serological_allele("cw7")
        assert a.gene == "C"

    def test_cw1(self) -> None:
        a = parse_serological_allele("Cw1")
        assert a.field1 == "01"

    def test_cw9_maps_to_03(self) -> None:
        # Cw9 ≈ C*03:03 in modern nomenclature
        a = parse_serological_allele("Cw9")
        assert a.gene == "C"
        assert a.field1 == "03"

    # --- DRB1 locus (DR prefix) ---

    def test_dr15(self) -> None:
        a = parse_serological_allele("DR15")
        assert a.gene == "DRB1"
        assert a.field1 == "15"

    def test_dr1(self) -> None:
        a = parse_serological_allele("DR1")
        assert a.gene == "DRB1"
        assert a.field1 == "01"

    def test_dr_broad_dr2(self) -> None:
        # DR2 (broad) → field1 "15" (DR15, most common split)
        a = parse_serological_allele("DR2")
        assert a.gene == "DRB1"
        assert a.field1 == "15"

    def test_dr_broad_dr3(self) -> None:
        a = parse_serological_allele("DR3")
        assert a.gene == "DRB1"
        assert a.field1 == "03"

    def test_dr17_split_of_dr3(self) -> None:
        a = parse_serological_allele("DR17")
        assert a.gene == "DRB1"
        assert a.field1 == "03"

    # --- DQB1 locus (DQ prefix) ---

    def test_dq5(self) -> None:
        a = parse_serological_allele("DQ5")
        assert a.gene == "DQB1"
        assert a.field1 == "05"

    def test_dq2(self) -> None:
        a = parse_serological_allele("DQ2")
        assert a.gene == "DQB1"
        assert a.field1 == "02"

    def test_dq_broad_dq1(self) -> None:
        # DQ1 (broad) → field1 "05" (DQ5 most common split)
        a = parse_serological_allele("DQ1")
        assert a.gene == "DQB1"
        assert a.field1 == "05"

    # --- DPB1 locus (DP prefix) ---

    def test_dp4(self) -> None:
        a = parse_serological_allele("DP4")
        assert a.gene == "DPB1"
        assert a.field1 == "04"

    # --- Properties for serological alleles ---

    def test_resolution_is_antigen(self) -> None:
        assert parse_serological_allele("A2").resolution == "antigen"

    def test_two_field_no_second_field(self) -> None:
        a = parse_serological_allele("A2")
        assert a.two_field == "A*02"

    def test_common_allele_a2(self) -> None:
        a = parse_serological_allele("A2")
        assert a.common_allele == "A*02:01"

    def test_common_allele_dr15(self) -> None:
        a = parse_serological_allele("DR15")
        assert a.common_allele == "DRB1*15:01"

    def test_common_allele_b27(self) -> None:
        a = parse_serological_allele("B27")
        assert a.common_allele == "B*27:05"

    def test_common_allele_cw7(self) -> None:
        a = parse_serological_allele("Cw7")
        assert a.common_allele == "C*07:01"

    def test_hla_prefix_accepted(self) -> None:
        a = parse_serological_allele("HLA-A2")
        assert a.gene == "A"

    # --- Error cases ---

    def test_who_notation_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_serological_allele("A*02:01")

    def test_unknown_antigen_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised"):
            parse_serological_allele("Z99")

    def test_no_number_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_serological_allele("A")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_serological_allele("")


# ===========================================================================
# Unified dispatcher — parse_hla_string
# ===========================================================================


class TestParseHLAString:
    def test_dispatches_who(self) -> None:
        a = parse_hla_string("A*02:01")
        assert a.source_notation == "who"
        assert a.field2 == "01"

    def test_dispatches_serological(self) -> None:
        a = parse_hla_string("A2")
        assert a.source_notation == "serological"

    def test_dispatches_dr_serological(self) -> None:
        a = parse_hla_string("DR15")
        assert a.gene == "DRB1"

    def test_hla_prefix_who(self) -> None:
        a = parse_hla_string("HLA-B*07:02")
        assert a.gene == "B"
        assert a.source_notation == "who"

    def test_hla_prefix_serological(self) -> None:
        a = parse_hla_string("HLA-B7")
        assert a.gene == "B"
        assert a.source_notation == "serological"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_hla_string("NOTANALLELE")

    def test_garbled_with_asterisk_raises(self) -> None:
        # Field1 must be at least 2 digits — single digit should fail
        with pytest.raises(ValueError):
            parse_hla_string("A*9")

    def test_whitespace_handled(self) -> None:
        a = parse_hla_string("  A*02:01  ")
        assert a.field1 == "02"


# ===========================================================================
# normalize_gene
# ===========================================================================


class TestNormalizeGene:
    def test_cw_to_c(self) -> None:
        assert normalize_gene("Cw") == "C"

    def test_dr_to_drb1(self) -> None:
        assert normalize_gene("DR") == "DRB1"

    def test_dq_to_dqb1(self) -> None:
        assert normalize_gene("DQ") == "DQB1"

    def test_a_stays_a(self) -> None:
        assert normalize_gene("A") == "A"

    def test_drb1_stays_drb1(self) -> None:
        assert normalize_gene("DRB1") == "DRB1"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown HLA gene"):
            normalize_gene("X")


# ===========================================================================
# HLAProfile construction — parse_hla_typing
# ===========================================================================


class TestParseHLATyping:
    def test_single_allele_per_locus(self) -> None:
        profile = parse_hla_typing({"A": "A*02:01", "DRB1": "DRB1*15:01"})
        assert "A" in profile.alleles
        assert len(profile.alleles["A"]) == 1
        assert profile.alleles["A"][0].field1 == "02"

    def test_two_alleles_as_list(self) -> None:
        profile = parse_hla_typing({"A": ["A*02:01", "A*24:02"]})
        assert len(profile.alleles["A"]) == 2
        assert profile.alleles["A"][1].field1 == "24"

    def test_two_alleles_slash_separated(self) -> None:
        profile = parse_hla_typing({"A": "A*02:01/A*24:02"})
        assert len(profile.alleles["A"]) == 2

    def test_mixed_notation_per_locus(self) -> None:
        profile = parse_hla_typing({"A": ["A*02:01", "A24"]})
        assert profile.alleles["A"][0].source_notation == "who"
        assert profile.alleles["A"][1].source_notation == "serological"

    def test_serological_value(self) -> None:
        profile = parse_hla_typing({"A": "A2", "B": "B7", "DR": "DR15"})
        assert profile.alleles["A"][0].source_notation == "serological"
        assert profile.alleles["B"][0].field1 == "07"
        # DR → normalised to DRB1
        assert "DRB1" in profile.alleles

    def test_locus_key_normalised(self) -> None:
        profile = parse_hla_typing({"Cw": "Cw7", "DR": "DR15", "DQ": "DQ5"})
        assert "C" in profile.alleles
        assert "DRB1" in profile.alleles
        assert "DQB1" in profile.alleles

    def test_role_stored(self) -> None:
        profile = parse_hla_typing({"A": "A*02:01"}, role="donor")
        assert profile.role == "donor"

    def test_empty_dict(self) -> None:
        profile = parse_hla_typing({})
        assert profile.typed_loci == []

    def test_typed_loci(self) -> None:
        profile = parse_hla_typing({"A": "A*02:01", "B": "B*07:02"})
        assert set(profile.typed_loci) == {"A", "B"}

    def test_first_allele(self) -> None:
        profile = parse_hla_typing({"A": ["A*02:01", "A*24:02"]})
        first = profile.first_allele("A")
        assert first is not None
        assert first.field1 == "02"

    def test_first_allele_missing_locus(self) -> None:
        profile = parse_hla_typing({"A": "A*02:01"})
        assert profile.first_allele("B") is None

    def test_get_locus_with_raw_key(self) -> None:
        profile = parse_hla_typing({"DR": "DR15"})
        alleles = profile.get_locus("DRB1")  # canonical key
        assert len(alleles) == 1

    def test_standard_loci_coverage(self) -> None:
        profile = parse_hla_typing({"A": "A*02:01", "B": "B*07:02"})
        cov = profile.standard_loci_coverage()
        assert cov["A"] is True
        assert cov["B"] is True
        assert cov["C"] is False
        assert cov["DRB1"] is False

    def test_full_five_loci(self) -> None:
        profile = parse_hla_typing(
            {
                "A": "A*02:01",
                "B": "B*07:02",
                "C": "C*07:01",
                "DRB1": "DRB1*15:01",
                "DQB1": "DQB1*06:02",
            },
            role="donor",
        )
        for locus in STANDARD_LOCI:
            assert profile.first_allele(locus) is not None


# ===========================================================================
# UCI HLA summary — parse_uci_hla_columns
# ===========================================================================


class TestParseUCIHLAColumns:
    def _row(
        self,
        match_score: int = 0,
        mismatched: int = 0,
        antigen: int = -1,
        allele: int = -1,
        hla_gri: int = 0,
    ) -> dict[str, object]:
        """Build a minimal cleaned-loader row for testing."""
        return {
            "hla_match_score": match_score,
            "hla_mismatched": mismatched,
            "n_antigen_mismatches": antigen,
            "n_allele_mismatches": allele,
            "hla_mismatch_type": hla_gri,
        }

    # --- Fully matched patients ---

    def test_fully_matched(self) -> None:
        s = parse_uci_hla_columns(self._row(0, 0, -1, -1, 0))
        assert s.match_grade == "10/10"
        assert s.is_fully_matched
        assert not s.mismatched
        assert s.n_antigen_mismatches == 0
        assert s.n_allele_mismatches == 0
        assert s.total_mismatches == 0
        assert s.mismatch_type == "matched"

    # --- 9/10 match (one mismatch) ---

    def test_9_of_10_antigen_only(self) -> None:
        # HLAmatch=1, Antigen=1→2 diffs, Alel=0→1 diff, HLAgrI=1
        # In UCI: Antigen=1 means 2 diffs, but for 9/10 we'd see Antigen=1, Alel=0
        # Most common in data: Antigen=1(=2), Alel=0(=1), HLAgrI=1
        s = parse_uci_hla_columns(self._row(1, 0, 1, 0, 1))
        assert s.match_grade == "9/10"
        assert s.total_mismatches == 1
        assert s.mismatch_type == "antigen_diff_only"
        assert s.n_antigen_mismatches == 2
        assert s.n_allele_mismatches == 1

    def test_9_of_10_allele_only(self) -> None:
        s = parse_uci_hla_columns(self._row(1, 0, 0, 1, 2))
        assert s.match_grade == "9/10"
        assert s.mismatch_type == "allele_diff_only"

    def test_9_of_10_drb1_diff(self) -> None:
        s = parse_uci_hla_columns(self._row(1, 0, 0, 1, 3))
        assert s.mismatch_type == "DRB1_diff_only"

    # --- 8/10 match ---

    def test_8_of_10_two_same_type(self) -> None:
        s = parse_uci_hla_columns(self._row(2, 1, 1, 1, 4))
        assert s.match_grade == "8/10"
        assert s.mismatched
        assert s.mismatch_type == "two_diffs_same_type"

    def test_8_of_10_two_mixed_type(self) -> None:
        s = parse_uci_hla_columns(self._row(2, 1, 1, 1, 5))
        assert s.mismatch_type == "two_diffs_mixed_type"

    # --- 7/10 match (complex / HLAgrI=7) ---

    def test_7_of_10_complex(self) -> None:
        s = parse_uci_hla_columns(self._row(3, 1, 1, 2, 7))
        assert s.match_grade == "7/10"
        assert s.mismatch_type == "complex"
        assert s.mismatch_type_code == 7

    # --- Sentinel decoding ---

    def test_sentinel_minus1_decodes_to_zero(self) -> None:
        s = parse_uci_hla_columns(self._row(antigen=-1, allele=-1))
        assert s.n_antigen_mismatches == 0
        assert s.n_allele_mismatches == 0

    def test_antigen_0_decodes_to_1(self) -> None:
        s = parse_uci_hla_columns(self._row(antigen=0))
        assert s.n_antigen_mismatches == 1

    def test_allele_3_decodes_to_4(self) -> None:
        s = parse_uci_hla_columns(self._row(allele=3))
        assert s.n_allele_mismatches == 4

    # --- is_fully_matched property ---

    def test_is_fully_matched_true(self) -> None:
        assert parse_uci_hla_columns(self._row(0, 0, -1, -1, 0)).is_fully_matched

    def test_is_fully_matched_false(self) -> None:
        assert not parse_uci_hla_columns(self._row(1, 0, 1, 0, 1)).is_fully_matched

    # --- Integration: all real data combinations from UCI ---

    @pytest.mark.parametrize(
        "match_score,mismatched,antigen,allele,hla_gri,expected_grade,expected_type",
        [
            (0, 0, -1, -1, 0, "10/10", "matched"),            # 93 patients
            (1, 0, 0, 1, 2, "9/10", "allele_diff_only"),      # 14 patients
            (1, 0, 1, 0, 1, "9/10", "antigen_diff_only"),     #  1 patient
            (1, 0, 0, 1, 3, "9/10", "DRB1_diff_only"),        #  3 patients
            (1, 0, 0, 2, 4, "9/10", "two_diffs_same_type"),   #  1 patient
            (1, 0, 0, 2, 5, "9/10", "two_diffs_mixed_type"),  #  1 patient
            (1, 0, 0, 3, 7, "9/10", "complex"),               #  1 patient
            (2, 1, 1, 0, 1, "8/10", "antigen_diff_only"),     # 41 patients
            (2, 1, 1, 0, 3, "8/10", "DRB1_diff_only"),        #  6 patients
            (2, 1, 1, 1, 4, "8/10", "two_diffs_same_type"),   # 11 patients
            (2, 1, 1, 1, 5, "8/10", "two_diffs_mixed_type"),  #  3 patients
            (2, 1, 1, 2, 7, "8/10", "complex"),               #  4 patients
            (3, 1, 2, 0, 4, "7/10", "two_diffs_same_type"),   #  7 patients
        ],
    )
    def test_real_data_combinations(
        self,
        match_score: int,
        mismatched: int,
        antigen: int,
        allele: int,
        hla_gri: int,
        expected_grade: str,
        expected_type: str,
    ) -> None:
        s = parse_uci_hla_columns(
            {
                "hla_match_score": match_score,
                "hla_mismatched": mismatched,
                "n_antigen_mismatches": antigen,
                "n_allele_mismatches": allele,
                "hla_mismatch_type": hla_gri,
            }
        )
        assert s.match_grade == expected_grade
        assert s.mismatch_type == expected_type


# ===========================================================================
# HLAAllele frozen / equality
# ===========================================================================


class TestHLAAlleleFrozen:
    def test_equal_alleles(self) -> None:
        a1 = parse_who_allele("A*02:01")
        a2 = parse_who_allele("A*02:01")
        assert a1 == a2

    def test_different_fields_not_equal(self) -> None:
        a1 = parse_who_allele("A*02:01")
        a2 = parse_who_allele("A*02:06")
        assert a1 != a2

    def test_hashable(self) -> None:
        a = parse_who_allele("A*02:01")
        s = {a}
        assert a in s

    def test_immutable(self) -> None:
        a = parse_who_allele("A*02:01")
        with pytest.raises((AttributeError, TypeError)):
            a.gene = "B"  # type: ignore[misc]


# ===========================================================================
# Table consistency checks
# ===========================================================================


class TestTableConsistency:
    def test_every_antigen_to_common_allele_is_parseable(self) -> None:
        """Every value in ANTIGEN_TO_COMMON_ALLELE must be a valid WHO string."""
        for antigen, allele_str in ANTIGEN_TO_COMMON_ALLELE.items():
            try:
                parsed = parse_who_allele(allele_str)
                assert parsed.field2 is not None, (
                    f"Common allele for {antigen!r} should have two fields: {allele_str!r}"
                )
            except ValueError as exc:
                pytest.fail(f"ANTIGEN_TO_COMMON_ALLELE[{antigen!r}] = {allele_str!r} is invalid: {exc}")

    def test_every_antigen_in_common_allele_is_in_field1(self) -> None:
        """Every key in ANTIGEN_TO_COMMON_ALLELE should also appear in ANTIGEN_TO_FIELD1."""
        missing = set(ANTIGEN_TO_COMMON_ALLELE) - set(ANTIGEN_TO_FIELD1)
        assert not missing, f"Keys in ANTIGEN_TO_COMMON_ALLELE but not ANTIGEN_TO_FIELD1: {missing}"

    def test_canonical_gene_covers_common_prefixes(self) -> None:
        required = {"A", "B", "C", "Cw", "DR", "DRB1", "DQ", "DQB1", "DP", "DPB1"}
        missing = required - set(CANONICAL_GENE)
        assert not missing, f"Missing from CANONICAL_GENE: {missing}"

    def test_mismatch_type_labels_cover_all_known_codes(self) -> None:
        # Codes observed in real UCI data: 0,1,2,3,4,5,7
        known_codes = {0, 1, 2, 3, 4, 5, 7}
        assert known_codes.issubset(set(MISMATCH_TYPE_LABELS))

    def test_standard_loci_are_canonical(self) -> None:
        for locus in STANDARD_LOCI:
            assert locus in CANONICAL_GENE.values(), f"{locus} not in CANONICAL_GENE values"

    def test_antigen_to_field1_fields_are_2_or_3_digits(self) -> None:
        for key, val in ANTIGEN_TO_FIELD1.items():
            # Allow special serological markers like "Bw4" which are not numeric
            if val.isdigit():
                assert len(val) in (2, 3), f"ANTIGEN_TO_FIELD1[{key!r}] = {val!r} is not 2-3 digits"


# ===========================================================================
# Edge cases and unusual inputs
# ===========================================================================


class TestEdgeCases:
    def test_parse_hla_string_four_field_allele(self) -> None:
        a = parse_hla_string("DRB1*15:01:01:02")
        assert a.field4 == "02"

    def test_parse_hla_typing_ignores_empty_strings_in_list(self) -> None:
        profile = parse_hla_typing({"A": ["A*02:01", ""]})
        assert len(profile.alleles["A"]) == 1

    def test_parse_hla_typing_slash_with_spaces(self) -> None:
        profile = parse_hla_typing({"A": "A*02:01 / A*24:02"})
        assert len(profile.alleles["A"]) == 2

    def test_serological_allele_str(self) -> None:
        a = parse_serological_allele("A2")
        # str() uses WHO-style with asterisk, first field only
        assert str(a) == "A*02"

    def test_parse_uci_hla_columns_with_pandas_series(self) -> None:
        import pandas as pd

        row = pd.Series({
            "hla_match_score": 0,
            "hla_mismatched": 0,
            "n_antigen_mismatches": -1,
            "n_allele_mismatches": -1,
            "hla_mismatch_type": 0,
        })
        s = parse_uci_hla_columns(row)
        assert s.is_fully_matched

    def test_who_allele_first_field_one_digit_rejected(self) -> None:
        # "A*2:01" should fail — first field must be 2+ digits
        with pytest.raises(ValueError):
            parse_who_allele("A*2:01")

    def test_serological_all_a_locus_antigens(self) -> None:
        """Smoke-test: every A-locus antigen in the table parses without error."""
        a_antigens = [k for k in ANTIGEN_TO_FIELD1 if k.startswith("A")]
        for ag in a_antigens:
            a = parse_serological_allele(ag)
            assert a.gene == "A"

    def test_serological_all_b_locus_antigens(self) -> None:
        b_antigens = [k for k in ANTIGEN_TO_FIELD1 if k.startswith("B")]
        for ag in b_antigens:
            a = parse_serological_allele(ag)
            assert a.gene == "B"

    def test_serological_all_dr_antigens(self) -> None:
        dr_antigens = [k for k in ANTIGEN_TO_FIELD1 if k.startswith("DR")]
        for ag in dr_antigens:
            a = parse_serological_allele(ag)
            assert a.gene == "DRB1"

    def test_serological_all_dq_antigens(self) -> None:
        dq_antigens = [k for k in ANTIGEN_TO_FIELD1 if k.startswith("DQ")]
        for ag in dq_antigens:
            a = parse_serological_allele(ag)
            assert a.gene == "DQB1"

    def test_serological_all_cw_antigens(self) -> None:
        cw_antigens = [k for k in ANTIGEN_TO_FIELD1 if k.startswith("Cw")]
        for ag in cw_antigens:
            a = parse_serological_allele(ag)
            assert a.gene == "C"
