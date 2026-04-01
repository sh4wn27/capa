"""Tests for capa.data.hla_parser."""

from __future__ import annotations

import pytest

from capa.data.hla_parser import HLAAllele, parse_allele, parse_hla_typing


class TestParseAllele:
    def test_two_field(self) -> None:
        allele = parse_allele("A*02:01")
        assert allele.gene == "A"
        assert allele.field1 == "02"
        assert allele.field2 == "01"
        assert allele.field3 is None

    def test_four_field(self) -> None:
        allele = parse_allele("DRB1*15:01:01:02")
        assert allele.gene == "DRB1"
        assert allele.field1 == "15"
        assert allele.field2 == "01"
        assert allele.field3 == "01"
        assert allele.field4 == "02"

    def test_two_field_str(self) -> None:
        allele = parse_allele("B*07:02")
        assert str(allele) == "B*07:02"

    def test_two_field_property(self) -> None:
        allele = parse_allele("DRB1*15:01:01")
        assert allele.two_field == "DRB1*15:01"

    def test_whitespace_stripped(self) -> None:
        allele = parse_allele("  A*02:01  ")
        assert allele.gene == "A"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_allele("not_an_allele")

    def test_missing_star_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_allele("A0201")


class TestParseHLATyping:
    def test_basic(self) -> None:
        typing = parse_hla_typing({"A": "A*02:01", "DRB1": "DRB1*15:01"})
        assert isinstance(typing["A"], HLAAllele)
        assert typing["A"].gene == "A"
        assert typing["DRB1"].field1 == "15"

    def test_empty(self) -> None:
        assert parse_hla_typing({}) == {}
