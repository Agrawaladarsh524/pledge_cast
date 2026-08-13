"""XBRL parser tests - PLAN.md sec.15, priority ***.

    "(5) pledge-present . explicit no-pledge . missing flag -> UNAVAILABLE .
     malformed -> quarantine not crash . numeric scale correctness"

Run against COMMITTED real filings, not mocks (sec.15). The fixtures are actual
NSE downloads chosen to cover both tag sets and both percentage scales.
"""

from __future__ import annotations

import pytest

from pledgecast.exceptions import ParseError
from pledgecast.ingest import xbrl

pytestmark = [pytest.mark.critical, pytest.mark.parser]


# --------------------------------------------------------------------------- #
# 1. pledge present                                                           #
# --------------------------------------------------------------------------- #
def test_pledge_present_parses_against_known_truth(xbrl_pledge_legacy, settings):
    """JPPOWER 2025-03-31, cross-checked against the master API's own numbers."""
    record = xbrl.parse_file(
        xbrl_pledge_legacy,
        symbol="JPPOWER",
        quarter_end="2025-03-31",
        submission_date="2025-04-11",
        settings=settings,
    )

    assert record.pledge_status == xbrl.PLEDGE_PRESENT
    assert record.promoter_shares == pytest.approx(1_644_830_118)
    assert record.total_shares == pytest.approx(6_853_458_827)
    # The master API reported pr_and_prgrp = 24 for this filing.
    assert record.promoter_holding_pct == pytest.approx(24.0, abs=0.05)
    assert record.pledged_shares == pytest.approx(1_302_697_997)
    assert record.pledge_pct_promoter == pytest.approx(79.2, abs=0.05)
    assert record.pledge_pct_equity == pytest.approx(19.01, abs=0.05)
    assert record.schema_generation == "legacy"


# --------------------------------------------------------------------------- #
# 2. explicit no-pledge                                                       #
# --------------------------------------------------------------------------- #
def test_explicit_zero_is_no_pledge_not_unavailable(xbrl_no_pledge, settings):
    """A reported zero and a missing report are different facts.

    A company with no encumbrance omits the numeric detail section entirely and
    files only the boolean questions. Reading that as UNAVAILABLE would discard
    a genuine, informative zero.
    """
    record = xbrl.parse_file(
        xbrl_no_pledge,
        symbol="TCS",
        quarter_end="2026-06-30",
        submission_date="2026-07-10",
        settings=settings,
    )

    assert record.pledge_status == xbrl.NO_PLEDGE
    assert record.pledge_pct_promoter == 0.0
    assert record.pledge_pct_equity == 0.0
    assert record.promoter_holding_pct is not None and record.promoter_holding_pct > 50


# --------------------------------------------------------------------------- #
# 3. missing data -> UNAVAILABLE                                              #
# --------------------------------------------------------------------------- #
def test_absent_encumbrance_data_is_unavailable_never_zero(settings):
    """No numeric tag AND no flag means unknown - which is not the same as 0."""
    payload = b"""<?xml version="1.0"?>
    <xbrl xmlns:x="http://x" xmlns:xbrli="http://www.xbrl.org/2003/instance">
      <xbrli:context id="c1">
        <xbrli:entity><xbrli:segment>
          <xbrli:explicitMember dimension="x:CategoryOfShareholdersAxis"
            >x:ShareholdingOfPromoterAndPromoterGroupMember</xbrli:explicitMember>
        </xbrli:segment></xbrli:entity>
      </xbrli:context>
      <xbrli:context id="c2">
        <xbrli:entity><xbrli:segment>
          <xbrli:explicitMember dimension="x:CategoryOfShareholdersAxis"
            >x:ShareholdingPatternMember</xbrli:explicitMember>
        </xbrli:segment></xbrli:entity>
      </xbrli:context>
      <x:NumberOfShares contextRef="c1">500</x:NumberOfShares>
      <x:NumberOfShares contextRef="c2">1000</x:NumberOfShares>
      <x:ShareholdingAsAPercentageOfTotalNumberOfShares contextRef="c1">50</x:ShareholdingAsAPercentageOfTotalNumberOfShares>
      <x:ShareholdingAsAPercentageOfTotalNumberOfShares contextRef="c2">100</x:ShareholdingAsAPercentageOfTotalNumberOfShares>
    </xbrl>"""

    record = xbrl.parse_bytes(
        payload,
        symbol="X",
        quarter_end="2024-03-31",
        submission_date="2024-04-10",
        settings=settings,
    )

    assert record.pledge_status == xbrl.UNAVAILABLE
    assert record.pledged_shares is None, "missing must stay None, never 0.0"
    assert record.pledge_pct_promoter is None
    assert record.promoter_holding_pct == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# 4. malformed -> quarantine, not a crash                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("truncated mid-document", b"<?xml version='1.0'?><xbrli:xbrl><unclosed>"),
        ("empty file", b""),
        ("an HTML error page", b"<html><body>404 Not Found</body></html>"),
        ("valid XML, wrong schema", b"<?xml version='1.0'?><root><a>1</a></root>"),
        ("binary garbage", b"\x00\x01\x02\xff\xfe not xml at all"),
    ],
)
def test_malformed_input_raises_parse_error(label, payload, settings):
    """Every failure mode must surface as ParseError - never another type."""
    with pytest.raises(ParseError):
        xbrl.parse_bytes(
            payload,
            symbol="X",
            quarter_end="2024-03-31",
            submission_date="2024-04-10",
            settings=settings,
        )


def test_quarantine_copies_and_leaves_the_original(xbrl_pledge_legacy, settings, tmp_path):
    """sec.5.2: data/raw/ is immutable research data. Quarantine copies, never moves."""
    import shutil

    source = tmp_path / "bad.xml"
    shutil.copy2(xbrl_pledge_legacy, source)

    destination = xbrl.quarantine(source, "test reason", settings)

    assert destination is not None and destination.exists()
    reason_file = destination.with_suffix(destination.suffix + ".reason.txt")
    assert reason_file.exists()
    assert "test reason" in reason_file.read_text(encoding="utf-8")
    assert source.exists(), "the original must never be moved or deleted"

    destination.unlink(missing_ok=True)
    reason_file.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 5. numeric scale correctness                                                #
# --------------------------------------------------------------------------- #
def test_fraction_scale_filings_are_normalised(xbrl_pledge_fraction_scale, settings):
    """From SHP V1.0 percentages are filed as fractions - 0.24, not 24.00.

    Nothing in the markup announces it (unitRef="pure", decimals="INF" in both
    conventions). Left uncorrected this is a silent 100x error in the project's
    primary feature across the most recent quarters.
    """
    record = xbrl.parse_file(
        xbrl_pledge_fraction_scale,
        symbol="JPPOWER",
        quarter_end="2026-06-30",
        submission_date="2026-07-13",
        settings=settings,
    )

    assert record.schema_generation == "modern"
    # Same company, same holding as the legacy-era fixture: 24%, not 0.24%.
    assert record.promoter_holding_pct == pytest.approx(24.0, abs=0.05)
    assert record.pledge_pct_promoter == pytest.approx(79.2, abs=0.5)
    assert 1.0 < record.promoter_holding_pct <= 100.0


def test_scale_detector_reads_the_convention_from_the_total(settings):
    """The detector keys off a fact that is 100% by definition, not off a date."""
    from lxml import etree

    def build(total_pct: str) -> bytes:
        return f"""<?xml version="1.0"?>
        <xbrl xmlns:x="http://x" xmlns:xbrli="http://www.xbrl.org/2003/instance">
          <xbrli:context id="c2"><xbrli:entity><xbrli:segment>
            <xbrli:explicitMember dimension="x:CategoryOfShareholdersAxis"
              >x:ShareholdingPatternMember</xbrli:explicitMember>
          </xbrli:segment></xbrli:entity></xbrli:context>
          <x:ShareholdingAsAPercentageOfTotalNumberOfShares contextRef="c2">{total_pct}</x:ShareholdingAsAPercentageOfTotalNumberOfShares>
        </xbrl>""".encode()

    for total, expected in (("100", 1.0), ("1.0", 100.0), ("100.00", 1.0)):
        root = etree.fromstring(build(total), parser=etree.XMLParser(recover=True))
        contexts = xbrl._build_context_map(root)
        assert xbrl.detect_percentage_scale(root, contexts) == expected


def test_percentages_never_escape_their_valid_range(xbrl_pledge_legacy, settings):
    """sec.10 range validation: pledge_pct in [0, 100]."""
    record = xbrl.parse_file(
        xbrl_pledge_legacy,
        symbol="JPPOWER",
        quarter_end="2025-03-31",
        submission_date="2025-04-11",
        settings=settings,
    )
    for value in (
        record.promoter_holding_pct,
        record.pledge_pct_promoter,
        record.pledge_pct_equity,
    ):
        assert value is not None
        assert settings.validation.pledge_pct_min <= value <= settings.validation.pledge_pct_max

    assert record.pledged_shares <= record.promoter_shares
    assert record.promoter_shares <= record.total_shares
