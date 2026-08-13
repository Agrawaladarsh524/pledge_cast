"""SHP XBRL parser - PLAN.md sec.7.1 ("XML -> typed rows, quarantine").

PLAN.md sec.16 says to port a working prototype. That prototype does not exist
(``d:\\pledge_cast`` is empty), so this parser was written from scratch against
real filings. The structure below was read off 100 downloaded files on
2026-08-13 and is documented here because nothing else records it.

FILE STRUCTURE
--------------
Every fact is dimensioned on a single axis, ``CategoryOfShareholdersAxis``, and
two members carry everything this project needs - both present in **all** schema
generations::

    ShareholdingOfPromoterAndPromoterGroupMember   the promoter aggregate
    ShareholdingPatternMember                      the grand total (all holders)

THREE TAXONOMY ERAS (two tag sets, two percentage scales)
---------------------------------------------------------
NSE changed the taxonomy twice inside this 20-quarter window, and the two
changes do not coincide::

    era 1  2021-09 .. 2025-03   legacy tags, percentages as PERCENTAGES (24.00)
    era 2  2025-06              modern tags, percentages as PERCENTAGES (24.00)
    era 3  2025-09 ..           modern tags, percentages as FRACTIONS   (0.24)

Tag set is detected by which tags are present, never by the ``<!--SHP V1.1-->``
comment, because era-2 files carry the new tags with no version comment at all::

    LEGACY                               MODERN
    PledgedOrEncumberedNumberOfShares    NumberOfSharesEncumbered
    PledgedOrEncumberedSharesHeld...     EncumberedSharesHeldAsPercentage...

The scale change is invisible in the markup - ``unitRef="pure"`` and
``decimals="INF"`` in both - so it is detected arithmetically per file by
``detect_percentage_scale``. See that function for why this matters.

**Why total encumbrance, not the pledge-only tag.** The modern taxonomy splits
encumbrance into pledge and non-disposal undertaking
(``NumberOfSharesEncumberedUnderPledged`` / ``...UnderNonDisposalUndertaking``);
the legacy one lumps both together as "PledgedOrEncumbered". Reading the
pledge-only tag after 2025-06 and the combined tag before it would inject an
artificial level drop halfway through the panel - a structural break the model
could learn as a time signal. So the comparable series, total encumbrance, is
used throughout. The NDU split is a documented limitation, not a bug.

THREE-WAY STATUS (sec.15 parser tests)
--------------------------------------
``pledge_status`` never collapses "reported zero" into "did not report"::

    PLEDGE_PRESENT   encumbrance tags present and > 0
    NO_PLEDGE        encumbrance tags present and == 0   (an explicit zero)
    UNAVAILABLE      tags absent entirely                (unknown, NOT zero)

A file that cannot be parsed is copied to ``data/quarantine/`` with its reason,
turning silent data loss into an auditable list (sec.8.1).
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lxml import etree

from pledgecast.exceptions import ParseError
from pledgecast.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from config import Settings

logger = get_logger(__name__)

CATEGORY_AXIS = "CategoryOfShareholdersAxis"
PROMOTER_MEMBER = "ShareholdingOfPromoterAndPromoterGroupMember"
TOTAL_MEMBER = "ShareholdingPatternMember"

# Tags shared by every generation.
TAG_SHARES = "NumberOfShares"
TAG_HOLDING_PCT = "ShareholdingAsAPercentageOfTotalNumberOfShares"

# Generation-specific encumbrance tags, most recent first.
#
# ``flags`` is a list because the modern taxonomy splits the single legacy
# question into three (pledge / NDU / other). Since the numeric tag used here is
# TOTAL encumbrance, "encumbered" means any of them is true.
ENCUMBRANCE_TAGS: dict[str, dict[str, Any]] = {
    "modern": {
        "shares": "NumberOfSharesEncumbered",
        "pct": "EncumberedSharesHeldAsPercentageOfTotalNumberOfShares",
        "flags": [
            "WhetherAnySharesHeldByPromotersAreEncumberedUnderPledgedForPromoterAndPromoterGroup",
            "WhetherAnySharesHeldByPromotersAreEncumberedUnderNonDisposalUndertakingForPromoterAndPromoterGroup",
            "WhetherAnySharesHeldByPromotersAreEncumberedOtherThanByWayOfPledgeOrNDUForPromoterAndPromoterGroup",
        ],
    },
    "legacy": {
        "shares": "PledgedOrEncumberedNumberOfShares",
        "pct": "PledgedOrEncumberedSharesHeldAsPercentageOfTotalNumberOfShares",
        "flags": [
            "WhetherAnySharesHeldByPromotersArePledgeOrOtherwiseEncumberedForPromoterAndPromoterGroup",
            "WhetherAnySharesHeldByPromotersArePledgeOrOtherwiseEncumbered",
        ],
    },
}

PLEDGE_PRESENT = "PLEDGE_PRESENT"
NO_PLEDGE = "NO_PLEDGE"
UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class PledgeRecord:
    """One parsed filing, shaped for the ``pledge_state`` table."""

    symbol: str
    quarter_end: str
    submission_date: str
    promoter_shares: float | None
    pledged_shares: float | None
    total_shares: float | None
    promoter_holding_pct: float | None
    pledge_pct_promoter: float | None
    pledge_pct_equity: float | None
    pledge_status: str
    schema_generation: str
    filing_id: int | None = None

    def to_row(self) -> dict[str, Any]:
        """Drop the parser-only fields the ``pledge_state`` schema does not have."""
        row = asdict(self)
        row.pop("schema_generation")
        return row


def _localname(element: Any) -> str:
    """Tag name without its namespace, tolerant of malformed documents.

    Deliberately avoids ``etree.QName``, which raises ``ValueError`` on a tag
    like ``xbrli:xbrl`` left behind by ``recover=True`` when the prefix was never
    declared - a truncated download produces exactly that, and a parser must
    return ParseError rather than leak an unexpected exception type. Also
    noticeably faster, which matters across ~2,000 elements x 6,000 files.
    """
    tag = element.tag
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag.rsplit(":", 1)[-1]


def _build_context_map(root: Any) -> dict[str, str | None]:
    """``contextRef`` -> its ``CategoryOfShareholdersAxis`` member (or None)."""
    mapping: dict[str, str | None] = {}
    for context in root.iter():
        if not isinstance(context.tag, str) or _localname(context) != "context":
            continue
        member: str | None = None
        for explicit in context.iter():
            if not isinstance(explicit.tag, str) or _localname(explicit) != "explicitMember":
                continue
            if (explicit.get("dimension") or "").split(":")[-1] == CATEGORY_AXIS:
                member = (explicit.text or "").strip().split(":")[-1]
        mapping[context.get("id")] = member
    return mapping


def _to_number(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _fact(root: Any, contexts: dict[str, str | None], tag: str, member: str) -> float | None:
    """First numeric value of ``tag`` reported against ``member``."""
    for element in root.iter():
        if not isinstance(element.tag, str) or _localname(element) != tag:
            continue
        if contexts.get(element.get("contextRef", "")) != member:
            continue
        value = _to_number(element.text)
        if value is not None:
            return value
    return None


def _flag(root: Any, tag: str) -> bool | None:
    """Boolean fact, or None when the tag is absent."""
    for element in root.iter():
        if isinstance(element.tag, str) and _localname(element) == tag:
            text = (element.text or "").strip().lower()
            if text in ("true", "1", "yes"):
                return True
            if text in ("false", "0", "no"):
                return False
    return None


def _has_tag(root: Any, tag: str) -> bool:
    return any(isinstance(e.tag, str) and _localname(e) == tag for e in root.iter())


def detect_generation(root: Any) -> str:
    """Which taxonomy this file uses, decided by tag presence.

    The numeric tag alone is not enough: a company with no encumbrance omits the
    whole detail section and files only the boolean questions, so the flags are
    checked too. Without this, every clean company (RELIANCE, TCS) after 2025-06
    was misread as UNAVAILABLE rather than an explicit NO_PLEDGE.
    """
    for generation, tags in ENCUMBRANCE_TAGS.items():
        if _has_tag(root, tags["shares"]):
            return generation
        if any(_has_tag(root, flag) for flag in tags["flags"]):
            return generation
    return "unknown"


def detect_percentage_scale(root: Any, contexts: dict[str, str | None]) -> float:
    """Multiplier that converts this file's percentages onto a 0-100 scale.

    NSE changed convention at the SHP V1.0 taxonomy (filings from 2025-09-30):
    percentages became decimal FRACTIONS - ``0.24`` where the same field
    previously read ``24.00``. Nothing in the file announces it; ``unitRef`` is
    ``pure`` and ``decimals`` is ``INF`` in both conventions, so it is invisible
    unless you check the arithmetic.

    Left uncorrected this is a silent 100x error in the project's primary
    feature across the most recent quarters - exactly the failure sec.15's
    "numeric scale correctness" test exists to catch.

    The detector uses a fact that is true by definition: all shareholders
    together hold 100% of the equity. Reading that total tells us which
    convention the file uses, per file, with no date hardcoded.
    """
    total = _fact(root, contexts, TAG_HOLDING_PCT, TOTAL_MEMBER)
    if total is None:
        return 1.0
    if 0.9 <= total <= 1.1:  # fractions: total reads 1.0
        return 100.0
    return 1.0  # percentages: total reads 100


def parse_bytes(
    payload: bytes,
    *,
    symbol: str,
    quarter_end: str,
    submission_date: str,
    filing_id: int | None = None,
    settings: Settings | None = None,
) -> PledgeRecord:
    """Parse one filing. Raises ``ParseError`` on anything unusable."""
    if settings is None:
        from config import get_settings

        settings = get_settings()

    if not payload:
        raise ParseError("empty file", reason="empty file")

    try:
        # recover=True: NSE occasionally emits stray entities mid-document, and
        # a whole filing should not be lost to one bad character. lxml raises
        # ValueError (not XMLSyntaxError) for some malformed inputs, so both are
        # caught - a parser must never leak an unexpected exception type.
        root = etree.fromstring(payload, parser=etree.XMLParser(recover=True, huge_tree=True))
    except (etree.XMLSyntaxError, etree.LxmlError, ValueError) as exc:
        raise ParseError(f"malformed XML: {exc}", reason=f"{type(exc).__name__}: {exc}") from exc

    if root is None:
        raise ParseError("XML parsed to nothing", reason="empty parse tree")

    contexts = _build_context_map(root)
    if PROMOTER_MEMBER not in set(contexts.values()):
        raise ParseError(
            f"no {PROMOTER_MEMBER} context found",
            reason="missing promoter aggregate context",
        )

    generation = detect_generation(root)
    pct_scale = detect_percentage_scale(root, contexts)

    promoter_shares = _fact(root, contexts, TAG_SHARES, PROMOTER_MEMBER)
    total_shares = _fact(root, contexts, TAG_SHARES, TOTAL_MEMBER)
    promoter_holding_pct = _fact(root, contexts, TAG_HOLDING_PCT, PROMOTER_MEMBER)
    if promoter_holding_pct is not None:
        promoter_holding_pct *= pct_scale

    # ---- encumbrance, three-way ------------------------------------------
    pledged_shares: float | None = None
    reported_pct_promoter: float | None = None
    pledge_pct_equity: float | None = None
    flag: bool | None = None

    if generation != "unknown":
        tags = ENCUMBRANCE_TAGS[generation]
        pledged_shares = _fact(root, contexts, tags["shares"], PROMOTER_MEMBER)
        reported_pct_promoter = _fact(root, contexts, tags["pct"], PROMOTER_MEMBER)
        pledge_pct_equity = _fact(root, contexts, tags["pct"], TOTAL_MEMBER)
        if reported_pct_promoter is not None:
            reported_pct_promoter *= pct_scale
        if pledge_pct_equity is not None:
            pledge_pct_equity *= pct_scale
        # Any of pledge / NDU / other counts, since the numeric field is total
        # encumbrance. None only when no flag is present at all.
        answers = [_flag(root, flag_tag) for flag_tag in tags["flags"]]
        given = [a for a in answers if a is not None]
        flag = any(given) if given else None

    if pledged_shares is None and flag is None:
        # Neither a number nor a flag: genuinely unknown. NOT zero (sec.15).
        status = UNAVAILABLE
    elif (pledged_shares or 0) > 0 or flag is True:
        status = PLEDGE_PRESENT
    else:
        status = NO_PLEDGE  # an explicit, reported zero

    # ---- derived percentages ---------------------------------------------
    # Prefer the FILER'S reported percentage (after scale normalisation) and
    # compute from share counts only when it is absent.
    #
    # Deriving these from NumberOfShares looks safer but is not: the denominator
    # SEBI defines excludes shares underlying depository receipts, whereas
    # NumberOfShares at the total member includes them. RELIANCE 2025-03 makes
    # the gap concrete - 6,645,496,096 / 13,532,372,898 = 49.11%, but
    # 6,645,496,096 / (13,532,372,898 - 269,806,428 DRs) = 50.11%, which is both
    # the filed figure and what the master API reports. The reported field is the
    # regulatory measure; the share ratio is only an approximation of it.
    #
    # The scale hazard that made deriving attractive is handled independently by
    # detect_percentage_scale, so nothing is given up by trusting the filer here.
    def _reported_or_ratio(
        reported: float | None, numerator: float | None, denominator: float | None, label: str
    ) -> float | None:
        computed = (
            100.0 * numerator / denominator if numerator is not None and denominator else None
        )
        if reported is None:
            return computed
        if computed is not None and abs(computed - reported) > 2.0:
            logger.debug(
                "%s %s: %s reported %.2f vs share-ratio %.2f (denominators differ)",
                symbol,
                quarter_end,
                label,
                reported,
                computed,
            )
        return reported

    promoter_holding_pct = _reported_or_ratio(
        promoter_holding_pct, promoter_shares, total_shares, "promoter_holding_pct"
    )
    pledge_pct_promoter = _reported_or_ratio(
        reported_pct_promoter, pledged_shares, promoter_shares, "pledge_pct_promoter"
    )
    pledge_pct_equity = _reported_or_ratio(
        pledge_pct_equity, pledged_shares, total_shares, "pledge_pct_equity"
    )

    if status == NO_PLEDGE:
        pledged_shares = pledged_shares if pledged_shares is not None else 0.0
        pledge_pct_promoter = pledge_pct_promoter if pledge_pct_promoter is not None else 0.0
        pledge_pct_equity = pledge_pct_equity if pledge_pct_equity is not None else 0.0

    # ---- range validation (sec.10) ---------------------------------------
    lo, hi = settings.validation.pledge_pct_min, settings.validation.pledge_pct_max
    for name, value in (
        ("promoter_holding_pct", promoter_holding_pct),
        ("pledge_pct_promoter", pledge_pct_promoter),
        ("pledge_pct_equity", pledge_pct_equity),
    ):
        if value is not None and not (lo <= value <= hi):
            raise ParseError(
                f"{name} = {value} outside [{lo}, {hi}]",
                reason=f"{name} out of range: {value}",
            )

    if promoter_shares is None and status == UNAVAILABLE:
        raise ParseError(
            "no promoter share count and no encumbrance data",
            reason="no usable facts",
        )

    return PledgeRecord(
        symbol=symbol,
        quarter_end=quarter_end,
        submission_date=submission_date,
        promoter_shares=promoter_shares,
        pledged_shares=pledged_shares,
        total_shares=total_shares,
        promoter_holding_pct=promoter_holding_pct,
        pledge_pct_promoter=pledge_pct_promoter,
        pledge_pct_equity=pledge_pct_equity,
        pledge_status=status,
        schema_generation=generation,
        filing_id=filing_id,
    )


def parse_file(
    path: Path | str,
    *,
    symbol: str,
    quarter_end: str,
    submission_date: str,
    filing_id: int | None = None,
    settings: Settings | None = None,
) -> PledgeRecord:
    """``parse_bytes`` for a file on disk."""
    path = Path(path)
    if not path.exists():
        raise ParseError(f"file not found: {path}", path=str(path), reason="missing file")
    try:
        return parse_bytes(
            path.read_bytes(),
            symbol=symbol,
            quarter_end=quarter_end,
            submission_date=submission_date,
            filing_id=filing_id,
            settings=settings,
        )
    except ParseError as exc:
        raise ParseError(str(exc), path=str(path), reason=exc.reason) from exc


def quarantine(path: Path | str, reason: str, settings: Settings | None = None) -> Path | None:
    """Copy a rejected file to ``data/quarantine/`` with a ``.reason.txt`` beside it.

    sec.8.1: "Turns silent data loss into an auditable list." The original in
    ``data/raw/`` is never moved or modified - it stays immutable research data.
    """
    if settings is None:
        from config import get_settings

        settings = get_settings()

    path = Path(path)
    if not path.exists():
        return None

    destination = settings.paths.quarantine_dir / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(path, destination)
        destination.with_suffix(destination.suffix + ".reason.txt").write_text(
            f"source: {path}\nreason: {reason}\n", encoding="utf-8"
        )
    except OSError as exc:  # pragma: no cover - disk failure, not a data problem
        logger.error("could not quarantine %s: %s", path, exc)
        return None

    logger.warning("quarantined %s: %s", path.name, reason)
    return destination


__all__ = [
    "CATEGORY_AXIS",
    "ENCUMBRANCE_TAGS",
    "NO_PLEDGE",
    "PLEDGE_PRESENT",
    "PROMOTER_MEMBER",
    "TOTAL_MEMBER",
    "UNAVAILABLE",
    "PledgeRecord",
    "detect_generation",
    "parse_bytes",
    "parse_file",
    "quarantine",
]
