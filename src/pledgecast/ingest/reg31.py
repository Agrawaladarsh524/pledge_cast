"""Regulation 31 pledge events - PLAN.md sec.1.1, sec.2.4.

**A feature layer, never the state layer.** sec.2.4 is explicit:

    "Reg 31 backfill | Reconstruct pre-2021 state -> feature layer only |
     Per-promoter events with a different denominator; reconstruction is
     error-prone"

The events reach further back than the XBRL archive (ADANIPOWER to 2017), which
makes them tempting for backfilling pledge levels before 2021-Q3. They are not
suitable for that: each row is one promoter's action, and its percentage is of
total equity rather than of promoter holding, so summing them does not
reconstruct the panel's ``pledge_pct_promoter``. They are used as an overlay on
the trajectory chart (sec.12) and nothing else.

Field mapping, read off the live payload on 2026-08-13::

    sr_dateof_creation     23-SEP-2025      -> event_date
    promoterName           S. B. Adani ...  -> promoter_name
    typeOfEvent            creation         -> event_type
    numofShares            137500000        -> shares
    perofShares            0.71             -> pct_equity
    nameOfLenderDebenture                   -> lender
    reasonForEncumbrance                    -> reason
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy.engine import Connection
from tqdm import tqdm

from pledgecast.db import repository as repo
from pledgecast.exceptions import DataIngestionError
from pledgecast.ingest.shareholding import parse_nse_date
from pledgecast.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from config import Settings

logger = get_logger(__name__)

# NSE writes these in several spellings; normalise to the sec.6 vocabulary.
_EVENT_TYPE_MAP = {
    "creation": "creation",
    "created": "creation",
    "invocation": "invocation",
    "invoked": "invocation",
    "release": "release",
    "released": "release",
    "revocation": "release",
    "revoked": "release",
}


def normalise_event_type(value: Any) -> str | None:
    """Map a raw ``typeOfEvent`` onto creation | release | invocation."""
    if not value:
        return None
    text = str(value).strip().lower()
    if text in _EVENT_TYPE_MAP:
        return _EVENT_TYPE_MAP[text]
    for key, mapped in _EVENT_TYPE_MAP.items():
        if key in text:
            return mapped
    logger.debug("unmapped Reg 31 event type %r", value)
    return text or None


def _number(value: Any) -> float | None:
    if value in (None, "", "NULL", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value in (None, "", "NULL", "-"):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def fetch_events(session, symbol: str, issuer: str, settings: Settings) -> list[dict]:
    """Reg 31 disclosures for one company -> ``pledge_events``-shaped rows."""
    payload = session.get_json(
        settings.ingest.endpoints["reg31"],
        params={"type": "reg31", "index": "equities", "symbol": symbol, "issuer": issuer},
    )
    records = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []

    rows: list[dict] = []
    for record in records:
        event_date = parse_nse_date(record.get("sr_dateof_creation")) or parse_nse_date(
            record.get("broadcastDateTime")
        )
        if not event_date:
            continue
        rows.append(
            {
                "symbol": symbol,
                "event_date": event_date,
                "promoter_name": _text(record.get("promoterName")),
                "event_type": normalise_event_type(record.get("typeOfEvent")),
                "shares": _number(record.get("numofShares")),
                "pct_equity": _number(record.get("perofShares")),
                "lender": _text(record.get("nameOfLenderDebenture")),
                "reason": _text(record.get("reasonForEncumbrance")),
            }
        )
    return rows


def ingest_events(
    session,
    conn: Connection,
    companies: list[dict],
    settings: Settings,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Fetch and store Reg 31 events for every company."""
    stats: dict[str, Any] = {"companies": len(companies), "events": 0, "failures": [], "types": {}}

    bar = tqdm(companies, desc="fetching Reg 31", unit="co", disable=not show_progress)
    for company in bar:
        symbol, issuer = company["symbol"], company["company_name"]
        try:
            rows = fetch_events(session, symbol, issuer, settings)
        except DataIngestionError as exc:
            stats["failures"].append((symbol, str(exc)[:160]))
            continue

        if not rows:
            continue

        # Dedupe within the payload, then replace the symbol's events wholesale.
        # The UNIQUE constraint cannot be relied on: its columns are nullable and
        # SQLite treats NULL as distinct from NULL, so ON CONFLICT would miss any
        # event with no promoter name and re-runs would accumulate duplicates.
        seen: set[tuple] = set()
        unique = []
        for row in rows:
            key = (row["event_date"], row["promoter_name"], row["event_type"], row["shares"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
            stats["types"][row["event_type"]] = stats["types"].get(row["event_type"], 0) + 1

        repo.replace_pledge_events(conn, symbol, unique)
        stats["events"] += len(unique)

    logger.info(
        "reg31: %d events across %d companies, %d failures, types=%s",
        stats["events"],
        stats["companies"],
        len(stats["failures"]),
        stats["types"],
    )
    return stats


__all__ = ["fetch_events", "ingest_events", "normalise_event_type"]
