"""Build the company universe - PLAN.md sec.1.3, sec.2.4.

sec.2.4 is explicit that the universe must NOT come from the pledge endpoint:

    "Universe source | Pledge list -> NIFTY 500 spine | The pledge endpoint has
     no zero-pledge control group"

That matters because a universe drawn from ``/api/corporate-pledgedata`` would
contain only companies that reported a pledge, leaving nothing to compare them
against. The model would be asked "which pledged company falls?" instead of
"does pledging predict falling?".

sec.1.2 lists no constituent endpoint, so the source used here is NSE's own
published index CSV (decision R2) - primary-source and licence-clean, which
sec.2.4 also requires by ruling out Screener and Trendlyne. It supplies exactly
the four columns ``companies`` needs, plus the full company name that the
shareholding master API requires as its ``issuer`` parameter.
"""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

from pledgecast.exceptions import DataIngestionError
from pledgecast.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from config import Settings

    from pledgecast.ingest.nse_session import NSESession

logger = get_logger(__name__)


def fetch_constituents(session: NSESession, settings: Settings | None = None) -> list[dict]:
    """Fetch the index constituent list -> ``companies``-shaped rows.

    Tries the published CSV first, then the live index API as a fallback.
    """
    if settings is None:
        from config import get_settings

        settings = get_settings()

    rows = _from_csv(session, settings)
    if not rows:
        logger.warning("constituent CSV unavailable, falling back to the index API")
        rows = _from_api(session, settings)

    if not rows:
        raise DataIngestionError(
            f"could not build the {settings.universe.index_name} universe from either "
            f"{settings.universe.constituents_csv_url} or "
            f"{settings.universe.fallback_api_path}"
        )

    logger.info("fetched %d %s constituents", len(rows), settings.universe.index_name)
    return rows


def _from_csv(session: NSESession, settings: Settings) -> list[dict]:
    """Columns: Company Name, Industry, Symbol, Series, ISIN Code."""
    try:
        response = session.get(settings.universe.constituents_csv_url, referer="/")
    except DataIngestionError as exc:
        logger.warning("constituent CSV fetch failed: %s", exc)
        return []

    text = response.text
    if "Symbol" not in text[:400]:
        logger.warning("constituent CSV did not look like a CSV: %r", text[:120])
        return []

    rows: list[dict] = []
    for record in csv.DictReader(io.StringIO(text)):
        symbol = (record.get("Symbol") or "").strip()
        name = (record.get("Company Name") or "").strip()
        if not symbol or not name:
            continue
        rows.append(
            {
                "symbol": symbol,
                "company_name": name,
                "isin": (record.get("ISIN Code") or "").strip() or None,
                "industry": (record.get("Industry") or "").strip() or None,
            }
        )
    return rows


def _from_api(session: NSESession, settings: Settings) -> list[dict]:
    """Fallback: ``/api/equity-stockIndices?index=NIFTY 500``."""
    try:
        payload = session.get_json(
            settings.universe.fallback_api_path,
            params={"index": settings.universe.index_name},
            referer="/market-data/live-equity-market",
        )
    except DataIngestionError as exc:
        logger.warning("index API fallback failed: %s", exc)
        return []

    rows: list[dict] = []
    for record in payload.get("data", []):
        symbol = (record.get("symbol") or "").strip()
        if not symbol or symbol == settings.universe.index_name:
            continue
        meta = record.get("meta") or {}
        name = (meta.get("companyName") or "").strip()
        if not name:
            continue
        rows.append(
            {
                "symbol": symbol,
                "company_name": name,
                "isin": (meta.get("isin") or "").strip() or None,
                "industry": (record.get("industry") or "").strip() or None,
            }
        )
    return rows


def write_universe_csv(rows: list[dict], settings: Settings | None = None) -> int:
    """Write ``data/universe.csv`` - COMMITTED to git (sec.8.1).

    "Makes the whole run reproducible from a clean clone." Sorted by symbol so
    the file has a stable diff between runs.
    """
    if settings is None:
        from config import get_settings

        settings = get_settings()

    path = settings.paths.universe_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["symbol", "company_name", "isin", "industry"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["symbol"]):
            writer.writerow({k: row.get(k) for k in fields})

    logger.info("wrote %s (%d rows)", path, len(rows))
    return len(rows)


def read_universe_csv(settings: Settings | None = None) -> list[dict]:
    """Read the committed universe - lets a clean clone skip the network."""
    if settings is None:
        from config import get_settings

        settings = get_settings()

    path = settings.paths.universe_csv
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


__all__ = ["fetch_constituents", "read_universe_csv", "write_universe_csv"]
