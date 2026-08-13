"""Shareholding-pattern filings - PLAN.md sec.1.2, sec.5.2, sec.10.

Three jobs, deliberately separate:

  ``discover_filings``  master API -> the ``filings`` ledger (JSON only, cheap)
  ``select_universe``   completeness filter -> ~300 of the 500 constituents
  ``download_pending``  concurrent, resumable XBRL download -> disk + ledger

Discovery is split from download because sec.1.3 wants the universe "filtered
for data completeness". Knowing which companies have a full filing history costs
one cheap JSON call each; downloading 500 companies' XBRL to find out would cost
an extra ~4,000 files.

**Field mapping**, read off the live payload on 2026-08-13::

    date            30-JUN-2026            -> quarter_end
    submissionDate  13-JUL-2026            -> submission_date   (the PIT anchor)
    xbrl            https://nsearchives... -> xbrl_url
    revisedData     'N' | 'Revised'        -> revision detection

Raw XML lands on disk and only a ledger row goes in SQLite (sec.5.2): what was
downloaded, when, its hash, and its parse status.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy.engine import Connection
from tqdm import tqdm

from pledgecast.db import repository as repo
from pledgecast.exceptions import DataIngestionError
from pledgecast.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from config import Settings

    from pledgecast.ingest.nse_session import NSESession

logger = get_logger(__name__)

_DATE_FORMATS = ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d-%m-%Y")


def parse_nse_date(value: Any) -> str | None:
    """``'30-JUN-2026'`` -> ``'2026-06-30'``. Tolerates a trailing time part."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.split(" ")[0]  # drop 'HH:MM:SS' where present
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    logger.debug("unparseable NSE date %r", value)
    return None


# --------------------------------------------------------------------------- #
# 1. discovery                                                                #
# --------------------------------------------------------------------------- #
def fetch_filing_list(
    session: NSESession,
    symbol: str,
    issuer: str,
    settings: Settings,
) -> list[dict]:
    """Master API -> ledger-shaped rows for one company, filtered to the window."""
    payload = session.get_json(
        settings.ingest.endpoints["shareholding_master"],
        params={
            "index": "equities",
            "from_date": settings.window.api_from_date,
            "to_date": settings.window.api_to_date,
            "symbol": symbol,
            "issuer": issuer,
        },
    )
    records = payload if isinstance(payload, list) else payload.get("data", [])
    if not isinstance(records, list):
        return []

    rows: list[dict] = []
    for record in records:
        quarter_end = parse_nse_date(record.get("date"))
        submission_date = parse_nse_date(record.get("submissionDate"))
        url = (record.get("xbrl") or "").strip()

        if not quarter_end or not url:
            continue
        # sec.9.3 needs a real submission date; without one the row cannot be
        # placed in time, so it is dropped rather than guessed at.
        if not submission_date:
            logger.debug("%s %s: no submissionDate, skipping", symbol, quarter_end)
            continue
        if not (settings.window.first_quarter_end <= quarter_end <= settings.window.last_quarter_end):
            continue

        rows.append(
            {
                "symbol": symbol,
                "quarter_end": quarter_end,
                "submission_date": submission_date,
                "xbrl_url": url,
                "status": "pending",
            }
        )
    return rows


def discover_filings(
    session: NSESession,
    conn: Connection,
    companies: list[dict],
    settings: Settings,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Populate the ``filings`` ledger for every company. JSON only, no downloads."""
    total_rows = 0
    failures: list[tuple[str, str]] = []
    per_symbol: dict[str, int] = {}

    iterator = tqdm(companies, desc="discovering filings", unit="co", disable=not show_progress)
    for company in iterator:
        symbol, issuer = company["symbol"], company["company_name"]
        try:
            rows = fetch_filing_list(session, symbol, issuer, settings)
        except DataIngestionError as exc:
            failures.append((symbol, str(exc)[:160]))
            per_symbol[symbol] = 0
            continue

        if rows:
            repo.upsert_filings(conn, rows)
        per_symbol[symbol] = len(rows)
        total_rows += len(rows)

    if failures:
        logger.warning("filing discovery failed for %d companies", len(failures))
        for symbol, reason in failures[:10]:
            logger.warning("  %s: %s", symbol, reason)

    return {
        "companies_queried": len(companies),
        "filings_found": total_rows,
        "failures": failures,
        "per_symbol": per_symbol,
    }


# --------------------------------------------------------------------------- #
# 2. completeness filter                                                      #
# --------------------------------------------------------------------------- #
def select_universe(conn: Connection, settings: Settings) -> dict[str, Any]:
    """Narrow the constituent list to the study universe (sec.1.3).

    Excluded companies are flagged ``in_universe = 0`` rather than deleted, so
    the audit trail of what was considered survives (sec.6).
    """
    filings = repo.load_filings(conn)
    if filings.empty:
        raise DataIngestionError("no filings discovered - cannot select a universe")

    counts = (
        filings.groupby("symbol")["quarter_end"].nunique().sort_values(ascending=False)
    )
    eligible = counts[counts >= settings.universe.min_filings_required]
    kept = list(eligible.head(settings.universe.target_size).index)

    all_symbols = repo.get_universe_symbols(conn, in_universe=False)
    dropped = [s for s in all_symbols if s not in set(kept)]

    repo.set_in_universe(conn, dropped, False)
    repo.set_in_universe(conn, kept, True)

    logger.info(
        "universe: kept %d of %d (>= %d quarters); dropped %d",
        len(kept), len(all_symbols), settings.universe.min_filings_required, len(dropped),
    )
    return {
        "kept": kept,
        "dropped": dropped,
        "quarter_counts": counts.to_dict(),
        "median_quarters_kept": float(counts.loc[kept].median()) if kept else 0.0,
    }


# --------------------------------------------------------------------------- #
# 3. download                                                                 #
# --------------------------------------------------------------------------- #
def local_path_for(symbol: str, xbrl_url: str, settings: Settings):
    """``data/raw/xbrl/<SYMBOL>/<original-filename>``.

    The URL basename is already unique per filing (it embeds the record id and
    a submission timestamp), so revisions never collide with originals.
    """
    basename = xbrl_url.rstrip("/").split("/")[-1] or "unnamed"
    if not basename.lower().endswith(".xml"):
        basename = f"{basename}.xml"
    return settings.paths.raw_xbrl_dir / symbol / basename


def download_pending(
    session: NSESession,
    conn: Connection,
    settings: Settings,
    *,
    symbols: list[str] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Concurrently download every ledger row not yet on disk.

    Resumable (sec.10): files already present are hashed and skipped, so a run
    interrupted at file 4,000 costs nothing to restart.
    """
    pending = repo.load_filings(conn, status="pending")
    if symbols is not None:
        pending = pending[pending["symbol"].isin(symbols)]
    if pending.empty:
        logger.info("no pending filings to download")
        return {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0, "seconds": 0.0}

    jobs = [
        (
            int(row.filing_id),
            row.symbol,
            row.xbrl_url,
            local_path_for(row.symbol, row.xbrl_url, settings),
        )
        for row in pending.itertuples()
    ]

    downloaded = skipped = failed = 0
    total_bytes = 0
    started = time.monotonic()

    def _fetch(job: tuple[int, str, str, Any]) -> tuple[int, Any]:
        filing_id, _symbol, url, dest = job
        try:
            path, digest, n_bytes, was_skipped = session.download(url, dest)
            return filing_id, (path, digest, n_bytes, was_skipped, None)
        except Exception as exc:  # noqa: BLE001 - recorded per row, never fatal
            return filing_id, (dest, None, 0, False, f"{type(exc).__name__}: {exc}")

    bar = tqdm(total=len(jobs), desc="downloading XBRL", unit="file", disable=not show_progress)
    with ThreadPoolExecutor(max_workers=settings.ingest.max_workers) as pool:
        futures = {pool.submit(_fetch, job): job for job in jobs}
        for future in as_completed(futures):
            filing_id, (path, digest, n_bytes, was_skipped, error) = future.result()

            if error:
                failed += 1
                repo.update_filing_status(conn, filing_id, "pending", error_message=error)
            else:
                total_bytes += n_bytes
                skipped += was_skipped
                downloaded += not was_skipped
                repo.update_filing_status(
                    conn, filing_id, "downloaded", local_path=str(path), sha256=digest
                )
            bar.update(1)
    bar.close()

    elapsed = time.monotonic() - started
    logger.info(
        "download: %d new, %d already on disk, %d failed, %.1f MB in %.1fs",
        downloaded, skipped, failed, total_bytes / 1e6, elapsed,
    )
    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "bytes": total_bytes,
        "seconds": elapsed,
        "seconds_per_file": elapsed / max(downloaded, 1),
    }


__all__ = [
    "discover_filings",
    "download_pending",
    "fetch_filing_list",
    "local_path_for",
    "parse_nse_date",
    "select_universe",
]
