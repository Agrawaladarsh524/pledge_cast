"""02 - Download and parse everything.

PLAN.md sec.16 Phase 2: "~35 min XBRL, ~5 min prices, ~5 min Reg 31."
(Measured 0.10 s/file on 2026-08-13, roughly half the sec.1.1 estimate.)

Four stages, each idempotent and independently resumable:

  1. XBRL download   concurrent, skips files already on disk
  2. XBRL parse      -> pledge_state, failures -> data/quarantine/
  3. prices          Yahoo adjclose -> prices + benchmark
  4. Reg 31          event disclosures -> pledge_events

Interrupt it at any point and re-run: downloaded files are hashed and skipped,
and every write is an upsert.

    python scripts/02_ingest_all.py
    python scripts/02_ingest_all.py --skip-xbrl --skip-reg31
    python scripts/02_ingest_all.py --symbols JPPOWER TCS
"""

from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401  - must precede pledgecast/config imports
from config import get_settings
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.ingest import prices, reg31, shareholding, xbrl
from pledgecast.ingest.nse_session import NSESession
from pledgecast.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and parse all PledgeCast data.")
    parser.add_argument("--symbols", nargs="+", default=None, help="Restrict to these symbols.")
    parser.add_argument("--skip-xbrl", action="store_true", help="Skip XBRL download + parse.")
    parser.add_argument("--skip-prices", action="store_true", help="Skip price ingestion.")
    parser.add_argument("--skip-reg31", action="store_true", help="Skip Reg 31 events.")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)
    settings.paths.ensure_exist()
    started = time.monotonic()

    results: dict[str, dict] = {}

    with NSESession(settings) as session, get_connection(settings=settings) as conn:
        companies = repo.load_companies(conn, in_universe=True)
        if companies.empty:
            logger.error("no companies in the universe - run scripts/01_build_universe.py first")
            return 1

        if args.symbols:
            wanted = {s.upper() for s in args.symbols}
            companies = companies[companies["symbol"].str.upper().isin(wanted)]

        company_rows = companies[["symbol", "company_name"]].to_dict(orient="records")
        symbols = [c["symbol"] for c in company_rows]
        logger.info("ingesting %d companies", len(symbols))

        # -- 1 + 2. XBRL -----------------------------------------------------
        if not args.skip_xbrl:
            results["download"] = shareholding.download_pending(
                session, conn, settings, symbols=symbols
            )
            results["parse"] = xbrl.parse_ledger(conn, settings)

        # -- 3. prices -------------------------------------------------------
        if not args.skip_prices:
            results["prices"] = prices.ingest_prices(conn, symbols, settings)

        # -- 4. Reg 31 -------------------------------------------------------
        if not args.skip_reg31:
            results["reg31"] = reg31.ingest_events(session, conn, company_rows, settings)

        counts = repo.table_counts(conn)
        ledger = repo.count_filings_by_status(conn)
        in_universe = len(repo.get_universe_symbols(conn))

    elapsed = time.monotonic() - started

    # ------------------------------------------------------------------ report
    print("=" * 78)
    print("PLEDGECAST - INGESTION COMPLETE")
    print("=" * 78)
    print(f"  elapsed               : {elapsed / 60:.1f} min")
    print(f"  companies in universe : {in_universe}")

    if "download" in results:
        d = results["download"]
        print(
            f"\n  XBRL download         : {d['downloaded']} new, {d['skipped']} cached, "
            f"{d['failed']} failed"
        )
        print(
            f"    volume              : {d['bytes'] / 1e6:.0f} MB in {d['seconds']:.0f}s "
            f"({d.get('seconds_per_file', 0):.3f} s/file)"
        )

    if "parse" in results:
        p = results["parse"]
        print(
            f"\n  XBRL parse            : {p['parsed']} parsed, {p['quarantined']} quarantined, "
            f"{p['superseded']} superseded revisions"
        )
        print(f"    schema generations  : {p['generations']}")
        print(f"    pledge statuses     : {p['statuses']}")
        if p["failures"]:
            print("    first failures      :")
            for sym, q, why in p["failures"][:5]:
                print(f"        {sym} {q}: {why}")

    if "prices" in results:
        pr = results["prices"]
        print(f"\n  prices                : {pr['symbols_stored']} symbols, {pr['rows']:,} rows")
        print(f"    benchmark           : {pr.get('benchmark_rows', 0):,} rows")
        if pr["too_short"]:
            print(f"    excluded, too short : {len(pr['too_short'])}")
        if pr["failures"]:
            print(f"    fetch failures      : {len(pr['failures'])}")
        flags = pr["corporate_action_flags"]
        print(
            f"    corporate-action flags: {len(flags)}"
            f"{'  <-- REVIEW THESE' if flags else '  (adjclose looks clean)'}"
        )
        for f in flags[:5]:
            print(f"        {f['symbol']} {f['trade_date']}: {f['return'] * 100:.1f}%")

    if "reg31" in results:
        r = results["reg31"]
        print(f"\n  Reg 31 events         : {r['events']:,}  types={r['types']}")

    print(f"\n  ledger status         : {ledger}")
    print("\n  row counts")
    for name in ("companies", "filings", "pledge_state", "pledge_events", "prices", "benchmark"):
        print(f"    {name:<16} {counts[name]:>10,}")

    print("\n  next: python scripts/03_build_panel.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
