"""01 - Build the company universe and discover their filings.

PLAN.md sec.16 Phase 2: "01_build_universe.py (300 symbols -> companies)".

Runs three steps, all idempotent:

  1. fetch the NIFTY 500 constituents          -> companies + data/universe.csv
  2. query the shareholding master per company -> filings ledger (status=pending)
  3. apply the completeness filter              -> ~300 kept, rest in_universe=0

Step 2 is JSON only. No XBRL is downloaded here - that is 02_ingest_all.py -
because knowing which companies have a full history is what step 3 needs, and
discovering it costs one cheap call per company instead of 20 file downloads.

    python scripts/01_build_universe.py
    python scripts/01_build_universe.py --limit 5      # pilot run
    python scripts/01_build_universe.py --skip-discovery
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  - must precede pledgecast/config imports
from config import get_settings
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.ingest import shareholding, universe
from pledgecast.ingest.nse_session import NSESession
from pledgecast.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PledgeCast universe.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N constituents. Use for a pilot run.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Restrict to specific symbols, e.g. --symbols JPPOWER TCS HFCL.",
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Only refresh companies + universe.csv; leave the filings ledger alone.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)
    settings.paths.ensure_exist()

    with NSESession(settings) as session, get_connection(settings=settings) as conn:
        # -- 1. constituents ------------------------------------------------
        rows = universe.fetch_constituents(session, settings)

        if args.symbols:
            wanted = {s.upper() for s in args.symbols}
            rows = [r for r in rows if r["symbol"].upper() in wanted]
            missing = wanted - {r["symbol"].upper() for r in rows}
            if missing:
                logger.warning("requested symbols not in the index: %s", sorted(missing))
        if args.limit:
            rows = rows[: args.limit]

        if not rows:
            logger.error("no constituents selected - nothing to do")
            return 1

        repo.upsert_companies(conn, rows)
        universe.write_universe_csv(rows, settings)

        # -- 2. filing discovery --------------------------------------------
        if args.skip_discovery:
            discovery = {
                "companies_queried": 0,
                "filings_found": 0,
                "failures": [],
                "per_symbol": {},
            }
        else:
            discovery = shareholding.discover_filings(session, conn, rows, settings)

        # -- 3. completeness filter -----------------------------------------
        selection: dict = {"kept": [], "dropped": [], "median_quarters_kept": 0.0}
        filings = repo.load_filings(conn)
        if not filings.empty:
            selection = shareholding.select_universe(conn, settings)

        counts = repo.table_counts(conn)
        per_symbol = discovery["per_symbol"]

    # ------------------------------------------------------------------ report
    print("=" * 78)
    print("PLEDGECAST - UNIVERSE BUILT")
    print("=" * 78)
    print(f"  index                 : {settings.universe.index_name}")
    print(f"  constituents fetched  : {len(rows)}")
    print(f"  companies in DB       : {counts['companies']:,}")
    print(f"  filings discovered    : {counts['filings']:,}")

    if per_symbol:
        found = sorted(per_symbol.values())
        full = sum(1 for v in found if v >= settings.window.expected_quarters)
        print(
            f"  quarters per company  : min={found[0]} median={found[len(found) // 2]} "
            f"max={found[-1]}"
        )
        print(
            f"  with the full {settings.window.expected_quarters} quarters : "
            f"{full}/{len(found)}"
        )

    if discovery["failures"]:
        print(f"  discovery failures    : {len(discovery['failures'])}")
        for symbol, reason in discovery["failures"][:5]:
            print(f"      {symbol}: {reason[:90]}")

    if selection["kept"]:
        print(
            f"\n  universe kept         : {len(selection['kept'])} "
            f"(>= {settings.universe.min_filings_required} quarters, "
            f"cap {settings.universe.target_size})"
        )
        print(f"  excluded (in_universe=0): {len(selection['dropped'])}")
        print(f"  median quarters kept  : {selection['median_quarters_kept']:.1f}")

    print(f"\n  universe.csv          : {settings.paths.universe_csv}")
    print("  next: python scripts/02_ingest_all.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
