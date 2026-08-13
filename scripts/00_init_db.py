"""00 - Create the SQLite schema.

PLAN.md sec.16 Phase 2: "00_init_db.py (schema, WAL)".

Idempotent: safe to re-run at any time. Use ``--force`` to drop and recreate
(destructive - it discards every prediction and model run recorded so far).

    python scripts/00_init_db.py
    python scripts/00_init_db.py --force
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import text

import _bootstrap  # noqa: F401  - must precede pledgecast/config imports
from config import get_settings
from pledgecast.db import repository
from pledgecast.db.connection import create_all, drop_all, get_connection, healthcheck
from pledgecast.db.schema import ALL_TABLES
from pledgecast.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the PledgeCast SQLite schema.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="DROP every table first. Destroys all predictions and model runs.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)

    # Directories that raw files, models, logs and figures land in (sec.5.2).
    settings.paths.ensure_exist()

    logger.info("database: %s", settings.db_path)

    with get_connection(settings=settings) as conn:
        if args.force:
            logger.warning("--force: dropping all tables")
            drop_all(conn)

        before = {
            r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))
        }
        create_all(conn)
        after = {
            r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))
        }
        created = sorted(set(ALL_TABLES) & (after - before))

        indexes = [
            r[0]
            for r in conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type = 'index' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            )
        ]
        counts = repository.table_counts(conn)

    health = healthcheck(settings=settings)

    print("=" * 74)
    print("PLEDGECAST - DATABASE INITIALISED")
    print("=" * 74)
    print(f"  path          : {settings.db_path}")
    print(f"  journal_mode  : {health['journal_mode']}   (WAL = concurrent API + dashboard reads)")
    print(f"  foreign_keys  : {health['foreign_keys']}")
    print(f"  tables        : {health['tables_present']}/{health['tables_expected']}")
    print(f"  indexes       : {len(indexes)}")
    if created:
        print(f"  newly created : {', '.join(created)}")
    else:
        print("  newly created : none (schema already present)")

    print("\n  row counts")
    for name in sorted(counts):
        print(f"    {name:<20} {counts[name]:>8,}")

    if health["missing_tables"]:
        print(f"\n  MISSING TABLES: {health['missing_tables']}")
        print("=" * 74)
        return 1

    print("\n  next: python scripts/01_build_universe.py")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
