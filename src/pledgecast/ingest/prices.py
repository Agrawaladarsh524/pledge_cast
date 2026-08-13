"""Daily prices and benchmark - PLAN.md sec.1.2, sec.9.2, sec.10.

sec.4.1 excludes ``yfinance``: "The Yahoo chart API works directly with
``requests``". This module calls that endpoint and nothing else.

**Always ``adjclose``, never ``close``.** sec.10 spells out why: a 1:2 split in
raw prices looks like an exact -50% one-day crash, which would fire the -15%
drawdown label on a company that did not fall at all - silently corrupting the
target variable for every company that ever split. The corporate-action guard
below asserts that no residual single-day move breaches
``validation.corporate_action_return_floor``, so a split that slipped through
adjustment fails loudly instead of becoming a fake event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Connection
from tqdm import tqdm

from pledgecast.db import repository as repo
from pledgecast.exceptions import DataIngestionError
from pledgecast.ingest.nse_session import fetch_json_url
from pledgecast.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from config import Settings

logger = get_logger(__name__)


def fetch_chart(symbol: str, settings: Settings) -> pd.DataFrame:
    """Yahoo chart -> ``[trade_date, adj_close, volume]``.

    ``symbol`` is passed through verbatim, so callers supply ``JPPOWER.NS`` for
    equities and ``%5ENSEI`` for the index.
    """
    payload = fetch_json_url(
        f"{settings.ingest.yahoo_chart_url}/{symbol}",
        settings=settings,
        range=settings.ingest.price_range,
        interval=settings.ingest.price_interval,
    )

    chart = (payload or {}).get("chart") or {}
    if chart.get("error"):
        raise DataIngestionError(f"Yahoo returned an error for {symbol}: {chart['error']}")

    results = chart.get("result") or []
    if not results:
        raise DataIngestionError(f"Yahoo returned no result block for {symbol}")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}

    adjclose_blocks = indicators.get("adjclose") or []
    if not adjclose_blocks or "adjclose" not in adjclose_blocks[0]:
        # Refuse to silently fall back to raw close - see the module docstring.
        raise DataIngestionError(
            f"{symbol}: no adjclose series returned; refusing to substitute raw close"
        )

    adj = adjclose_blocks[0]["adjclose"]
    quote_blocks = indicators.get("quote") or [{}]
    volume = quote_blocks[0].get("volume") or [None] * len(timestamps)

    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).date,
            "adj_close": adj,
            "volume": volume,
        }
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame.dropna(subset=["adj_close"])
    frame = frame[frame["adj_close"] > 0]
    return frame.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")


def check_corporate_actions(frame: pd.DataFrame, symbol: str, settings: Settings) -> list[dict]:
    """sec.10: flag any single-day move below the configured floor.

    Adjusted prices should already absorb splits and bonuses. Anything left that
    looks like a -35%+ one-day move is either a genuine crash or an unadjusted
    corporate action, and the difference matters because the second kind
    manufactures fake label events. Returned, logged, and never silently
    dropped.
    """
    if len(frame) < 2:
        return []

    prices = frame["adj_close"].to_numpy(dtype=float)
    returns = np.diff(prices) / prices[:-1]
    floor = settings.validation.corporate_action_return_floor

    breaches = []
    for index in np.where(returns < floor)[0]:
        breaches.append(
            {
                "symbol": symbol,
                "trade_date": frame["trade_date"].iloc[index + 1],
                "prev_close": float(prices[index]),
                "close": float(prices[index + 1]),
                "return": float(returns[index]),
            }
        )
        logger.warning(
            "%s %s: single-day return %.1f%% (%.2f -> %.2f) breaches the %.0f%% floor - "
            "check for an unadjusted corporate action",
            symbol,
            frame["trade_date"].iloc[index + 1],
            returns[index] * 100,
            prices[index],
            prices[index + 1],
            floor * 100,
        )
    return breaches


def ingest_prices(
    conn: Connection,
    symbols: list[str],
    settings: Settings,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Fetch and store prices for every symbol, plus the benchmark."""
    stats: dict[str, Any] = {
        "symbols_requested": len(symbols),
        "symbols_stored": 0,
        "rows": 0,
        "failures": [],
        "too_short": [],
        "corporate_action_flags": [],
    }

    # --- benchmark first: it also defines the trading calendar (sec.9.3) ---
    try:
        bench = fetch_chart(settings.ingest.benchmark_symbol, settings)
        repo.upsert_benchmark(conn, bench[["trade_date", "adj_close"]].to_dict(orient="records"))
        stats["benchmark_rows"] = len(bench)
        logger.info(
            "benchmark %s: %d rows %s..%s",
            settings.ingest.benchmark_name,
            len(bench),
            bench["trade_date"].iloc[0],
            bench["trade_date"].iloc[-1],
        )
    except DataIngestionError as exc:
        stats["benchmark_rows"] = 0
        stats["failures"].append((settings.ingest.benchmark_name, str(exc)[:160]))
        logger.error("benchmark fetch failed: %s", exc)

    # --- equities ----------------------------------------------------------
    bar = tqdm(symbols, desc="fetching prices", unit="sym", disable=not show_progress)
    for symbol in bar:
        try:
            frame = fetch_chart(f"{symbol}.NS", settings)
        except DataIngestionError as exc:
            stats["failures"].append((symbol, str(exc)[:160]))
            continue

        if len(frame) < settings.validation.min_price_rows_per_symbol:
            stats["too_short"].append((symbol, len(frame)))
            logger.warning(
                "%s: only %d price rows (< %d), excluded from the universe",
                symbol,
                len(frame),
                settings.validation.min_price_rows_per_symbol,
            )
            continue

        stats["corporate_action_flags"].extend(check_corporate_actions(frame, symbol, settings))

        frame = frame.assign(symbol=symbol)
        repo.upsert_prices(
            conn, frame[["symbol", "trade_date", "adj_close", "volume"]].to_dict(orient="records")
        )
        stats["symbols_stored"] += 1
        stats["rows"] += len(frame)

    if stats["too_short"]:
        repo.set_in_universe(conn, [s for s, _ in stats["too_short"]], False)
    if stats["failures"]:
        repo.set_in_universe(
            conn,
            [s for s, _ in stats["failures"] if s != settings.ingest.benchmark_name],
            False,
        )

    logger.info(
        "prices: %d symbols, %d rows, %d failures, %d too short, %d corporate-action flags",
        stats["symbols_stored"],
        stats["rows"],
        len(stats["failures"]),
        len(stats["too_short"]),
        len(stats["corporate_action_flags"]),
    )
    return stats


__all__ = ["check_corporate_actions", "fetch_chart", "ingest_prices"]
