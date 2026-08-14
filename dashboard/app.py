"""PledgeCast dashboard - entry, sidebar and the shared cache (PLAN.md sec.14).

    "You will not write a single line of HTML, CSS, or JavaScript."

    "One rule: never use st.markdown(..., unsafe_allow_html=True). The moment
     you do, you have started writing CSS you would have to defend."

That rule is honoured throughout - every layout here is ``st.columns``,
``st.tabs``, ``st.metric``, ``st.expander``, ``st.selectbox`` or
``st.dataframe``, and there is no HTML anywhere in ``dashboard/``.

**Every loader lives here, not in the pages.** sec.14.2 asks for
``@st.cache_data(ttl=300)`` on every data load; putting them in one module means
three pages share one cache instead of each warming its own. Pages import from
this module, which works because Streamlit places the main script's folder on
``sys.path``. The page body sits inside :func:`main` so that importing ``app``
from a page does not re-render the home page - and, more importantly, does not
call ``st.set_page_config`` a second time, which raises.

**The dashboard never calls the API** (sec.7.2): it imports
``inference/service.py`` directly, so the demo cannot be broken by a server that
is not running.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401  - must precede config/pledgecast imports
from config import get_settings
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.evaluation import power
from pledgecast.exceptions import PledgeCastError
from pledgecast.inference.service import PredictionService

SETTINGS = get_settings()
CACHE_TTL = SETTINGS.dashboard.cache_ttl_seconds


# --------------------------------------------------------------------------- #
# resources                                                                    #
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_service() -> PredictionService:
    """One service for the whole session - the model is unpickled once."""
    return PredictionService(SETTINGS)


# --------------------------------------------------------------------------- #
# cached loaders (sec.14.2)                                                    #
# --------------------------------------------------------------------------- #
# Each opens its own connection. A SQLAlchemy Connection is unhashable, so it
# can never be a cached function's argument - passing one in would fail at the
# cache layer rather than at the query.
@st.cache_data(ttl=CACHE_TTL)
def load_panel(valid_only: bool = False) -> pd.DataFrame:
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_panel(conn, valid_only=valid_only)


@st.cache_data(ttl=CACHE_TTL)
def load_predictions(run_id: str | None = None, source: str | None = None) -> pd.DataFrame:
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_predictions(conn, run_id=run_id, source=source)


@st.cache_data(ttl=CACHE_TTL)
def load_company_predictions(symbol: str) -> pd.DataFrame:
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_predictions(conn, symbol=symbol)


@st.cache_data(ttl=CACHE_TTL)
def load_model_runs() -> pd.DataFrame:
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_model_runs(conn)


@st.cache_data(ttl=CACHE_TTL)
def load_metrics() -> pd.DataFrame:
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_metrics(conn)


@st.cache_data(ttl=CACHE_TTL)
def load_backtest() -> pd.DataFrame:
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_backtest_results(conn)


@st.cache_data(ttl=CACHE_TTL)
def load_detectability() -> pd.DataFrame:
    """Every experiment's delta with a confidence interval, re-derived here.

    Recomputed from the stored out-of-fold predictions rather than read from
    ``model_metrics``, because an interval is a property of a PAIR of runs and
    the metrics table stores one run per row. Re-deriving also means the number
    on this page cannot drift away from the predictions it claims to describe.

    This exists because the comparison table below shows deltas of about 0.02
    against an interval half-width of about 0.03. Without the interval beside
    them, a reader ranks the experiments and believes the order.
    """
    with get_connection(settings=SETTINGS) as conn:
        runs = repo.load_model_runs(conn)
        panel = repo.load_panel(conn, valid_only=True)
        if runs.empty or panel.empty:
            return pd.DataFrame()

        # One training session only. Run ids are `<stamp>_<model>_<experiment>`,
        # and mixing stamps would compare runs fitted on different panels.
        stamp = runs["run_id"].str.split("_").str[0].max()
        session = runs[runs["run_id"].str.startswith(f"{stamp}_")]

        labels = panel[["symbol", "observation_date", "label"]]
        oof: dict[tuple[str, str], pd.DataFrame] = {}
        for row in session.itertuples(index=False):
            frame = repo.load_predictions(conn, run_id=row.run_id, source="backtest")
            if frame.empty:
                continue
            oof[(row.model_name, row.experiment)] = frame.merge(
                labels, on=["symbol", "observation_date"], how="inner"
            )

    rows = []
    for (model_name, experiment), treatment in oof.items():
        if experiment not in SETTINGS.experiments:
            continue
        baseline = SETTINGS.experiment_baseline(experiment)
        control = oof.get((model_name, baseline))
        if baseline == experiment or control is None:
            continue
        result = power.paired_delta_ci(
            treatment,
            control,
            n_bootstrap=SETTINGS.power.n_bootstrap,
            confidence_level=SETTINGS.power.confidence_level,
            seed=SETTINGS.power.bootstrap_seed,
            min_rows=SETTINGS.evaluation.min_rows_per_quarter_for_auc,
        )
        rows.append(
            {
                "experiment": experiment,
                "vs": baseline,
                "model": model_name,
                "delta": result.get("delta"),
                "ci_low": result.get("ci_low"),
                "ci_high": result.get("ci_high"),
                "min_detectable": result.get("half_width"),
                "dates": result.get("n_dates"),
                "verdict": result.get("verdict"),
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(["experiment", "model"]) if not table.empty else table


@st.cache_data(ttl=CACHE_TTL)
def load_pledge_state(symbol: str | None = None) -> pd.DataFrame:
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_pledge_state(conn, symbol=symbol)


@st.cache_data(ttl=CACHE_TTL)
def load_pledge_events(symbol: str | None = None) -> pd.DataFrame:
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_pledge_events(conn, symbol=symbol)


@st.cache_data(ttl=CACHE_TTL)
def load_prices(symbol: str) -> pd.DataFrame:
    # `symbols` (plural) - the repository filters on a sequence, not one string.
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_prices(conn, symbols=[symbol])


@st.cache_data(ttl=CACHE_TTL)
def load_companies() -> pd.DataFrame:
    with get_connection(settings=SETTINGS) as conn:
        return repo.load_companies(conn, in_universe=True)


@st.cache_data(ttl=CACHE_TTL)
def load_active_run() -> dict | None:
    with get_connection(settings=SETTINGS) as conn:
        return repo.get_active_run(conn)


@st.cache_data(ttl=CACHE_TTL)
def load_model_info() -> dict:
    with get_connection(settings=SETTINGS) as conn:
        return get_service().model_info(conn)


@st.cache_data(ttl=CACHE_TTL)
def score_date(observation_date: str) -> pd.DataFrame:
    """Score one observation date through the shared service (sec.7.2).

    ``persist=False``: browsing the dashboard should not append a prediction row
    per page view. sec.13's "every prediction is persisted" is about serving
    decisions, and the scanner is a view over scores the batch run already made.
    """
    with get_connection(settings=SETTINGS) as conn:
        return get_service().score_date(conn, observation_date, persist=False)


@st.cache_data(ttl=CACHE_TTL)
def company_history(symbol: str) -> dict:
    with get_connection(settings=SETTINGS) as conn:
        return get_service().company_history(conn, symbol)


# --------------------------------------------------------------------------- #
# the four sec.10 failure modes, rendered readably                             #
# --------------------------------------------------------------------------- #
def check_ready() -> bool:
    """Report DB and model state in words. sec.10: "never a traceback"."""
    try:
        panel = load_panel()
    except PledgeCastError as exc:
        st.error(f"Database unavailable: {exc}")
        st.caption("Run `make init-db && make ingest && make build`.")
        return False
    except Exception as exc:  # noqa: BLE001 - a dashboard must not show a stack trace
        st.error(f"Could not read the database: {exc}")
        return False

    if panel.empty:
        st.warning("The panel is empty - no data for this selection.")
        st.caption("Run `make build` to assemble the point-in-time panel.")
        return False

    if load_active_run() is None:
        st.error("No active model - run `make train`.")
        return False
    return True


def guard(fn, *args, **kwargs):
    """Run a loader, turning any failure into a readable message (sec.10)."""
    try:
        return fn(*args, **kwargs)
    except PledgeCastError as exc:
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")
    return None


def sidebar() -> None:
    """Shared across all pages - which model is answering, and how good it is."""
    run = load_active_run()
    with st.sidebar:
        st.subheader("Active model")
        if run is None:
            st.error("none - run `make train`")
            return
        st.caption(run["run_id"])
        st.write(f"**{run['model_name']}** / {run['experiment']}")
        st.write(f"{len(run['feature_list'])} features · seed {run['random_seed']}")

        info = guard(load_model_info)
        if info:
            auc = info["metrics"].get("within_quarter_auc")
            if auc is not None:
                st.metric("within-quarter AUC", f"{auc:.4f}")
            st.caption(f"trained {run['created_at'][:10]} · {run['n_folds']} folds")

        st.divider()
        st.caption(
            "Scores come from src/inference/service.py - the same code path the "
            "API uses. The dashboard does not call the API."
        )


# --------------------------------------------------------------------------- #
# home                                                                         #
# --------------------------------------------------------------------------- #
def _headline(metrics: pd.DataFrame, runs: pd.DataFrame) -> tuple[float | None, pd.DataFrame]:
    """expB_full - exp0_null on the primary metric, model by model (sec.2.3)."""
    merged = metrics[metrics["fold"] == -1].merge(runs[["run_id", "model_name", "experiment"]])
    primary = merged[merged["metric_name"] == SETTINGS.headline.metric]
    table = primary.pivot_table(
        index="model_name", columns="experiment", values="metric_value", aggfunc="last"
    )
    experiment, baseline = SETTINGS.headline.experiment, SETTINGS.headline.baseline
    if experiment not in table.columns or baseline not in table.columns:
        return None, table
    table["delta"] = table[experiment] - table[baseline]
    return float(table["delta"].median()), table


def main() -> None:
    st.set_page_config(page_title="PledgeCast", page_icon=":chart_with_downwards_trend:",
                       layout="wide")
    st.title("PledgeCast")
    st.caption(
        "Explainable early warning for promoter-pledge-driven downside risk in Indian equities."
    )

    if not check_ready():
        return
    sidebar()

    metrics, runs = load_metrics(), load_model_runs()
    delta, table = _headline(metrics, runs)

    st.header("The result")
    st.write(
        f"**Does pledge trajectory add anything over what volatility and size already tell you?** "
        f"The whole project answers one question, and the answer is the difference between two "
        f"experiments on the same metric: `{SETTINGS.headline.experiment}` minus "
        f"`{SETTINGS.headline.baseline}`, measured by within-quarter ROC-AUC."
    )

    if delta is not None:
        left, right = st.columns([1, 2])
        with left:
            st.metric(
                "median delta across models",
                f"{delta:+.4f}",
                delta=f"{'adds signal' if delta > 0 else 'adds nothing'}",
                delta_color="inverse" if delta <= 0 else "normal",
            )
        with right:
            st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

        if delta <= 0:
            st.info(
                "Pledge trajectory adds no incremental early warning once volatility and size "
                "are accounted for. That is a legitimate result, not a failed one - the study "
                "was designed to be able to return it, and reporting it is the point."
            )

    st.divider()
    panel = load_panel()
    a, b, c, d = st.columns(4)
    a.metric("companies", f"{panel['symbol'].nunique():,}")
    b.metric("observation dates", f"{panel['observation_date'].nunique():,}")
    c.metric("panel rows", f"{len(panel):,}")
    labelled = panel[panel["label_is_valid"] == 1]
    d.metric("event rate", f"{labelled['label'].mean():.1%}")

    st.caption(
        "Use the pages in the sidebar: **Risk Scanner** for the watchlist, "
        "**Company Investigation** for one company, **Model Validation** for how far to trust it."
    )


if __name__ == "__main__":
    main()
