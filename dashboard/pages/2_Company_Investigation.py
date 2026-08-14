"""Page 2 - Company Investigation (PLAN.md sec.14.1).

    "Symbol dropdown · metric row (current probability, pledge %, QoQ change,
     volatility) · pledge trajectory with Reg 31 markers · price chart with
     shaded event windows · SHAP waterfall · human-readable explanation ·
     prediction history table."

sec.11.1 calls the templated sentence the thing that "makes it feel like a risk
tool instead of a black box", so it sits above the waterfall rather than below
it: the reader gets the claim in English first and the arithmetic underneath.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import _bootstrap  # noqa: F401  - must precede config/pledgecast imports
from app import (
    SETTINGS,
    get_service,
    guard,
    load_companies,
    load_company_predictions,
    load_panel,
    load_pledge_events,
    load_pledge_state,
    load_prices,
    score_date,
    sidebar,
)
from pledgecast.db.connection import get_connection
from pledgecast.explain import shap_runner

st.title("Company Investigation")

companies = guard(load_companies)
panel = guard(load_panel)
if companies is None or panel is None or panel.empty:
    st.warning("No data for this selection.")
    st.stop()

sidebar()

symbols = sorted(panel["symbol"].unique())
names = dict(zip(companies["symbol"], companies["company_name"], strict=False))

symbol = st.selectbox(
    "Company",
    symbols,
    index=symbols.index("JPPOWER") if "JPPOWER" in symbols else 0,
    format_func=lambda s: f"{s} - {names.get(s, s)}",
)

rows = panel[panel["symbol"] == symbol].sort_values("observation_date")
if rows.empty:
    st.warning(f"No panel rows for {symbol}.")
    st.stop()

latest_date = rows["observation_date"].iloc[-1]
scored = guard(score_date, latest_date)
current = None
if scored is not None and not scored.empty:
    match = scored[scored["symbol"] == symbol]
    current = None if match.empty else match.iloc[0]

# ------------------------------------------------------------------- metrics
st.subheader(f"{names.get(symbol, symbol)} - as of {latest_date}")
a, b, c, d = st.columns(4)
if current is not None:
    a.metric("predicted probability", f"{current['probability']:.3f}",
             delta=current["risk_band"], delta_color="off")
else:
    a.metric("predicted probability", "n/a")

last = rows.iloc[-1]


def _show(column, label, value, fmt: str, suffix: str = ""):
    column.metric(label, "n/a" if pd.isna(value) else f"{format(value, fmt)}{suffix}")


_show(b, "pledged % of promoter stake", last["pledge_pct_promoter"], ".1f", "%")
_show(c, "pledge change, 1 quarter", last["pledge_chg_1q"], "+.1f", " pp")
_show(d, "volatility 90d", last["volatility_90d"], ".2f")

if current is not None and len(current["warnings"]) > 1:
    for warning in current["warnings"][:-1]:
        st.warning(warning)

# -------------------------------------------------------------------- charts
trajectory, price, explanation, history = st.tabs(
    ["Pledge trajectory", "Price & event windows", "Why this score", "Prediction history"]
)

with trajectory:
    state = guard(load_pledge_state, symbol)
    events = guard(load_pledge_events, symbol)
    if state is None or state.empty:
        st.info("No pledge history recorded for this company.")
    else:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=state["quarter_end"],
                y=state["pledge_pct_promoter"],
                mode="lines+markers",
                name="% of promoter stake pledged",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=state["quarter_end"],
                y=state["pledge_pct_equity"],
                mode="lines+markers",
                name="% of total equity pledged",
            )
        )

        # sec.12: Reg 31 creation/release/invocation overlaid on the pledge line.
        if events is not None and not events.empty:
            for kind, symbol_marker in (
                ("creation", "triangle-up"),
                ("release", "triangle-down"),
                ("invocation", "x"),
            ):
                block = events[events["event_type"] == kind]
                if block.empty:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=block["event_date"],
                        y=[0] * len(block),
                        mode="markers",
                        name=f"Reg 31 {kind} ({len(block)})",
                        marker={"symbol": symbol_marker, "size": 9},
                        hovertext=block["promoter_name"],
                    )
                )

        figure.update_layout(
            title=f"{symbol} - pledge trajectory with Reg 31 disclosures",
            yaxis_title="percent",
            hovermode="x unified",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "Quarterly shareholding filings give the level; Reg 31 disclosures give the "
            "individual events between them. Measured across this panel the quarterly level "
            "is unchanged in 90.5% of quarters - the pledge moves far more slowly than the "
            "event stream suggests."
        )

with price:
    prices = guard(load_prices, symbol)
    if prices is None or prices.empty:
        st.info("No price history for this company.")
    else:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=prices["trade_date"], y=prices["adj_close"], name="adjusted close")
        )

        # Shade the forward window of every quarter that became an event.
        events_in_panel = rows[(rows["label"] == 1) & (rows["label_is_valid"] == 1)]
        for row in events_in_panel.itertuples(index=False):
            start = pd.Timestamp(row.observation_date)
            end = start + pd.Timedelta(days=int(SETTINGS.label.horizon_trading_days * 1.45))
            figure.add_vrect(
                x0=start,
                x1=end,
                fillcolor="crimson",
                opacity=0.15,
                line_width=0,
            )

        figure.update_layout(
            title=f"{symbol} - adjusted close, shaded where a {SETTINGS.label.drawdown_threshold:.0%} "
            f"drawdown followed",
            yaxis_title="price",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            f"A shaded band marks an observation date whose next "
            f"{SETTINGS.label.horizon_trading_days} trading days contained a decline of "
            f"{SETTINGS.label.drawdown_threshold:.0%} or worse from that date's price - the "
            "label the model is trained to predict. Always measured on adjusted close."
        )

with explanation:
    if current is None:
        st.info("This company was not scored on the latest date.")
    else:
        service = get_service()
        try:
            with get_connection(settings=SETTINGS) as conn:
                info = service.model_info(conn)
                features = info["features"]
                records = service.explain_row(conn, current, features)
                detail = service.explain_detail(conn, current, features)

            st.info(
                shap_runner.summarise(
                    records,
                    float(current["probability"]),
                    decile=int(current["risk_decile"]),
                    band=current["risk_band"],
                    top_n=SETTINGS.explain.top_n_features,
                )
            )
            st.caption(
                "Generated from ranked SHAP values by a template - no language model is "
                "involved anywhere in this project (sec.11.1)."
            )

            figure = shap_runner.waterfall(
                detail["explainer"],
                detail["values"],
                detail["matrix"],
                detail["names"],
                index=0,
                max_display=SETTINGS.explain.beeswarm_max_display,
            )
            st.pyplot(figure, clear_figure=True)

            st.dataframe(
                pd.DataFrame(shap_runner.merge_indicators(records)).rename(
                    columns={
                        "feature_name": "feature",
                        "feature_value": "value",
                        "shap_value": "SHAP",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:  # noqa: BLE001 - sec.10: partial response, never a traceback
            st.warning(f"Could not explain this prediction: {exc}")

with history:
    predictions = guard(load_company_predictions, symbol)
    if predictions is None or predictions.empty:
        st.info("No stored predictions for this company yet - run `make score`.")
    else:
        st.dataframe(
            predictions[
                ["observation_date", "probability", "risk_decile", "source", "run_id"]
            ].sort_values("observation_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Every prediction is persisted as a side effect of serving (sec.13) - "
            "walk-forward rows are `backtest`, live API calls are `api`."
        )
