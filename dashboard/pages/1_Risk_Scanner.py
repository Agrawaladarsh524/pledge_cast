"""Page 1 - Risk Scanner (PLAN.md sec.14.1).

    "Quarter selector · top-N slider · ranked table (symbol, probability,
     decile, pledge %, delta pledge, volatility) · risk histogram ·
     pledge-vs-risk scatter coloured by volatility."

The scatter is starred in sec.12 as the chart that "visually exposes the
confound", and it earns that: colour it by volatility and the high-risk end
fills with hot points at every pledge level, including zero. The relationship
the eye wants to see between pledge % and risk is largely volatility wearing a
pledge label.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

import _bootstrap  # noqa: F401  - must precede config/pledgecast imports
from app import SETTINGS, guard, load_panel, score_date, sidebar

st.title("Risk Scanner")
st.caption("Ranked watchlist for one observation date. Scores come from the shared service.")

panel = guard(load_panel)
if panel is None or panel.empty:
    st.warning("No data for this selection.")
    st.stop()

sidebar()

dates = sorted(panel["observation_date"].unique(), reverse=True)
left, right = st.columns([2, 3])
with left:
    date = st.selectbox("Observation date", dates, index=0)
with right:
    top_n = st.slider(
        "Show top N",
        min_value=5,
        max_value=SETTINGS.dashboard.max_top_n,
        value=SETTINGS.dashboard.default_top_n,
        step=5,
    )

scored = guard(score_date, date)
if scored is None or scored.empty:
    st.warning(f"No companies could be scored for {date}.")
    st.stop()

# The most recent date is the sec.9.4 embargo quarter: featured, never labelled.
# It is the only genuinely forward-looking view, and saying so matters more than
# it costs.
if (scored["label_is_valid"] == 0).all():
    st.info(
        f"{date} has no realised outcome yet - its label needs "
        f"{SETTINGS.label.horizon_trading_days} trading days of future prices. "
        "These are forward predictions, not scored history."
    )

a, b, c, d = st.columns(4)
a.metric("companies scored", f"{len(scored):,}")
b.metric("median probability", f"{scored['probability'].median():.3f}")
b.caption(f"base rate {SETTINGS.label.expected_event_rate:.0%}")
c.metric("pledged companies", f"{int((scored['pledge_pct_promoter'] > 0).sum()):,}")
d.metric(
    "HIGH or CRITICAL",
    f"{int(scored['risk_band'].isin(['HIGH', 'CRITICAL']).sum()):,}",
)

# ------------------------------------------------------------------ the table
st.subheader(f"Top {top_n} by predicted risk")
columns = {
    "symbol": "symbol",
    "probability": "probability",
    "risk_decile": "decile",
    "risk_band": "band",
    "pledge_pct_promoter": "pledge % of promoter",
    "pledge_chg_1q": "pledge change 1q (pp)",
    "volatility_90d": "volatility 90d",
}
table = scored.head(top_n)[list(columns)].rename(columns=columns)
st.dataframe(
    table.style.format(
        {
            "probability": "{:.3f}",
            "pledge % of promoter": "{:.1f}",
            "pledge change 1q (pp)": "{:+.1f}",
            "volatility 90d": "{:.2f}",
        },
        na_rep="n/a",
    ),
    use_container_width=True,
    hide_index=True,
)

flagged = scored.head(top_n)
flagged = flagged[flagged["warnings"].map(len) > 1]
if not flagged.empty:
    with st.expander(f"{len(flagged)} of the top {top_n} carry data warnings"):
        for row in flagged.itertuples(index=False):
            st.write(f"**{row.symbol}** - {'; '.join(row.warnings[:-1])}")

# ------------------------------------------------------------------- charts
histogram, scatter = st.tabs(["Risk distribution", "Pledge vs risk (the confound)"])

with histogram:
    figure = px.histogram(
        scored,
        x="probability",
        nbins=30,
        color="risk_band",
        category_orders={"risk_band": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
        labels={"probability": "predicted probability"},
        title=f"Predicted risk across {len(scored)} companies on {date}",
    )
    figure.add_vline(
        x=SETTINGS.label.expected_event_rate,
        line_dash="dash",
        annotation_text="base rate",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Is this a risky quarter overall, or a calm one? The whole distribution shifts "
        "quarter to quarter - measured on this panel the event rate runs from 1.0% to 60.7%, "
        "which is why the model is scored within a date rather than across all of them."
    )

with scatter:
    plot = scored.dropna(subset=["pledge_pct_promoter", "volatility_90d"])
    figure = px.scatter(
        plot,
        x="pledge_pct_promoter",
        y="probability",
        color="volatility_90d",
        hover_name="symbol",
        color_continuous_scale="Turbo",
        labels={
            "pledge_pct_promoter": "% of promoter stake pledged",
            "probability": "predicted probability",
            "volatility_90d": "volatility 90d",
        },
        title="Pledge % against predicted risk, coloured by volatility",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "sec.12 marks this the chart that exposes the confound. Risk rises with colour "
        "(volatility) far more cleanly than with position on the x-axis (pledge). Companies "
        "at 0% pledged reach the top of the risk range, and heavily pledged companies sit "
        "at the bottom of it."
    )
