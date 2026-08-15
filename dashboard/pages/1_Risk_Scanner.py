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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import _bootstrap  # noqa: F401  - must precede config/pledgecast imports
import components as C
import theme
from app import SETTINGS, guard, load_companies, load_panel, score_date, sidebar

st.title("Risk Scanner")
st.caption(
    "Ranked watchlist for one observation date. Scores come from the shared service - "
    "the same code path the API uses."
)

# The finding, before the ranking. This is the page most easily mistaken for an
# actionable stock screen, and the statement that its ranking is essentially a
# volatility ranking used to sit in the second tab, below the fold, behind a
# click. A reader who deep-linked here never met it.
C.finding_banner()

panel = guard(load_panel)
if panel is None or panel.empty:
    C.empty_state(
        "The panel is empty, so there is nothing to rank.",
        "Run `make init-db && make ingest && make build`.",
    )
    st.stop()

sidebar()

dates = sorted(panel["observation_date"].unique(), reverse=True)
# The forward date is marked in the selector rather than after selection: it was
# only discoverable by choosing it and paying for a rescore.
forward = {
    d for d in dates
    if not panel.loc[panel["observation_date"] == d, "label_is_valid"].astype(bool).any()
}
# 3:2, not 2:3 - the date changes the entire dataset while the size control only
# truncates a table, so the more consequential control gets the wider column.
left, right = st.columns([3, 2])
with left:
    date = st.selectbox(
        "Observation date",
        dates,
        index=0,
        format_func=lambda d: f"{d}   (forward - no outcome yet)" if d in forward else str(d),
    )
with right:
    top_n = st.slider(
        "Rows to show",
        min_value=5,
        max_value=SETTINGS.dashboard.max_top_n,
        value=SETTINGS.dashboard.default_top_n,
        step=5,
        help=f"Truncates the filtered set. The cohort is "
        f"{panel['symbol'].nunique()} companies; this caps the view at "
        f"{SETTINGS.dashboard.max_top_n}.",
    )

# Freshness. Every number on this page is scoped to one date and nothing said how
# old that date was, so a quarter-old view and a two-year-old view looked
# identical.
_age = (pd.Timestamp.today().normalize() - pd.Timestamp(date)).days
st.caption(
    f"Observation date {date} - {_age:,} days ago. "
    f"Newest date in the panel: {dates[0]}."
)

with st.spinner(f"Scoring {date}..."):
    scored = guard(score_date, date)
if scored is None or scored.empty:
    C.empty_state(
        f"No companies could be scored for {date}.",
        "The panel has rows for this date but the model produced none - check `make train`.",
    )
    st.stop()

# Measured from the panel at render time, not hardcoded. The caption below the
# histogram used to quote "1.0% to 60.7%" as prose, which goes stale silently the
# next time `make build` runs against more data.
_rates = (
    panel[panel["label_is_valid"] == 1].groupby("observation_date")["label"].mean()
)
event_rate_low, event_rate_high = (float(_rates.min()), float(_rates.max())) if len(
    _rates
) else (0.0, 0.0)

# The most recent date is the sec.9.4 embargo quarter: featured, never labelled.
# It is the only genuinely forward-looking view, and saying so matters more than
# it costs.
labelled_date = not (scored["label_is_valid"] == 0).all()
if not labelled_date:
    st.info(
        f"{date} has no realised outcome yet - its label needs "
        f"{SETTINGS.label.horizon_trading_days} trading days of future prices. "
        "These are forward predictions, not scored history."
    )

# Company names, so the ranked output is readable. Bare tickers made the primary
# result of the primary discovery page a lookup exercise for 300 NIFTY 500
# constituents; the loader is already cached and shared with page 2.
companies = guard(load_companies)
names = (
    dict(zip(companies["symbol"], companies["company_name"], strict=False))
    if companies is not None
    else {}
)
scored = scored.assign(company=scored["symbol"].map(lambda s: names.get(s, "")))

# Bands at or above the HIGH cutoff, derived from config rather than named as
# literals - renaming a band in config.yaml used to drop this count to zero.
bands = SETTINGS.evaluation.risk_bands
high_cut = bands.get("HIGH", 0.6)
elevated = {name for name, upper in bands.items() if upper > high_cut} | {"HIGH"}

# ---------------------------------------------------------------- the table
C.section(
    "Ranked by predicted risk",
    "Select a row to investigate that company.",
    top=True,
)

pick_bands = st.pills(
    "Risk band",
    theme.BAND_ORDER,
    selection_mode="multi",
    default=theme.BAND_ORDER,
    label_visibility="collapsed",
)
pledged_only = st.checkbox(
    "Pledged companies only",
    value=False,
    help="Excludes companies with no pledge and those whose pledge state was "
    "never determinable - the two are not the same thing.",
)

filtered = scored[scored["risk_band"].isin(pick_bands or theme.BAND_ORDER)]
if pledged_only:
    filtered = filtered[filtered["pledge_pct_promoter"] > 0]
shown = filtered.head(top_n)
# Percent, not a bare decimal. The page reports a base rate as a percentage, so
# showing probability as 0.312 beside it asked the reader to convert between
# representations across adjacent elements. column_config takes printf-style
# formats only on this version, so the scaling happens here rather than in the
# format string - "%.1f%%" on a 0-1 value would render 0.312 as "0.3%".
shown = shown.assign(probability_pct=shown["probability"] * 100.0)

if shown.empty:
    C.empty_state(
        "No companies match those filters on this date.",
        "Widen the band selection, or clear the pledged-only filter.",
    )
else:
    selection = C.table(
        shown,
        {
            "symbol": C.Column("symbol", kind="text", width="small"),
            "company": C.Column("company", kind="text",
                                help="From the universe table (sec.8.1)."),
            "probability_pct": C.Column(
                "probability", format="%.1f%%", kind="progress",
                extra={"min_value": 0.0, "max_value": 100.0},
                help="A RISK SCORE, not a calibrated forecast: the model's Brier skill "
                     "is at or below zero, so 31% should be read as 'riskier than 12%' "
                     "rather than as a 31% chance. See Model Validation. Differences "
                     "below about 3 percentage points are inside this design's "
                     "measurement interval, so ordering within a band is not meaningful."),
            "risk_decile": C.Column("decile", format="%d",
                                    help="Rank within this observation date, 10 = riskiest."),
            "risk_band": C.Column("band", kind="text",
                                  help=", ".join(f"{n} < {u:.2f}" for n, u in
                                                 sorted(bands.items(), key=lambda kv: kv[1]))),
            "pledge_pct_promoter": C.Column(
                "pledged", format="%.1f%%",
                help="Share of the promoter's own holding that is pledged."),
            "pledge_chg_1q": C.Column(
                "1q change", format="%+.1f pp",
                help="Change in that share over one quarter, in percentage points. "
                     "Unchanged in about nine company-quarters in ten."),
            "volatility_90d": C.Column(
                "volatility", format="%.2f",
                help="Annualised 90-day realised volatility. The feature that does most "
                     "of the work in this ranking."),
        },
        selection_mode="single-row",
        on_select="rerun",
        key="scanner_rows",
    )

    # Scanner -> Company. The page existed to say which companies to look at and
    # then offered no way to look at one: the reader had to memorise a ticker
    # and re-find it in a 300-item dropdown on another page.
    picked = selection.selection.rows if selection and selection.selection else []
    if picked:
        symbol = shown.iloc[picked[0]]["symbol"]
        st.session_state["investigate_symbol"] = symbol
        # Same entrypoint-relative resolution as st.page_link: correct under
        # `streamlit run dashboard/app.py`, and an exception when this page is
        # run on its own. The symbol is already in session state either way, so
        # the fallback loses the jump, not the intent.
        try:
            st.switch_page("pages/2_Company_Investigation.py")
        except Exception:  # noqa: BLE001 - sec.10: a message, never a traceback
            st.info(f"Open **Company Investigation** to see {symbol}.")

    C.data_quality(shown, scope=f"the {len(shown)} rows shown")

st.caption(
    f"Showing {len(shown)} of {len(filtered)} matching companies, out of {len(scored)} "
    f"scored on {date}."
)

# The cohort summary sits BELOW the ranked table: the list is the answer and the
# summary is context, so it does not belong above the answer. Two metrics that
# move, not four - "companies scored" is ~300 by construction on every date.
n_unknown = int(scored["pledge_pct_promoter"].isna().sum())
C.metric_row(
    C.Metric(
        "median probability",
        f"{scored['probability'].median():.1%}",
        help="Across all companies scored on this date. Compare it against the "
        "realised rate marked on the distribution chart below - the two move "
        "independently, which is why the model is scored within a date.",
    ),
    C.Metric(
        "elevated risk",
        f"{int(scored['risk_band'].isin(elevated).sum()):,}",
        help=f"Companies at or above the HIGH cutoff (probability >= {high_cut:.2f}), "
        f"out of {len(scored)} scored.",
    ),
    C.Metric(
        "pledged",
        f"{int((scored['pledge_pct_promoter'] > 0).sum()):,}",
        help=f"Companies with a pledge above zero. A further {n_unknown} have no "
        f"determinable pledge state and are counted in neither direction - the "
        f"field is three-state, not two.",
    ),
)

# ------------------------------------------------------------------- charts
# The confound scatter goes first. sec.12 stars it as the chart that carries the
# project's finding, and it used to open second, behind a click, on the tab that
# is not the default. The labels are the questions the captions already ask -
# more specific than "Risk distribution" and less presumptuous than the old
# "(the confound)", which stated a conclusion in a navigation item where the
# reader has no evidence to judge it.
C.section("Where the risk is coming from")
scatter, histogram = st.tabs(
    ["Is it pledge, or is it volatility?", "Is this quarter risky overall?"]
)

with scatter:
    plot = scored.dropna(subset=["pledge_pct_promoter", "volatility_90d"])
    figure = px.scatter(
        plot,
        x="pledge_pct_promoter",
        y="probability",
        color="volatility_90d",
        hover_name="symbol",
        # Was Turbo. A rainbow ramp is not monotonic in luminance, so the
        # ordering this chart asks the reader to read out of colour is lost in
        # greyscale and under colour blindness - on the one chart sec.12 says
        # exposes the confound. Viridis is perceptually uniform and CVD-safe.
        color_continuous_scale=theme.SEQUENTIAL,
        # Redundant encoding: volatility is now in BOTH hue and marker size, so
        # the argument survives colour loss entirely.
        size="volatility_90d",
        size_max=11,
        opacity=0.55,
        labels={
            "pledge_pct_promoter": "% of promoter stake pledged",
            "probability": "predicted probability",
            "volatility_90d": "volatility 90d",
        },
        title=(
            f"Risk tracks volatility, not pledge "
            f"({len(plot)} of {len(scored)} companies with both values)"
        ),
    )
    # The pledge-to-risk slope, fitted here rather than by px.trendline: that
    # route needs statsmodels, and sec.4.2 pins the dependency set deliberately.
    # A degree-1 polyfit is the same least-squares line.
    if len(plot) >= 2 and plot["pledge_pct_promoter"].nunique() > 1:
        x = plot["pledge_pct_promoter"].to_numpy(dtype=float)
        y = plot["probability"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        ends = np.array([x.min(), x.max()])
        figure.add_trace(
            go.Scatter(
                x=ends,
                y=slope * ends + intercept,
                mode="lines",
                name=f"fit: {slope:+.5f} per pp pledged",
                line={"color": theme.CLARET, "width": 2, "dash": "dash"},
                hoverinfo="name",
            )
        )
    C.chart(
        figure,
        "sec.12 marks this the chart that exposes the confound. Risk rises with colour and "
        "marker size (both volatility) far more cleanly than with position on the x-axis "
        "(pledge). Companies at 0% pledged reach the top of the risk range, and heavily "
        "pledged companies sit at the bottom of it. The fitted line is the pledge-to-risk "
        "relationship on its own - the claim is that it is nearly flat, so it is drawn "
        "rather than asserted.",
    )

with histogram:
    figure = px.histogram(
        scored,
        x="probability",
        color="risk_band",
        color_discrete_map=theme.BAND_COLOURS,
        category_orders={"risk_band": theme.BAND_ORDER},
        labels={"probability": "predicted probability", "risk_band": "band"},
        title=f"Predicted risk across {len(scored)} companies on {date}",
    )
    # Explicit bin edges rather than nbins=30. The band cutoffs are at 0.20, 0.40
    # and 0.60; 30 bins over [0, 1] puts an edge at 0.033-multiples, so bins
    # straddled every cutoff and rendered half in one band colour and half in the
    # next - visual noise at exactly the three decision boundaries. 0.025 divides
    # all three exactly.
    figure.update_traces(xbins={"start": 0.0, "end": 1.0, "size": 0.025})
    for band, upper in sorted(SETTINGS.evaluation.risk_bands.items(), key=lambda kv: kv[1]):
        if upper < 1.0:
            figure.add_vline(x=upper, line_dash="dot", line_color=theme.RULE_2,
                             annotation_text=f"{band} ends", annotation_font_size=10)

    # sec.9.6: the realised rate for THIS date, not the panel-wide constant. The
    # configured expected_event_rate is a single number for the whole study while
    # the realised rate runs from 1.0% to 60.7% across dates, so drawing the
    # constant here and labelling it "base rate" contradicted the caption below
    # it. The embargo quarter has no realised rate, and says so.
    if labelled_date:
        realised = float(scored["label"].mean())
        figure.add_vline(x=realised, line_dash="dash", line_color=theme.INK_2,
                         annotation_text=f"realised rate {realised:.1%}")
    else:
        figure.add_vline(x=SETTINGS.label.expected_event_rate, line_dash="dash",
                         line_color=theme.AMBER,
                         annotation_text=f"panel expected rate "
                                         f"{SETTINGS.label.expected_event_rate:.0%}")
    C.chart(
        figure,
        f"Is this a risky quarter overall, or a calm one? The whole distribution shifts "
        f"quarter to quarter - measured on this panel the event rate runs from "
        f"{event_rate_low:.1%} to {event_rate_high:.1%}, which is why the model is scored "
        f"within a date rather than across all of them. Band colour darkens with severity; "
        f"it carries no information the x-axis does not.",
    )
