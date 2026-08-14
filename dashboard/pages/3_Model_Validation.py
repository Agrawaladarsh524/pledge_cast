"""Page 3 - Model Validation (PLAN.md sec.14.1).

    "Model comparison table · ROC/PR curves by experiment · quintile bars
     (model vs. null) · separation ratio over time · active model metadata ·
     a plainly-worded limitations box (20 quarters, survivorship bias,
     volatility confound)."

sec.12 marks the quintile bars as **THE HEADLINE RESULT**, so they open the page
rather than closing it. sec.9.9 requires the null model's bars beside the
model's own: "if pledge-aware quintiles separate no better than volatility-only
quintiles, the honest headline is 'pledge trajectory adds no incremental early
warning once volatility is accounted for' - publish that."

The limitations box is not a disclaimer added at the end. sec.2.5 asks for these
to be stated openly, and a validation page that showed only the favourable
numbers would be the exact failure this project is designed to avoid.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

import _bootstrap  # noqa: F401  - must precede config/pledgecast imports
import components as C
from app import (
    SETTINGS,
    guard,
    load_active_run,
    load_backtest,
    load_detectability,
    load_metrics,
    load_model_runs,
    load_panel,
    load_predictions,
    sidebar,
)

st.title("Model Validation")
st.caption("How far the numbers on the other pages can be trusted.")

runs = guard(load_model_runs)
metrics = guard(load_metrics)
panel = guard(load_panel, True)
active = guard(load_active_run)

if runs is None or runs.empty or metrics is None or metrics.empty:
    st.warning("No trained models yet - run `make train`.")
    st.stop()

sidebar()

aggregate = metrics[metrics["fold"] == -1].merge(
    runs[["run_id", "model_name", "experiment"]], on="run_id"
)

# --------------------------------------------------------------- the headline
st.header("Economic backtest - the headline result")
backtest_rows = guard(load_backtest)

if backtest_rows is None or backtest_rows.empty:
    st.info("No backtest results yet - run `make evaluate`.")
else:
    labelled = backtest_rows.merge(runs[["run_id", "model_name", "experiment"]], on="run_id")
    pooled = (
        labelled.groupby(["experiment", "quintile"])
        .agg(n_companies=("n_companies", "sum"), n_events=("n_events", "sum"))
        .reset_index()
    )
    pooled["event_rate"] = pooled["n_events"] / pooled["n_companies"]

    figure = px.bar(
        pooled,
        x="quintile",
        y="event_rate",
        color="experiment",
        barmode="group",
        labels={"event_rate": "realised event rate", "quintile": "risk quintile (5 = riskiest)"},
        title="Realised event rate by predicted-risk quintile, model against the null",
    )
    figure.add_hline(
        y=float(panel["label"].mean()),
        line_dash="dash",
        annotation_text="base rate",
    )
    st.plotly_chart(figure, use_container_width=True)

    summary = []
    for experiment, block in pooled.groupby("experiment"):
        rates = block.set_index("quintile")["event_rate"]
        top = rates.get(SETTINGS.evaluation.n_quintiles)
        bottom = rates.get(1)
        summary.append(
            {
                "experiment": experiment,
                "Q1 (safest)": bottom,
                "Q5 (riskiest)": top,
                "Q5 - Q1": None if top is None or bottom is None else top - bottom,
                "Q5 / Q1": None if not bottom else top / bottom,
            }
        )
    summary_frame = pd.DataFrame(summary)
    st.dataframe(
        summary_frame.style.format(
            {c: "{:.3f}" for c in summary_frame.columns if c != "experiment"}, na_rep="n/a"
        ),
        use_container_width=True,
        hide_index=True,
    )

    # sec.9.9: "Report per quarter, not just pooled - the spread shows whether
    # the edge is stable or came from one lucky correction."
    st.subheader("Separation over time")
    per_date = []
    for (experiment, date), block in labelled.groupby(["experiment", "observation_date"]):
        rates = block.set_index("quintile")["event_rate"]
        top, bottom = rates.get(SETTINGS.evaluation.n_quintiles), rates.get(1)
        if top is None or bottom is None:
            continue
        per_date.append(
            {"experiment": experiment, "observation_date": date, "Q5 - Q1": top - bottom}
        )
    spread = pd.DataFrame(per_date)
    if not spread.empty:
        figure = px.line(
            spread.sort_values("observation_date"),
            x="observation_date",
            y="Q5 - Q1",
            color="experiment",
            markers=True,
            title="Quintile separation per quarter - stable edge, or one lucky correction?",
        )
        figure.add_hline(y=0, line_dash="dash")
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "A single pooled number would hide this. The separation swings from roughly zero "
            "to over 0.5 across quarters, and the two quarters where it goes negative are the "
            "ones with almost no events at all."
        )

# ---------------------------------------------------------- detectability
st.header("Is the difference real?")
st.caption(
    "Every comparison below with a confidence interval, so a delta can be read as a "
    "measurement rather than a ranking. The interval is a block bootstrap over "
    "observation dates - companies inside one quarter share a market regime, so "
    "resampling rows instead of dates would report an interval several times too narrow."
)
detectability = guard(load_detectability)
if detectability is None or detectability.empty:
    st.info("No paired runs to compare yet.")
else:
    verdicts = detectability["verdict"].value_counts().to_dict()
    C.metric_row(
        C.Metric(
            "Indistinguishable from zero",
            f"{verdicts.get('ZERO', 0)}",
            help="The interval contains zero, so the sign of the delta carries no "
            "information. sec.2.2: a null is a legitimate result.",
        ),
        C.Metric(
            "Measurably better",
            f"{verdicts.get('POSITIVE', 0)}",
            help="Interval entirely above zero. This is the count the study was "
            "designed to be able to return, and did not.",
        ),
        C.Metric(
            "Measurably worse",
            f"{verdicts.get('NEGATIVE', 0)}",
            help="Interval entirely below zero - adding these features actively "
            "cost within-quarter AUC against the experiment's own baseline.",
        ),
    )

    # The forest plot leads; the numbers follow it. Three adjacent columns of
    # four-decimal floats asked the reader to do interval arithmetic in their
    # head twenty-four times, and the one question they actually have - does this
    # cross zero? - is a matter of looking once the interval is drawn.
    C.chart(
        C.forest(detectability),
        "Each dot is a measured difference; each bar is its confidence interval. "
        "A bar crossing the vertical line is a measurement of nothing - not a small "
        "effect. Nothing lands entirely to the right of the line, which is the result.",
    )

    with st.expander("The same numbers as a table"):
        C.table(
            detectability,
            {
                "experiment": C.Column("experiment", kind="text",
                                       help="The treatment being measured."),
                "vs": C.Column("against", kind="text",
                               help="Its own baseline. Comparisons never cross populations."),
                "model": C.Column("model", kind="text"),
                "delta": C.Column("delta", format="%+.4f",
                                  help="Treatment minus baseline on within-quarter AUC, "
                                       "paired by observation date."),
                "ci_low": C.Column("CI low", format="%+.4f",
                                   help="Bootstrap-t lower bound over observation dates."),
                "ci_high": C.Column("CI high", format="%+.4f",
                                    help="Bootstrap-t upper bound over observation dates."),
                "min_detectable": C.Column(
                    "min detectable", format="%.4f",
                    help="Half the interval width - the smallest difference this "
                         "design could have called non-zero."),
                "dates": C.Column("dates", format="%d",
                                  help="Observation dates contributing to the pairing."),
                "verdict": C.Column("verdict", kind="text"),
            },
        )
    st.caption(
        "**How to read a ZERO.** The interval contains zero, so the sign of the delta "
        "carries no information - a delta of -0.018 with an interval of +/-0.033 is a "
        "measurement of nothing, not a small negative effect. `min_detectable` is the "
        "smallest difference this design could have called non-zero; anything smaller was "
        "never findable with 19 observation dates, and no amount of extra modelling "
        "would change that."
    )

# ------------------------------------------------------------- comparison
# ------------------------------------------------------------- limitations
# Moved up, and out of a single st.warning. sec.2.5 asks for these to be stated
# openly, and 300 words of the most important prose in the project was formatted
# as one amber block at the bottom of a six-header page - which is the format
# readers are trained to skip. One bordered container per limitation, each with
# its own heading and its own supporting number, placed where the reader has just
# seen the result and not yet seen the supporting detail.
st.header("What to distrust")
st.caption("Read these before using any number on this dashboard.")

_LIMITATIONS = [
    (
        "Twenty quarters",
        f"The NSE XBRL archive begins {SETTINGS.window.first_quarter_end}. That gives 20 "
        "quarters, 19 of them labelled, and 11 walk-forward test folds. Per-fold AUC ranges "
        "from roughly 0.39 to 0.76 - the mean is a summary of a very wide spread, not a "
        "stable estimate.",
    ),
    (
        "Survivorship bias",
        "The universe is today's NIFTY 500 constituents. Companies delisted or removed from "
        "the index during the study window are absent, and those are disproportionately the "
        "ones that failed - so the realised event rate here is a floor.",
    ),
    (
        "The volatility confound",
        "Pledged companies tend to be leveraged smallcaps. A model given only volatility and "
        "turnover separates the target nearly as well as the full 13-feature model. This is "
        "the reason the study exists and the reason its headline result is a difference "
        "between two models rather than one model's score.",
    ),
    (
        "The pledge barely moves",
        "Measured on this panel, the quarterly pledge percentage is unchanged in 90.5% of "
        "company-quarters. Four of the eight pledge features are zero for nine rows in ten. "
        "Quarterly disclosure may simply be too slow-moving to carry early warning, which is "
        "a fact about the data rather than about the model.",
    ),
    (
        "Not investment advice",
        "This is a research artefact. It reports a null result about a data source, not a "
        "recommendation about any company.",
    ),
]

for index, (heading, body) in enumerate(_LIMITATIONS, start=1):
    with st.container(border=True):
        st.markdown(f"**{index}. {heading}**")
        st.write(body)

st.header("Model comparison")
primary = aggregate[aggregate["metric_name"] == SETTINGS.evaluation.primary_metric]
wide = aggregate.pivot_table(
    index=["experiment", "model_name"],
    columns="metric_name",
    values="metric_value",
    aggfunc="last",
)
keep = [
    c
    for c in (
        "within_quarter_auc",
        "within_quarter_auc_std",
        "within_quarter_auc_min",
        "within_quarter_auc_max",
        "pooled_auc",
        "pr_auc",
        "brier",
        "brier_skill_score",
    )
    if c in wide.columns
]
st.dataframe(wide[keep].style.format("{:.4f}", na_rep="n/a"), use_container_width=True)
st.caption(
    "Within-quarter AUC is the primary metric (sec.9.6) - computed per observation date and "
    "then averaged, which is the only form immune to the market-timing confound. Pooled AUC is "
    "shown beside it precisely because it looks better and means less. Accuracy is absent on "
    "purpose: always predicting 'no event' scores 77% here."
)

if not primary.empty:
    figure = px.bar(
        primary,
        x="experiment",
        y="metric_value",
        color="model_name",
        barmode="group",
        labels={"metric_value": SETTINGS.evaluation.primary_metric},
        title="Within-quarter AUC by experiment and model",
    )
    figure.add_hline(y=0.5, line_dash="dash", annotation_text="chance")
    st.plotly_chart(figure, use_container_width=True)

# ------------------------------------------------------------ ROC / PR
st.header("ROC and precision-recall by experiment")
if active is None:
    st.info("No active model.")
else:
    curves_roc, curves_pr = go.Figure(), go.Figure()
    drawn = 0
    for experiment in SETTINGS.experiments:
        match = runs[
            (runs["model_name"] == active["model_name"]) & (runs["experiment"] == experiment)
        ]
        if match.empty:
            continue
        predictions = guard(load_predictions, match.iloc[0]["run_id"], "backtest")
        if predictions is None or predictions.empty:
            continue
        joined = predictions.merge(
            panel[["symbol", "observation_date", "label"]],
            on=["symbol", "observation_date"],
            how="inner",
        ).dropna(subset=["label", "probability"])
        if joined.empty or joined["label"].nunique() < 2:
            continue

        fpr, tpr, _ = roc_curve(joined["label"], joined["probability"])
        curves_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=experiment))
        precision, recall, _ = precision_recall_curve(joined["label"], joined["probability"])
        curves_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=experiment))
        drawn += 1

    if drawn:
        curves_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line={"dash": "dash"})
        )
        curves_roc.update_layout(
            title=f"ROC, pooled out-of-fold ({active['model_name']})",
            xaxis_title="false positive rate",
            yaxis_title="true positive rate",
        )
        base = float(panel["label"].mean())
        curves_pr.add_hline(y=base, line_dash="dash", annotation_text=f"base rate {base:.1%}")
        curves_pr.update_layout(
            title=f"Precision-recall ({active['model_name']})",
            xaxis_title="recall",
            yaxis_title="precision",
        )
        left, right = st.columns(2)
        left.plotly_chart(curves_roc, use_container_width=True)
        right.plotly_chart(curves_pr, use_container_width=True)
        st.caption(
            "These are POOLED across dates, so they carry the market-timing confound the "
            "primary metric removes. They are shown because sec.12 asks for them and because "
            "the comparison between experiments is still informative - not as the headline."
        )
    else:
        st.info("No out-of-fold predictions stored yet - run `make train`.")

# ---------------------------------------------------------------- metadata
st.header("Active model")
if active is not None:
    left, right = st.columns(2)
    with left:
        st.write(f"**run_id** `{active['run_id']}`")
        st.write(f"**model** {active['model_name']} · **experiment** {active['experiment']}")
        st.write(f"**trained** {active['created_at']}")
        st.write(f"**seed** {active['random_seed']} · **folds** {active['n_folds']}")
        st.write(f"**rows** {active['n_train_rows']:,}")
    with right:
        st.write("**features**")
        st.write(", ".join(active["feature_list"]))
    with st.expander("Hyperparameters"):
        st.json(active["hyperparams"])

# ------------------------------------------------------------- limitations
