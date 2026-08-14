"""Shared rendering for the four pages (PLAN.md sec.14).

Not a widget library. Every function here exists because the same thing was
being done three or four times in slightly different ways, and the differences
were mistakes rather than choices: four identical dead empty-state strings, three
unit conventions inside one seven-column table, the same class of caveat rendered
as ``st.info`` on one page and ``st.warning`` on two others, and one heading
level skipped on one page but not the others.

The contracts are deliberately narrow. :func:`metric_row` refuses a fourth metric
and refuses a metric with no ``help``; :func:`chart` will not render a figure with
no caption. That is the point - the generic-dashboard reflexes this project was
drifting toward are the ones these signatures make awkward to express.

sec.14 forbids HTML, CSS and JavaScript. Nothing here emits any: the whole module
is ``st.*`` calls and Plotly figures.

**Import direction.** Pages import this module and ``app``; ``app`` never imports
this module. The two functions that need a loader import it inside the function
body, which keeps the module importable in any order and is why there is no
circular-import dance anywhere in ``dashboard/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme


# --------------------------------------------------------------------------- #
# structure                                                                    #
# --------------------------------------------------------------------------- #
def section(title: str, note: str | None = None, *, top: bool = False) -> None:
    """One heading level, one rule, on every page.

    The pages disagreed about depth - the scanner ran title -> subheader with no
    header level while validation ran title -> header -> subheader - so sections
    of equal importance rendered at different sizes depending on which page you
    were on. Section boundaries were ASCII comment banners in the source, which
    the reader cannot see.
    """
    if not top:
        st.divider()
    st.subheader(title)
    if note:
        st.caption(note)


def empty_state(message: str, fix: str | None = None) -> None:
    """Say what is missing and name the command that fixes it (sec.10).

    ``check_ready`` in app.py already does this well - "Run ``make init-db &&
    make ingest && make build``" - but the pages never called it and shipped a
    bare "No data for this selection." instead, four times, with no remedy and
    (on the scanner) before any selector had rendered.
    """
    st.warning(message)
    if fix:
        st.caption(fix)


# --------------------------------------------------------------------------- #
# metrics                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Metric:
    """One metric. ``help`` is required, and that is the whole design.

    Every metric on this dashboard is a number whose meaning depends on a
    threshold, a denominator or a convention that lives in config.yaml - "HIGH or
    CRITICAL" is meaningless without the cutoffs, "pledged companies" without its
    treatment of unknown state. A tooltip is the cheapest place to put that, and
    making it mandatory is cheaper than remembering.
    """

    label: str
    value: str
    help: str
    delta: str | None = None
    delta_color: str = "normal"


MAX_METRICS = 3


def metric_row(*metrics: Metric) -> None:
    """At most three metrics, each with a tooltip.

    The four-metric strip is the single most recognisable tell of a generic
    dashboard, and both strips here were padding: "companies scored" is ~300 by
    construction on every date the selector offers. Three is a limit rather than
    a target - two is usually right.

    Raises rather than truncating: a silently dropped metric is worse than a
    failing page, and this only fires at development time.
    """
    if not metrics:
        return
    if len(metrics) > MAX_METRICS:
        raise ValueError(
            f"{len(metrics)} metrics requested; the limit is {MAX_METRICS}. "
            "A four-card strip is decoration - cut the ones that do not vary, or "
            "move the cohort summary below the answer it is describing."
        )
    missing = [m.label for m in metrics if not m.help]
    if missing:
        raise ValueError(f"metric(s) {missing} have no help text; every threshold needs one")

    for column, metric in zip(st.columns(len(metrics)), metrics, strict=True):
        column.metric(
            metric.label,
            metric.value,
            delta=metric.delta,
            delta_color=metric.delta_color,
            help=metric.help,
        )


# --------------------------------------------------------------------------- #
# tables                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Column:
    """A column's label, format, unit and tooltip in one place.

    Replaces ``pandas.Styler.format``, which gives formatting and nothing else -
    which is why the scanner's units ended up incoherent: a percentage with no
    ``%``, a unit stranded in a header, and a volatility figure with no unit at
    all, so 0.42 could not be read as 42% or anything else.
    """

    label: str
    format: str | None = None
    help: str | None = None
    width: str | None = None
    kind: str = "number"
    extra: dict[str, Any] = field(default_factory=dict)

    def build(self):
        common = {"label": self.label, "help": self.help, "width": self.width}
        if self.kind == "progress":
            return st.column_config.ProgressColumn(
                **common, format=self.format, **self.extra
            )
        if self.kind == "text":
            return st.column_config.TextColumn(**common, **self.extra)
        return st.column_config.NumberColumn(**common, format=self.format, **self.extra)


def table(frame: pd.DataFrame, columns: dict[str, Column], **kwargs):
    """``st.dataframe`` with a real column spec.

    Only the columns named in ``columns`` are shown, in that order, which also
    makes the page's column order explicit instead of inherited from whatever
    the query happened to return.
    """
    present = [name for name in columns if name in frame.columns]
    return st.dataframe(
        frame[present],
        column_config={name: columns[name].build() for name in present},
        use_container_width=True,
        hide_index=True,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# charts                                                                       #
# --------------------------------------------------------------------------- #
def chart(figure: go.Figure, caption: str) -> None:
    """A figure and the sentence saying what to conclude from it.

    The captions on this dashboard are one of its genuinely good habits - they
    state the takeaway in words rather than restating the axis labels - and they
    are also the only accessible description of a Plotly canvas. Requiring one
    keeps both true as charts are added.
    """
    if not caption:
        raise ValueError("every chart states its takeaway in words; caption is required")
    st.plotly_chart(figure, use_container_width=True)
    st.caption(caption)


def forest(detectability: pd.DataFrame, *, height: int | None = None) -> go.Figure:
    """Every measured difference as an estimate with an interval.

    The detectability table is the best analysis in this project and it was
    rendered as a spreadsheet: ``delta``, ``ci_low`` and ``ci_high`` as three
    adjacent columns of four-decimal floats, which asks the reader to do interval
    arithmetic in their head, twenty-four times.

    A forest plot is the form built for exactly this - a dot at the estimate,
    whiskers at its interval, and a vertical line of no effect - and it makes the
    question the reader actually has ("does this cross zero?") a matter of
    looking rather than of comparing signs. It is the standard way a null is
    reported in the trial literature, which is the register this project's result
    belongs in.

    Colour is redundant with position: a crossing interval is visible whether or
    not the verdict colour resolves.
    """
    if detectability.empty:
        return go.Figure()

    frame = detectability.dropna(subset=["delta", "ci_low", "ci_high"]).copy()
    if frame.empty:
        return go.Figure()

    frame["label"] = frame["experiment"] + "  ·  " + frame["model"]
    frame = frame.sort_values("delta", ascending=True).reset_index(drop=True)

    # From config, not hardcoded: power.confidence_level drives the interval the
    # bars actually show, so a label reading "95%" would silently lie if it moved.
    from app import SETTINGS

    level = f"{SETTINGS.power.confidence_level:.0%}"

    figure = go.Figure()
    for verdict, block in frame.groupby("verdict", sort=False):
        figure.add_trace(
            go.Scatter(
                x=block["delta"],
                y=block["label"],
                mode="markers",
                name=str(verdict),
                marker={
                    "size": 9,
                    "color": theme.VERDICT_COLOURS.get(str(verdict), theme.NEUTRAL),
                    "line": {"width": 0},
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": (block["ci_high"] - block["delta"]).tolist(),
                    "arrayminus": (block["delta"] - block["ci_low"]).tolist(),
                    "color": theme.RULE_2,
                    "thickness": 1.4,
                    "width": 4,
                },
                customdata=block[["vs", "ci_low", "ci_high", "dates"]].to_numpy(),
                hovertemplate=(
                    "%{y}<br>vs %{customdata[0]}"
                    "<br>delta %{x:+.4f}"
                    f"<br>{level} CI "
                    "[%{customdata[1]:+.4f}, %{customdata[2]:+.4f}]"
                    "<br>%{customdata[3]} dates<extra></extra>"
                ),
            )
        )

    figure.add_vline(
        x=0,
        line_color=theme.INK_2,
        line_width=1.5,
        annotation_text="no effect",
        annotation_position="top",
    )
    figure.update_layout(
        title=f"Every comparison as an estimate, with its {level} interval",
        xaxis_title="difference in within-quarter AUC against the experiment's own baseline",
        yaxis_title=None,
        height=height or max(320, 26 * len(frame) + 130),
        margin={"l": 8, "r": 8, "t": 64, "b": 8},
    )
    figure.update_yaxes(tickfont={"family": theme.MONO, "size": 11})
    return figure


# --------------------------------------------------------------------------- #
# data quality                                                                 #
# --------------------------------------------------------------------------- #
def data_quality(frame: pd.DataFrame, *, scope: str) -> None:
    """Render the service's ``data_warnings``, and say so when there are none.

    Two rules, both learned from the bug this replaced. Select the column the
    service built rather than slicing the full warnings list by position - that
    slice dropped 653 warnings across the 19 labelled dates. And render an
    explicit negative when the set is empty, because a component that only
    appears when it has something to say makes its own absence ambiguous: the
    reader cannot tell "nothing is wrong" from "the check did not run", and for a
    long time it was the latter.
    """
    if "data_warnings" not in frame.columns:
        return
    flagged = frame[frame["data_warnings"].map(len) > 0]
    if flagged.empty:
        st.caption(f"No data warnings {scope}.")
        return
    with st.expander(f"{len(flagged)} of {scope} carry data warnings"):
        for row in flagged.itertuples(index=False):
            st.write(f"**{row.symbol}** - {'; '.join(row.data_warnings)}")


# --------------------------------------------------------------------------- #
# the finding                                                                  #
# --------------------------------------------------------------------------- #
def finding_banner(*, link: bool = True) -> None:
    """The project's headline result, on every page that shows a number.

    The scanner is the page most easily mistaken for an actionable stock screen -
    a ranked watchlist of companies by crash probability - and the one statement
    that this ranking is essentially a volatility ranking sat in its second tab,
    below the fold, behind a click. A reader who deep-linked to it never met the
    finding at all.

    Loaders are imported here rather than at module scope so that ``app`` and
    this module can be imported in either order.
    """
    from app import SETTINGS, _headline, load_metrics, load_model_runs

    try:
        delta, _ = _headline(load_metrics(), load_model_runs())
    except Exception:  # noqa: BLE001 - a missing headline must not take a page down
        return
    if delta is None:
        return

    verdict = "adds nothing" if delta <= 0 else "adds signal"
    st.info(
        f"**The finding: pledge trajectory {verdict}.** "
        f"`{SETTINGS.headline.experiment}` minus `{SETTINGS.headline.baseline}` on "
        f"within-quarter AUC is **{delta:+.4f}** (median across models). Rankings on "
        f"this dashboard are driven by volatility and size, not by pledge data."
    )
    if link:
        # st.page_link resolves its path against the ENTRYPOINT script, so the
        # same string that is correct under `streamlit run dashboard/app.py`
        # raises StreamlitPageNotFoundError whenever a page is executed as its
        # own entrypoint - which is how the smoke tests run them, and how anyone
        # debugging a single page would run it too.
        #
        # sec.10 says a failure is a message, never a traceback. A navigation
        # link is the least important element on the page and must not be able
        # to take the page down with it, so the caveat above always renders and
        # the link degrades to a caption.
        try:
            st.page_link(
                "pages/3_Model_Validation.py",
                label="See the intervals behind that number",
                icon=":material/query_stats:",
            )
        except Exception:  # noqa: BLE001 - navigation is decoration; the finding is not
            st.caption("See **Model Validation** for the intervals behind that number.")


__all__ = [
    "MAX_METRICS",
    "Column",
    "Metric",
    "chart",
    "data_quality",
    "empty_state",
    "finding_banner",
    "forest",
    "metric_row",
    "section",
    "table",
]
