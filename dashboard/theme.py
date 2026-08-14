"""One palette and one chart template for the whole dashboard (PLAN.md sec.14).

Before this module there was no chart system at all: eleven Plotly figures across
three pages each inherited Plotly's defaults, so there was no colourway, no type
scale, no margin rule and no hover convention - and two figures encoded meaning
in colour badly enough to undermine the argument they were drawn to make.

Everything here is Python. sec.14 forbids HTML, CSS and JavaScript; a registered
``plotly.io`` template is none of those, and it is the only way to give eleven
figures a shared visual language without repeating layout arguments in every
call site.

**The two encoding defects this fixes.**

*Turbo on the confound scatter.* sec.12 marks that chart as the one that
"visually exposes the confound", and it carried volatility - the variable the
whole argument rests on - in hue alone, on a rainbow ramp. Turbo is not monotonic
in luminance: it is unreadable in greyscale and scrambles under the common forms
of colour blindness, so roughly one male reader in twelve could not recover the
ordering the caption told them to read. :data:`SEQUENTIAL` replaces it with
Viridis, which is perceptually uniform and colour-vision-deficiency safe.

*A qualitative palette on ordinal risk bands.* ``LOW < MEDIUM < HIGH <
CRITICAL`` is an ordered scale, but ``px.histogram(color="risk_band")`` assigned
Plotly's categorical colours, so LOW could render red and HIGH blue - severity
carried no visual order at all. :data:`BAND_COLOURS` is a single-hue ramp that
darkens monotonically toward claret, so the order survives both greyscale and
colour blindness because lightness carries it.

**Why these colours.** The qualitative colourway is derived from the Okabe-Ito
palette, which was designed for colour-vision deficiency, with petrol leading so
the first series matches the app's accent and yellow dropped because it fails on
a light ground. The accent is deliberately far from both semantic colours: amber
means "discount this number" and claret means "measurably worse", so an
interactive element must not be able to be mistaken for either.

The literal values are duplicated in ``.streamlit/config.toml``, which Streamlit
reads before any Python runs and which has no slot for semantic colours. That
duplication is the reason this module exists rather than a dict inside app.py -
one obvious place to change them both.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------- #
# surfaces and ink - must match .streamlit/config.toml                         #
# --------------------------------------------------------------------------- #
GROUND = "#F7FAFA"
SURFACE = "#EDF2F2"
INK = "#0F1617"
INK_2 = "#3C4A4C"
INK_3 = "#657476"
RULE = "#D6E0DF"
RULE_2 = "#C0CECC"

# --------------------------------------------------------------------------- #
# accent and semantics - never the same colour                                 #
# --------------------------------------------------------------------------- #
PETROL = "#0B6E6E"  # accent: interactive, active model, the primary series
CLARET = "#96143C"  # measurably worse; realised drawdown windows
AMBER = "#9A5B08"   # discount this number: stale, imputed, embargo
MOSS = "#3E6115"    # measurably better
NEUTRAL = INK_3     # indistinguishable from zero - a null is not a failure

#: Verdict -> colour. :mod:`pledgecast.evaluation.power` emits these four.
VERDICT_COLOURS = {
    "ZERO": NEUTRAL,
    "POSITIVE": MOSS,
    "NEGATIVE": CLARET,
    "UNKNOWN": RULE_2,
}

#: Ordinal, not categorical. Lightness falls monotonically (77 -> 68 -> 52 -> 30)
#: so the ordering survives greyscale printing and every form of colour blindness.
BAND_COLOURS = {
    "LOW": "#B8C4C3",
    "MEDIUM": "#C99BA4",
    "HIGH": "#B85E76",
    "CRITICAL": "#8C1236",
}
BAND_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

#: Perceptually uniform and CVD-safe. Replaces Turbo everywhere.
SEQUENTIAL = "Viridis"

#: Okabe-Ito, reordered to lead with the accent, yellow dropped (illegible on a
#: light ground). Eight series - enough for every experiment in config.yaml.
COLORWAY = [
    PETROL,
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    INK_2,
]

SANS = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
    '"Helvetica Neue", Arial, sans-serif'
)
MONO = 'ui-monospace, "Cascadia Mono", "SF Mono", Consolas, monospace'

TEMPLATE = "pledgecast"


def _axis() -> dict:
    return {
        "gridcolor": RULE,
        "linecolor": RULE_2,
        "zerolinecolor": RULE_2,
        "zerolinewidth": 1,
        "ticks": "outside",
        "ticklen": 4,
        "tickcolor": RULE_2,
        "tickfont": {"size": 12, "color": INK_2},
        "title": {"font": {"size": 12.5, "color": INK_2}},
        "automargin": True,
    }


def register() -> None:
    """Register the template and make it the default for every figure.

    Idempotent - Streamlit re-executes the whole script on each interaction, so
    this runs on every rerun and must not accumulate state.
    """
    pio.templates[TEMPLATE] = go.layout.Template(
        layout={
            "font": {"family": SANS, "size": 13, "color": INK},
            # Left-aligned titles read as headings rather than as captions, and
            # the FT convention this borrows puts the FINDING in the title.
            "title": {"font": {"size": 15.5, "color": INK}, "x": 0, "xanchor": "left"},
            "paper_bgcolor": GROUND,
            "plot_bgcolor": GROUND,
            "colorway": COLORWAY,
            "xaxis": _axis(),
            "yaxis": _axis(),
            "colorscale": {"sequential": SEQUENTIAL, "diverging": "RdBu"},
            "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": "outside", "len": 0.8}},
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0,
                "title": {"text": ""},
                "font": {"size": 12},
            },
            # Monospaced hover: the numbers line up, and these are all numbers.
            "hoverlabel": {
                "font": {"family": MONO, "size": 12},
                "bgcolor": INK,
                "bordercolor": INK,
            },
            "margin": {"l": 8, "r": 8, "t": 52, "b": 8},
            "hovermode": "closest",
        }
    )
    pio.templates.default = TEMPLATE


def matplotlib_rc() -> dict:
    """rcParams that put a matplotlib figure on the same ground as the Plotly ones.

    The SHAP waterfall (sec.12 specifies matplotlib for it) is the only figure in
    the app that Plotly does not draw. Left alone it renders on matplotlib's
    default white with matplotlib's default type, which reads as a foreign object
    dropped into the page - on the explainability tab, which is the one place a
    reader is being asked to trust the model's reasoning.

    Only the frame is themed. The bars keep SHAP's own red/blue, which is a
    convention readers arrive already knowing and which this project has no
    reason to reinvent.
    """
    return {
        "figure.facecolor": GROUND,
        "axes.facecolor": GROUND,
        "savefig.facecolor": GROUND,
        "savefig.transparent": False,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": RULE_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "grid.color": RULE,
        "font.size": 11,
        # HiDPI: st.pyplot ships a raster, and the default 100 dpi is visibly
        # soft next to Plotly's vector output on the same page.
        "figure.dpi": 150,
    }


__all__ = [
    "AMBER",
    "BAND_COLOURS",
    "BAND_ORDER",
    "CLARET",
    "COLORWAY",
    "GROUND",
    "INK",
    "INK_2",
    "INK_3",
    "MONO",
    "MOSS",
    "NEUTRAL",
    "PETROL",
    "RULE",
    "RULE_2",
    "SANS",
    "SEQUENTIAL",
    "SURFACE",
    "TEMPLATE",
    "VERDICT_COLOURS",
    "matplotlib_rc",
    "register",
]
