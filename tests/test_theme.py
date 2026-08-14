"""The dashboard palette, and the one place it is duplicated (PLAN.md sec.14).

sec.14 forbids HTML, CSS and JavaScript, so the app's colours have to come from
two places that cannot import each other: ``.streamlit/config.toml``, which
Streamlit reads before any Python runs, and ``dashboard/theme.py``, which the
Plotly template and the SHAP figure read at render time.

That duplication is deliberate and it is also the obvious thing to get wrong -
change one, forget the other, and the charts sit on a ground a shade away from
the page they are drawn on, which reads as a rendering bug rather than a design
choice. These tests make the two files agree by assertion.

The encoding rules are pinned here too. Both were violated before this module
existed, on the two charts that carry the project's argument, and neither
violation is the kind that shows up as an error.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = PROJECT_ROOT / "dashboard"
CONFIG = PROJECT_ROOT / ".streamlit" / "config.toml"

if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(DASHBOARD))

import theme  # noqa: E402


@pytest.fixture(scope="module")
def toml_theme() -> dict:
    if not CONFIG.exists():
        pytest.skip("no .streamlit/config.toml")
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))["theme"]


# --------------------------------------------------------------------------- #
# the duplication, checked                                                     #
# --------------------------------------------------------------------------- #
def test_the_toml_and_the_python_palette_agree(toml_theme):
    """One shade of drift makes every chart look like it failed to load."""
    assert toml_theme["primaryColor"].upper() == theme.PETROL.upper()
    assert toml_theme["backgroundColor"].upper() == theme.GROUND.upper()
    assert toml_theme["secondaryBackgroundColor"].upper() == theme.SURFACE.upper()
    assert toml_theme["textColor"].upper() == theme.INK.upper()


def test_charts_are_drawn_on_the_page_background(toml_theme):
    """A chart on its own ground is a visible rectangle, whatever the colour."""
    theme.register()
    import plotly.io as pio

    layout = pio.templates[theme.TEMPLATE].layout
    assert layout.paper_bgcolor.upper() == toml_theme["backgroundColor"].upper()
    assert layout.plot_bgcolor.upper() == toml_theme["backgroundColor"].upper()


def test_the_shap_figure_shares_that_background():
    """The one matplotlib figure in the app must not arrive on default white."""
    rc = theme.matplotlib_rc()
    assert rc["figure.facecolor"] == theme.GROUND
    assert rc["axes.facecolor"] == theme.GROUND
    assert rc["savefig.facecolor"] == theme.GROUND
    assert rc["figure.dpi"] >= 150, "st.pyplot ships a raster; 100 dpi reads soft"


def test_registering_the_template_is_idempotent():
    """Streamlit re-executes the script on every interaction."""
    import plotly.io as pio

    theme.register()
    theme.register()
    assert pio.templates.default == theme.TEMPLATE
    assert len(pio.templates[theme.TEMPLATE].layout.colorway) == len(theme.COLORWAY)


# --------------------------------------------------------------------------- #
# encoding rules - both of these were broken                                   #
# --------------------------------------------------------------------------- #
def test_no_rainbow_colour_scales_anywhere_in_the_dashboard():
    """Turbo carried volatility on the chart sec.12 says exposes the confound.

    A rainbow ramp is not monotonic in luminance, so the ordering the caption
    told the reader to read out of hue was unrecoverable in greyscale and under
    the common colour-vision deficiencies.
    """
    banned = ("turbo", "jet", "rainbow", "hsv")
    offenders = []
    for path in sorted(DASHBOARD.rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for scale in banned:
            if re.search(rf'color_continuous_scale\s*=\s*["\']{scale}', text):
                offenders.append(f"{path.name}: {scale}")
    assert not offenders, f"perceptually non-uniform colour scales: {offenders}"


def test_the_sequential_scale_is_perceptually_uniform():
    assert theme.SEQUENTIAL in {"Viridis", "Cividis", "Magma", "Inferno", "Plasma"}


def test_risk_bands_are_encoded_as_an_ordinal_scale():
    """LOW < MEDIUM < HIGH < CRITICAL is ordered; a qualitative palette is not.

    Lightness has to fall monotonically, because that is the channel that
    survives greyscale and every form of colour blindness. Before this, the
    bands took Plotly's categorical colours and LOW could render red.
    """
    def luminance(hex_colour: str) -> float:
        r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    ordered = [theme.BAND_COLOURS[band] for band in theme.BAND_ORDER]
    lums = [luminance(colour) for colour in ordered]
    assert lums == sorted(lums, reverse=True), (
        f"band lightness must fall with severity, got {list(zip(theme.BAND_ORDER, lums, strict=False))}"
    )


def test_every_configured_band_has_a_colour():
    """Renaming a band in config.yaml must not silently drop it from the chart."""
    from config import get_settings

    configured = set(get_settings().evaluation.risk_bands)
    assert configured == set(theme.BAND_COLOURS)
    assert configured == set(theme.BAND_ORDER)


def test_the_accent_is_not_a_semantic_colour():
    """A selected row must never be mistakable for a bad result."""
    assert theme.PETROL not in {theme.CLARET, theme.AMBER, theme.MOSS}
    assert theme.PETROL not in theme.VERDICT_COLOURS.values()


def test_every_power_verdict_has_a_colour():
    """power.py emits four; a missing one renders as an untraceable default."""
    from pledgecast.evaluation import power

    for verdict in (power.ZERO, power.POSITIVE, power.NEGATIVE, power.UNKNOWN):
        assert verdict in theme.VERDICT_COLOURS


def test_a_null_result_is_not_coloured_as_a_failure():
    """sec.2.2: a null is a legitimate result. ZERO must read neutral."""
    from pledgecast.evaluation import power

    assert theme.VERDICT_COLOURS[power.ZERO] == theme.NEUTRAL
    assert theme.VERDICT_COLOURS[power.ZERO] != theme.CLARET
