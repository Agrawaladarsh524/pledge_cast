"""Dashboard smoke tests - the safety net for the sec.14 UI (PLAN.md sec.15).

sec.14 constrains the dashboard hard ("you will not write a single line of HTML,
CSS, or JavaScript") and sec.10 constrains its failure behaviour ("never a
traceback"). Neither was enforced by anything before this file: the pages were
only ever exercised by a human opening a browser, so a rename in the repository
or a typo in a page body would be discovered in a demo.

``streamlit.testing.v1.AppTest`` runs a page headless, in-process, and exposes
the element tree it produced. That gives two guarantees worth having:

1. **Every page renders.** Whatever the state of the database, no page may raise.
   ``AppTest.exception`` is the sec.10 rule expressed as an assertion - a page
   that shows "run `make ingest`" passes, a page that shows a stack trace fails.
2. **The constraint holds.** A static scan for ``unsafe_allow_html`` fails the
   suite the moment anyone reaches for it, including a future me.

The data-dependent assertions are separated and skipped when the panel is empty,
so the suite is honest on a clean checkout and in CI (where ``DB_PATH`` points at
a database that deliberately does not exist) while still being strict locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="the dashboard is optional at runtime")

from streamlit.testing.v1 import AppTest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = PROJECT_ROOT / "dashboard"

# Streamlit puts the main script's folder on sys.path, which is how `import
# _bootstrap` and `from app import ...` resolve inside a page. AppTest executes
# the script in THIS process, so the same path has to be arranged by hand or
# every page fails at its first import for a reason that has nothing to do with
# the page.
if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(DASHBOARD))

HOME = DASHBOARD / "app.py"
# Filenames, and therefore sidebar order, are exactly as sec.14.1 and the sec.16
# file tree specify. Reordering them so the validation page came first would put
# the evidence ahead of the watchlist, which reads better - but the repository
# matching docs/PLAN.md is worth more than the improvement, and the caveat now
# travels with the watchlist anyway via components.finding_banner.
PAGES = {
    "home": HOME,
    "scanner": DASHBOARD / "pages" / "1_Risk_Scanner.py",
    "company": DASHBOARD / "pages" / "2_Company_Investigation.py",
    "validation": DASHBOARD / "pages" / "3_Model_Validation.py",
}

# Scoring 300 rows through a fitted pipeline on a cold cache is far past
# AppTest's 3-second default, and a timeout here would read as a page failure.
TIMEOUT = 180


def run(path: Path) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    app.run()
    return app


@pytest.fixture(scope="module")
def home() -> AppTest:
    return run(HOME)


@pytest.fixture(scope="module")
def has_data() -> bool:
    """True when the database holds a panel and an active model.

    Both data-dependent assertions below need this, and asking the repository
    directly is cheaper and clearer than inferring it from rendered elements.
    """
    try:
        from config import get_settings
        from pledgecast.db import repository as repo
        from pledgecast.db.connection import get_connection

        settings = get_settings()
        with get_connection(settings=settings) as conn:
            return not repo.load_panel(conn).empty and repo.get_active_run(conn) is not None
    except Exception:  # noqa: BLE001 - "no data" is exactly what a failure means here
        return False


def text_of(app: AppTest) -> str:
    """Every string the page rendered, lowercased, as one haystack."""
    parts: list[str] = []
    for name in (
        "title", "header", "subheader", "markdown", "caption", "text",
        "info", "warning", "error", "success", "metric",
    ):
        for element in getattr(app, name):
            for attribute in ("value", "body", "label"):
                got = getattr(element, attribute, None)
                if isinstance(got, str):
                    parts.append(got)
    return "\n".join(parts).lower()


# --------------------------------------------------------------------------- #
# 1. every page renders, whatever the database holds (sec.10)                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(PAGES))
def test_page_renders_without_raising(name):
    """sec.10: "never a traceback". An empty database is a message, not a crash."""
    app = run(PAGES[name])
    assert not app.exception, (
        f"{name} raised {app.exception[0].message if app.exception else ''}"
    )


@pytest.mark.parametrize("name", list(PAGES))
def test_page_says_something(name):
    """A page that renders nothing at all passes the exception check vacuously."""
    app = run(PAGES[name])
    assert text_of(app).strip(), f"{name} rendered no text"


# --------------------------------------------------------------------------- #
# 2. the sec.14 constraint, enforced rather than trusted                       #
# --------------------------------------------------------------------------- #
def test_no_html_css_or_javascript_anywhere_in_the_dashboard():
    """sec.14: "You will not write a single line of HTML, CSS, or JavaScript."

    The rule is quoted in app.py's own docstring, and until now nothing checked
    it. ``unsafe_allow_html`` is the only door into HTML that Streamlit offers,
    so guarding it guards the whole constraint.

    Checked on the parsed syntax tree rather than on the file's text: app.py
    quotes the prohibition verbatim, so a substring scan fails on the very
    docstring that states the rule. What matters is whether the argument is ever
    PASSED, which is a question about calls, not about characters.
    """
    import ast

    offenders = []
    for path in sorted(DASHBOARD.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                keyword.arg == "unsafe_allow_html" for keyword in node.keywords
            ):
                offenders.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}:{node.lineno}")

    assert not offenders, f"sec.14 forbids HTML; unsafe_allow_html passed at {offenders}"


# --------------------------------------------------------------------------- #
# 3. the things each page exists to say (skipped without data)                 #
# --------------------------------------------------------------------------- #
def test_home_states_the_research_question(home, has_data):
    """sec.2.3's question is the home page's whole job."""
    if not has_data:
        pytest.skip("no panel or no active model")
    body = text_of(home)
    assert "pledge" in body
    assert "volatility" in body


def test_home_reports_the_headline_delta(home, has_data):
    """The delta between the two headline experiments must reach the page."""
    if not has_data:
        pytest.skip("no panel or no active model")
    labels = [metric.label.lower() for metric in home.metric]
    assert any("delta" in label for label in labels), labels


@pytest.mark.parametrize("name", ["scanner", "company", "validation"])
def test_page_renders_a_dataframe(name, has_data):
    """Each sub-page's primary output is tabular; an empty page is a regression."""
    if not has_data:
        pytest.skip("no panel or no active model")
    app = run(PAGES[name])
    assert len(app.dataframe) >= 1, f"{name} rendered no table"


def test_validation_states_its_limitations(has_data):
    """sec.2.5: the limitations are not optional furniture."""
    if not has_data:
        pytest.skip("no panel or no active model")
    body = text_of(run(PAGES["validation"]))
    for term in ("survivorship", "confound", "quarters"):
        assert term in body, f"limitations text lost {term!r}"


def test_scanner_offers_a_date_and_a_size_control(has_data):
    if not has_data:
        pytest.skip("no panel or no active model")
    app = run(PAGES["scanner"])
    assert len(app.selectbox) >= 1, "no observation-date selector"
    assert len(app.slider) >= 1, "no top-N control"
