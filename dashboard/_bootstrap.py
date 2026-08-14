"""Make ``config`` and ``pledgecast`` importable from the Streamlit entry point.

The twin of ``scripts/_bootstrap.py``. Streamlit puts the main script's folder
on ``sys.path``, so ``import _bootstrap`` resolves here for both ``app.py`` and
every file under ``pages/``. Six duplicated lines, and the alternative is
packaging tooling PLAN.md sec.4 rules out.

Import it FIRST, before any ``config`` or ``pledgecast`` import.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

for _path in (PROJECT_ROOT, SRC_ROOT):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

__all__ = ["PROJECT_ROOT", "SRC_ROOT"]
