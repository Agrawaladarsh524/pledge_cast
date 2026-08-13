"""Make ``config`` and ``pledgecast`` importable regardless of how a script runs.

Decision R5: PLAN.md sec.8 places ``config.py`` at the repo root and the package
under ``src/``, but lists no ``pyproject.toml`` and sec.4 rules out Poetry. This
module adds the two needed paths - no packaging tooling, no new dependency.

Import it FIRST in every entry point::

    import _bootstrap  # noqa: F401
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
