"""Pytest bootstrap for the engine test suite.

The engine is an independent top-level package.  This file inserts the repository
root on ``sys.path`` so that ``import engine`` works no matter which directory
pytest is invoked from, without depending on environment variables or an
installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
