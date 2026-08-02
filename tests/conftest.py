"""
Adds src/ and dashboard/ to sys.path so tests can use the same flat
imports (`from config import ...`, `from common import ...`) those
directories use internally -- neither is a package (no __init__.py), by
design, so they aren't importable as `src.config`/`dashboard.common`
without this. Streamlit does the equivalent automatically at runtime by
putting the app script's directory on sys.path.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _dir in (REPO_ROOT / "src", REPO_ROOT / "dashboard"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
