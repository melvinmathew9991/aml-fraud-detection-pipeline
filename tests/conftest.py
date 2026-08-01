"""
Adds src/ to sys.path so tests can use the same flat imports
(`from config import ...`) that the pipeline modules use internally --
src/ is not a package (no __init__.py), by design, so it isn't importable
as `src.config` without this.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
