"""
Guards every `from common import ...` in dashboard/pages/*.py against the
name actually existing on dashboard/common.py.

This exists because of a real regression. Sprint 5 imported FEATURE_COLUMNS,
CURRENT_BUNDLE_SNAPSHOT_STEP and MODEL_LIMITATIONS into common.py purely to
re-export them to the pages; Sprint 6 (`0c86106`) removed those three lines,
which look exactly like unused imports to a linter, and nothing in this repo
noticed. `ruff` passed, `pytest` passed, CI was green, the container built and
deployed -- because tests/test_dashboard_common.py exercises common.py and the
pages are never imported by anything. The failure surfaced only when Streamlit
Community Cloud ran pages/4_Model_Card.py in a browser, six weeks later:

    ImportError: cannot import name 'CURRENT_BUNDLE_SNAPSHOT_STEP' from 'common'

The check is static rather than an actual import of each page. Importing a page
executes its Streamlit calls (st.set_page_config and friends), which needs a
script-run context that does not exist under pytest, and Streamlit's own
AppTest machinery is heavier than this needs -- the bug is a missing attribute
on a module, so reading the import statements and looking the names up is the
whole test.
"""

import ast
from pathlib import Path

import common
import pytest

PAGES_DIR = Path(common.__file__).resolve().parent / "pages"
PAGE_FILES = sorted(PAGES_DIR.glob("*.py"))


def _names_imported_from_common(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "common"
        for alias in node.names
    ]


def test_pages_directory_is_found_and_non_empty():
    # If this fails the rest of the module silently tests nothing.
    assert PAGE_FILES, f"no page modules found under {PAGES_DIR}"


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_every_name_a_page_imports_from_common_exists(page):
    missing = [n for n in _names_imported_from_common(page) if not hasattr(common, n)]
    assert not missing, (
        f"{page.name} imports {missing} from common, which common.py does not "
        f"define. This breaks the page at import time in the deployed dashboard."
    )
