"""
Tests config.py's path resolution.

Regression cover for a real defect found in the pre-deployment audit
(2026-08-03): config.yaml sets `mlflow.tracking_uri: "sqlite:///mlflow.db"`,
a *relative* URI, and train_pipeline.py passed it straight to
mlflow.set_tracking_uri(). MLflow resolves a relative sqlite path against the
process working directory, not the project root -- so running the pipeline
from src/ wrote to src/mlflow.db instead of the project store. 54 runs across
2 experiments ended up in that orphan database while the canonical store held
108, silently splitting a third of the project's experiment history.

Every other path in train_pipeline.py (RAW_PATH, PROCESSED_DIR, MODEL_DIR,
REPORTS_DIR) was already anchored to PROJECT_ROOT; the tracking URI was the
one that was not.
"""

from pathlib import Path

import pytest

from config import PROJECT_ROOT, load_config, resolve_tracking_uri


def test_relative_sqlite_uri_is_anchored_to_project_root():
    resolved = resolve_tracking_uri("sqlite:///mlflow.db")
    assert resolved == "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()
    # The whole point: the resolved path must not depend on the caller's cwd.
    assert Path(resolved[len("sqlite:///"):]).is_absolute()


def test_resolution_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from_tmp = resolve_tracking_uri("sqlite:///mlflow.db")
    monkeypatch.chdir(PROJECT_ROOT / "src")
    from_src = resolve_tracking_uri("sqlite:///mlflow.db")
    assert from_tmp == from_src, "tracking URI must not vary with cwd -- this is the bug"


def test_absolute_sqlite_uri_is_left_alone():
    absolute = "sqlite:///" + (PROJECT_ROOT / "elsewhere.db").as_posix()
    assert resolve_tracking_uri(absolute) == absolute


@pytest.mark.parametrize(
    "uri",
    [
        "http://localhost:5000",
        "https://mlflow.example.com",
        "databricks",
        "file:./mlruns",
    ],
)
def test_non_sqlite_backends_are_untouched(uri):
    assert resolve_tracking_uri(uri) == uri


def test_configured_tracking_uri_is_relative_so_resolution_is_load_bearing():
    """If config.yaml ever switches to an absolute URI this test should be
    revisited -- but while it is relative, resolution is what keeps the store
    in one place."""
    uri = load_config()["mlflow"]["tracking_uri"]
    assert uri.startswith("sqlite:///")
    assert not Path(uri[len("sqlite:///"):]).is_absolute()
