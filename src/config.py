"""
config.py

Loads config.yaml once and exposes it as a plain dict, so paths and
hyperparameters live in one place instead of being hardcoded across scripts.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_tracking_uri(uri: str, project_root: Path = PROJECT_ROOT) -> str:
    """Anchor a relative ``sqlite:///`` MLflow tracking URI to the project root.

    Every path in train_pipeline.py is resolved against PROJECT_ROOT, but
    ``mlflow.set_tracking_uri()`` resolves a relative sqlite path against the
    *process working directory*. Running the pipeline from ``src/`` therefore
    wrote to ``src/mlflow.db`` rather than the project store -- which is what
    happened: 54 runs across 2 experiments landed in an orphan database that
    nothing reads, while the canonical store held 108.

    Absolute paths and non-sqlite backends are returned unchanged.
    """
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    path = uri[len(prefix):]
    if Path(path).is_absolute():
        return uri
    return prefix + (project_root / path).as_posix()
