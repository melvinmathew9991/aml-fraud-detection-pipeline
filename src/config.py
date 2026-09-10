"""
config.py

Loads config.yaml once and exposes it as a plain dict, so paths and
hyperparameters live in one place instead of being hardcoded across scripts.

Also holds `git_commit_hash`, the provenance stamp written into every model's
metadata.json and into the drift job's reference manifest. It lives here rather
than in train_pipeline.py (where it started) because Sprint 8 needs the same
stamp from a job that must not import the training pipeline -- importing that
module runs its logging setup and mkdir side effects as a side effect of asking
for a commit hash.
"""

import logging
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

logger = logging.getLogger(__name__)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit_hash() -> str:
    """Short HEAD hash, suffixed `-dirty` when the working tree has changes.

    Best-effort by design: a missing git, a tarball checkout or a detached
    environment must degrade to "nogit" rather than fail whatever run asked for
    provenance.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).strip())
        return f"{commit}-dirty" if dirty else commit
    except Exception as exc:  # noqa: BLE001 -- best-effort provenance lookup, must never fail the run
        logger.debug("git_commit_hash: falling back to 'nogit' (%s)", exc)
        return "nogit"


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
