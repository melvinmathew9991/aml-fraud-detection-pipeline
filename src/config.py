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
    with open(path, "r") as f:
        return yaml.safe_load(f)
