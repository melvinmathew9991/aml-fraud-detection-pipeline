#!/usr/bin/env python
"""
tasks.py -- the project's task runner.

Deliberately a Python script rather than a Makefile: `make` is not available
in this project's Windows development environment (verified 2026-08-03 -- no
make, mingw32-make, gmake or nmake), so a Makefile could not be run or
verified here. Shipping commands that cannot be executed on the machine that
ships them is the "documented but unverified" pattern this repo has been
caught by repeatedly (see AUDIT.md §5). Python is guaranteed present, since
this is a Python project, and works identically on Windows, Linux and macOS.

No third-party dependencies -- this must run before `pip install` has.

    python tasks.py            # list tasks
    python tasks.py check      # what CI's lint-test job runs, minus smoke-train
    python tasks.py api        # serve the API locally with reload

The `check` task mirrors .github/workflows/ci.yml deliberately: passing it
locally should mean the PR gate passes, so failures are found before the
~3 minute CI round-trip rather than after.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*cmd: str, cwd: Path | None = None) -> int:
    """Run a command, echoing it first so the output is reproducible by hand."""
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    return subprocess.call(cmd, cwd=str(cwd or ROOT))


def py(*args: str) -> int:
    """Run a module with the *current* interpreter, not whatever `python` resolves
    to on PATH -- this repo has both 3.11 and 3.13 installed."""
    return run(sys.executable, *args)


# --------------------------------------------------------------------------- tasks


def task_test() -> int:
    """Run the full pytest suite."""
    return py("-m", "pytest", "-q")


def task_lint() -> int:
    """Run ruff over the same paths CI does."""
    return py("-m", "ruff", "check", "src/", "dashboard/", "tests/", "tasks.py")


def task_typecheck() -> int:
    """Run mypy (scope is pinned to src/inference in pyproject.toml)."""
    return py("-m", "mypy")


def task_check() -> int:
    """lint -> typecheck -> test, in CI's order. Stops at the first failure."""
    for name, fn in (("lint", task_lint), ("typecheck", task_typecheck), ("test", task_test)):
        code = fn()
        if code != 0:
            print(f"\n[check] {name} FAILED (exit {code})", file=sys.stderr)
            return code
    print("\n[check] lint + typecheck + test all passed")
    return 0


def task_api() -> int:
    """Serve the FastAPI app locally on :8000 with autoreload."""
    return py("-m", "uvicorn", "api.main:app", "--app-dir", "src",
              "--reload", "--host", "127.0.0.1", "--port", "8000")


def task_dashboard() -> int:
    """Serve the Streamlit dashboard. Set API_BASE_URL if the API is elsewhere."""
    return py("-m", "streamlit", "run", "dashboard/Home.py")


def task_train() -> int:
    """Run the full training pipeline (~28 min on the reference machine)."""
    return py("src/train_pipeline.py")


def task_bundle() -> int:
    """Re-export the serving bundle from the latest run under models/."""
    return py("src/export_bundle.py")


def task_sample_data() -> int:
    """Generate the 50k-row synthetic sample -- REFUSES to clobber the real dataset.

    src/generate_sample_data.py writes to data/raw/paysim_transactions.csv, the
    same path the real ~493MB PaySim dataset occupies. It destroyed that file
    once (AUDIT.md §5, Sprint 6). The script itself still overwrites silently;
    this guard is why the task exists rather than documenting the raw command.
    """
    target = ROOT / "data" / "raw" / "paysim_transactions.csv"
    if target.exists():
        size_mb = target.stat().st_size / (1024 ** 2)
        if size_mb > 50:
            print(
                f"REFUSING: {target} is {size_mb:.0f} MB -- that is the real PaySim\n"
                f"dataset, not a sample. Generating would overwrite it in place and\n"
                f"it is only recoverable by re-downloading from Kaggle.\n\n"
                f"Move or delete it first if you really want a synthetic sample.",
                file=sys.stderr,
            )
            return 1
        print(f"note: overwriting existing {size_mb:.1f} MB sample at {target}")
    return py("src/generate_sample_data.py")


def task_clean() -> int:
    """Remove Python caches. Does not touch data/, models/, reports/ or mlflow.db."""
    removed = 0
    for pattern in ("**/__pycache__", "**/.pytest_cache", "**/*.pyc"):
        for path in ROOT.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1
    print(f"removed {removed} cache entries")
    return 0


TASKS = {
    "check": task_check,
    "test": task_test,
    "lint": task_lint,
    "typecheck": task_typecheck,
    "api": task_api,
    "dashboard": task_dashboard,
    "train": task_train,
    "bundle": task_bundle,
    "sample-data": task_sample_data,
    "clean": task_clean,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", nargs="?", choices=sorted(TASKS), help="task to run")
    args = parser.parse_args()

    if not args.task:
        width = max(len(name) for name in TASKS)
        print("tasks:\n")
        for name in sorted(TASKS):
            summary = (TASKS[name].__doc__ or "").strip().splitlines()[0]
            print(f"  {name:<{width}}  {summary}")
        print("\nusage: python tasks.py <task>")
        return 0

    return TASKS[args.task]()


if __name__ == "__main__":
    sys.exit(main())
