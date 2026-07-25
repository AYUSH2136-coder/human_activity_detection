"""
General-purpose helper utilities.
"""

import json
import time
import yaml
from pathlib import Path
from contextlib import contextmanager
from typing import Any


# ── Project root ───────────────────────────────────────────────────────────────
def get_project_root() -> Path:
    """Return the absolute project root directory (contains pyproject.toml)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[2]  # fallback: src/utils/../../


# ── YAML ───────────────────────────────────────────────────────────────────────
def load_yaml(path: str | Path) -> dict:
    """Load and return a YAML file as a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    """Save a dict to a YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# ── JSON ───────────────────────────────────────────────────────────────────────
def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save arbitrary data to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    """Load and return a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Timer ──────────────────────────────────────────────────────────────────────
@contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    tag = f"[{label}] " if label else ""
    print(f"  ⏱  {tag}Done in {elapsed:.2f}s")


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


# ── Directory helpers ──────────────────────────────────────────────────────────
def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (including parents) if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def resolve_path(relative: str | Path, root: Path | None = None) -> Path:
    """Resolve a relative path against the project root."""
    if root is None:
        root = get_project_root()
    return (root / relative).resolve()
