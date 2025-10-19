"""
io_paths.py — Common I/O utilities for run directory management and JSON files
Used by:
  • cli_intake.py
  • task_detect.py
  • plan_and_cost.py
"""

from __future__ import annotations
from pathlib import Path
import json, time
from typing import Any, Dict, Optional

# ------------------------------------------------------------
# Base paths
# ------------------------------------------------------------

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Run directory helpers
# ------------------------------------------------------------

def resolve_run_dir(run_dir_arg: Optional[str] = None) -> Path:
    """
    Resolve the target run directory.
    - If run_dir_arg is provided, return that path (creating if necessary).
    - Otherwise, pick the most recently modified directory under data/out.
    """
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    # No argument → auto-pick latest under data/out
    subdirs = [p for p in OUT_DIR.iterdir() if p.is_dir()]
    if not subdirs:
        raise SystemExit("❌ No run directory found. Run the intake step first.")
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = subdirs[0]
    print(f"📁 Auto-selected latest run directory: {latest}")
    return latest

def next_out_path(run_dir: Path, prefix: str, ext: str) -> Path:
    """
    Generate a new timestamped output path inside the run directory.
    Example:
      next_out_path(run_dir, 'selected_tasks', 'json')
      -> data/out/<run>/selected_tasks_20251019_172015.json
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f"{prefix}_{ts}.{ext}"

def require_latest_in(
    run_dir: Path,
    prefix: str,
    suffix: str,
    missing_msg: str = "Required file not found."
) -> Path:
    """
    Find the most recent file matching prefix/suffix in a run dir.
    Example:
      require_latest_in(run_dir, 'intake_', '.json')
    """
    matches = sorted(
        (p for p in run_dir.glob(f"{prefix}*{suffix}") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise SystemExit(f"❌ {missing_msg}")
    return matches[0]

# ------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """Write a dictionary to JSON file (UTF-8, indented)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

# ------------------------------------------------------------
# Misc utilities
# ------------------------------------------------------------

def list_runs(limit: int = 10) -> None:
    """Print the most recent run directories for quick navigation."""
    runs = sorted(
        [p for p in OUT_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        print("No runs found under data/out.")
        return
    print("Recent runs:")
    for p in runs[:limit]:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
        print(f"  • {p.name} (modified {ts})")

# ------------------------------------------------------------
# Example usage (debug)
# ------------------------------------------------------------

if __name__ == "__main__":
    # Try auto-resolving a run directory
    rd = resolve_run_dir()
    print(f"Resolved run dir: {rd}")
    # Find latest intake file
    try:
        intake = require_latest_in(rd, "intake_", ".json")
        print(f"Latest intake: {intake}")
    except SystemExit as e:
        print(e)
