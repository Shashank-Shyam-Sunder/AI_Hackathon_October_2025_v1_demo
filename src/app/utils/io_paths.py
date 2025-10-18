# src/app/utils/io_paths.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import re

from ..config import settings

# --- Root locations ---
DATA_ROOT = Path("data")
RAW_DIR   = DATA_ROOT / "raw"                                 # keep catalog/static inputs here
OUT_ROOT  = Path(settings.app.output_dir or "data/out")       # all run outputs live under here

# --- Basics ---
def ensure_dirs() -> None:
    """Ensure OUT_ROOT exists."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

def slugify(name: str) -> str:
    """Filesystem-safe slug for folder names."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "run"

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Run folder lifecycle ---
def run_dir_name(name: str, when: Optional[str] = None) -> str:
    """Return folder name '<slug>-YYYYMMDD_HHMMSS' (does not create it)."""
    ts = when or timestamp()
    return f"{slugify(name)}-{ts}"

def make_run_dir(name: str, when: Optional[str] = None) -> Path:
    """
    Create and return a new run directory under OUT_ROOT.
    Call this immediately after you collect the user's name (via input()).
    """
    ensure_dirs()
    rd = OUT_ROOT / run_dir_name(name, when)
    rd.mkdir(parents=True, exist_ok=True)
    return rd

def write_run_meta(run_dir: Path, name: str, extra: Optional[Dict[str, Any]] = None) -> Path:
    """
    Persist minimal run metadata (useful for UI and later steps).
    Writes 'run.json' inside the run directory.
    """
    meta = {
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
    }
    if extra:
        meta.update(extra)
    path = run_dir / "run.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path

def read_run_meta(run_dir: Path) -> Dict[str, Any]:
    """Read 'run.json' if present; returns {} if missing."""
    path = run_dir / "run.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def list_run_dirs() -> List[Path]:
    """All existing run directories (sorted newest first)."""
    ensure_dirs()
    dirs = [p for p in OUT_ROOT.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)

def latest_run_dir() -> Optional[Path]:
    """Most recently modified run directory, or None."""
    dirs = list_run_dirs()
    return dirs[0] if dirs else None

def resolve_run_dir(run_dir: Optional[str]) -> Path:
    """
    If run_dir is given, return it (must exist). Otherwise return latest_run_dir().
    Raise with a clear message if nothing is available.
    """
    if run_dir:
        p = Path(run_dir)
        if not p.exists():
            raise SystemExit(f"Run directory not found: {p}")
        return p
    p = latest_run_dir()
    if not p:
        raise SystemExit("No run directory found. Create one first (e.g., via cli_intake).")
    return p

# --- File helpers (within a run dir) ---
def latest_file_in(dirpath: Path, prefix: str, suffix: str) -> Optional[Path]:
    """Latest file in dirpath matching prefix/suffix, or None."""
    files = sorted(dirpath.glob(f"{prefix}*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def require_latest_in(dirpath: Path, prefix: str, suffix: str, missing_msg: str) -> Path:
    """Like latest_file_in but raises if not found (nice for pipeline steps)."""
    p = latest_file_in(dirpath, prefix, suffix)
    if not p:
        raise SystemExit(missing_msg)
    return p

def next_out_path(run_dir: Path, stem_prefix: str, ext: str, reuse_ts_from: Optional[Path] = None) -> Path:
    """
    Build an output path like '<run_dir>/<stem_prefix>_<ts>.<ext>'.
    If reuse_ts_from is provided, reuse that file's timestamp stem part.
    """
    if reuse_ts_from is not None:
        # expects filenames like 'intake_YYYYMMDD_HHMMSS.json' => take part after first underscore
        base = reuse_ts_from.stem
        ts_part = base.split("_", 1)[-1] if "_" in base else timestamp()
    else:
        ts_part = timestamp()
    return run_dir / f"{stem_prefix}_{ts_part}.{ext.lstrip('.')}"
