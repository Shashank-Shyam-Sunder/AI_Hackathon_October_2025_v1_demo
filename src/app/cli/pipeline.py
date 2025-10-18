#!/usr/bin/env python3
"""
pipeline.py — run the full flow:
  1) Interactive intake (asks NAME first, then fields)
  2) task_detect on that run
  3) plan_and_cost on that run

You can still run each step individually.
"""

from __future__ import annotations
import argparse
from pathlib import Path

# relative imports only
from ..core.cli_intake import main as intake_main
from ..core.task_detect import main as detect_main
from ..core.plan_and_cost import main as cost_main
from ..utils.io_paths import latest_run_dir, resolve_run_dir

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the full pipeline")
    parser.add_argument(
        "--reuse-run-dir",
        help="Optional: skip interactive intake and reuse an existing run dir (data/out/<slug>-<ts>)",
        default=None,
    )
    args = parser.parse_args(argv)

    if args.reuse_run_dir:
        run_dir: Path = resolve_run_dir(args.reuse_run_dir)
    else:
        # 1) Run interactive intake (this will create a new run dir)
        rc = intake_main()
        if rc not in (0, None):
            return int(rc)
        run_dir = latest_run_dir()
        if not run_dir or not run_dir.exists():
            raise SystemExit("Intake did not create a run directory. Aborting.")

    # 2) Detect
    print(f"\n→ Running task_detect for {run_dir} ...")
    rc = detect_main(["--run-dir", str(run_dir)])
    if rc not in (0, None):
        return int(rc)

    # 3) Plan & cost
    print(f"\n→ Running plan_and_cost for {run_dir} ...")
    rc = cost_main(["--run-dir", str(run_dir)])
    if rc not in (0, None):
        return int(rc)

    print(f"\n✅ Pipeline complete for: {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
