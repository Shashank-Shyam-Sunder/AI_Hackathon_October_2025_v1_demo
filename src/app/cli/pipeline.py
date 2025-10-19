#!/usr/bin/env python3
"""
pipeline.py — run the full flow:
  1) Interactive intake  -> creates data/out/<slug>-<ts>
  2) task_detect         -> writes selected_tasks_*.json, explain_*.md
  3) plan_and_cost       -> writes cost_breakdown_*.json, cost_summary_*.md

Examples:
  python -m src.app.cli.pipeline
  python -m src.app.cli.pipeline --llm-provider openai
  python -m src.app.cli.pipeline --reuse-run-dir "data/out/acme-20251019_193012"
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from contextlib import contextmanager

# step entrypoints (sibling package: core)
from ..core.cli_intake import main as intake_main
from ..core.task_detect import main as detect_main
from ..core.plan_and_cost import main as cost_main

# only use what io_paths actually provides
from ..utils.io_paths import resolve_run_dir

@contextmanager
def _temp_argv(args_list: list[str] | None):
    """
    Temporarily replace sys.argv so sub-steps that parse argparse
    see the arguments we want to pass.
    """
    old_argv = sys.argv[:]
    sys.argv = ["__pipeline_step__"] + (args_list or [])
    try:
        yield
    finally:
        sys.argv = old_argv

def _run_step(fn, argv: list[str] | None) -> int:
    with _temp_argv(argv):
        rc = fn()
    return int(rc or 0)

def _latest_run_dir() -> Path:
    """
    Use io_paths.resolve_run_dir(None) to pick the most recent run under data/out.
    This also prints the 'Auto-selected latest run directory' line for consistency.
    """
    return resolve_run_dir(None)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the full Better App Cost Estimator pipeline")
    parser.add_argument(
        "--reuse-run-dir",
        help="Optional: skip interactive intake and reuse an existing run dir (data/out/<slug>-<ts>)",
        default=None,
    )
    parser.add_argument(
        "--llm-provider",
        choices=["perplexity", "openai", "grok", "google", "none"],
        help="Pass through to task_detect (if omitted, you'll be prompted there).",
        default=None,
    )
    parser.add_argument(
        "--catalog",
        help="Catalog path to pass to task_detect and plan_and_cost (default internal).",
        default=None,
    )
    args = parser.parse_args(argv)

    # 1) Intake (or reuse)
    if args.reuse_run_dir:
        run_dir: Path = resolve_run_dir(args.reuse_run_dir)
        if not run_dir.exists():
            raise SystemExit(f"--reuse-run-dir not found: {args.reuse_run_dir}")
        print(f"↪ Reusing existing run directory: {run_dir}")
    else:
        print("→ Running interactive intake ...")
        rc = _run_step(intake_main, [])
        if rc != 0:
            return rc
        # pick the most recently created run via io_paths helper
        run_dir = _latest_run_dir()
        if not run_dir.exists():
            raise SystemExit("Intake did not create a run directory. Aborting.")
        print(f"📁 New run directory: {run_dir}")

    # 2) Detect
    print(f"\n→ Running task_detect for {run_dir} ...")
    detect_argv = ["--run-dir", str(run_dir)]
    if args.llm_provider:
        detect_argv += ["--llm-provider", args.llm_provider]
    if args.catalog:
        detect_argv += ["--catalog", args.catalog]
    rc = _run_step(detect_main, detect_argv)
    if rc != 0:
        return rc

    # 3) Plan & cost
    print(f"\n→ Running plan_and_cost for {run_dir} ...")
    cost_argv = ["--run-dir", str(run_dir)]
    if args.catalog:
        cost_argv += ["--catalog", args.catalog]
    rc = _run_step(cost_main, cost_argv)
    if rc != 0:
        return rc

    print(f"\n✅ Pipeline complete for: {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
