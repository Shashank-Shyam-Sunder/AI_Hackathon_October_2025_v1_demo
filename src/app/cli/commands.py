# import your script entrypoints
# adjust these names to match the actual functions/mains in your files
# If your scripts currently run as top-level __main__, we’ll call them via helper functions.
# src/app/cli/commands.py
import click

# relative imports ONLY
from ..utils.io_paths import ensure_dirs, RAW_DIR, OUT_ROOT
from ..core import cli_intake, task_detect, plan_and_cost

@click.group()
def cli():
    """AI Cost Estimator CLI"""

@cli.command("intake")
def intake_cmd():
    """Interactive intake (asks NAME first; writes intake_* into a new run folder)."""
    ensure_dirs()
    rc = cli_intake.main([])
    raise SystemExit(rc or 0)

@cli.command("detect")
@click.option("--run-dir", type=click.Path(), help="Run directory (data/out/<slug>-<ts>)")
@click.option("-i", "--input", "input_path", type=click.Path(exists=True),
              help="Explicit intake_*.json path (legacy style). If given without --run-dir, run-dir is inferred.")
@click.option("--topk", type=int, default=2, show_default=True, help="Top-k catalog rows per group")
def detect_cmd(run_dir, input_path, topk):
    """Detect tasks from an intake JSON and write detected_* into the same run folder."""
    ensure_dirs()
    args = []
    if run_dir:
        args += ["--run-dir", run_dir]
    if input_path:
        args += ["-i", input_path]
    if topk is not None:
        args += ["--topk", str(topk)]
    rc = task_detect.main(args)
    raise SystemExit(rc or 0)

@cli.command("plan")
@click.option("--run-dir", type=click.Path(), help="Run directory (data/out/<slug>-<ts>)")
@click.option("-d", "--detected", "detected_path", type=click.Path(exists=True),
              help="Explicit detected_*.json path (legacy style). If given without --run-dir, run-dir is inferred.")
def plan_cmd(run_dir, detected_path):
    """Build MIN/MAX plans and cost report; writes cost_report_* into the same run folder."""
    ensure_dirs()
    args = []
    if run_dir:
        args += ["--run-dir", run_dir]
    if detected_path:
        args += ["-d", detected_path]
    rc = plan_and_cost.main(args)
    raise SystemExit(rc or 0)

@cli.command("run")
def run_all_cmd():
    """Run full pipeline: intake → detect → plan (uses the same run folder)."""
    ensure_dirs()
    # 1) Intake (creates a new run folder)
    rc = cli_intake.main([])
    if rc not in (0, None):
        raise SystemExit(rc)
    # 2) Detect (auto-picks latest run folder)
    rc = task_detect.main([])
    if rc not in (0, None):
        raise SystemExit(rc)
    # 3) Plan (auto-picks latest run folder)
    rc = plan_and_cost.main([])
    raise SystemExit(rc or 0)

if __name__ == "__main__":
    cli()

