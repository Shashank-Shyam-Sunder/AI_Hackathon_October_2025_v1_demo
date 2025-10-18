#!/usr/bin/env python3
"""
cli_intake.py — Step 1 (interactive)
- Prompts the user for: NAME FIRST, then other intake fields.
- Creates a per-run folder under data/out:  data/out/<slug>-YYYYMMDD_HHMMSS/
- Writes:  intake_<timestamp>.json  inside that run folder.
- Stays standalone; chaining/pipeline will be added later via a separate module.
"""

from __future__ import annotations
import json
import sys
from typing import Optional

# I/O helpers (UI-agnostic)
from ..utils.io_paths import make_run_dir, write_run_meta, next_out_path, timestamp

# ---------- tiny prompt helpers ----------
def prompt_str(label: str, default: Optional[str] = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        val = input(f"{label}{suffix}: ").strip()
        if not val and default is not None:
            val = default
        if required and not val:
            print("  • This field is required. Please enter a value.")
            continue
        return val

def prompt_float(label: str, default: Optional[float] = None, min_val: Optional[float] = None) -> Optional[float]:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip()
        if not raw:
            return default
        try:
            v = float(raw)
            if min_val is not None and v < min_val:
                print(f"  • Must be ≥ {min_val}.")
                continue
            return v
        except ValueError:
            print("  • Please enter a number.")

def prompt_int(label: str, default: Optional[int] = None, min_val: Optional[int] = None) -> Optional[int]:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
            if min_val is not None and v < min_val:
                print(f"  • Must be ≥ {min_val}.")
                continue
            return v
        except ValueError:
            print("  • Please enter an integer.")

def prompt_choice(label: str, choices: list[str], default: Optional[str] = None) -> str:
    ch_str = "/".join(choices)
    while True:
        suffix = f" [{default}]" if default is not None else ""
        val = input(f"{label} ({ch_str}){suffix}: ").strip().lower()
        if not val and default is not None:
            val = default
        if val in choices:
            return val
        print(f"  • Choose one of: {', '.join(choices)}")

def prompt_yes_no(label: str, default: Optional[bool] = None) -> bool:
    def show_default(d):
        if d is True: return "y"
        if d is False: return "n"
        return None
    d = show_default(default)
    while True:
        suffix = f" [{d}]" if d is not None else ""
        val = input(f"{label} (y/n){suffix}: ").strip().lower()
        if not val and d is not None:
            return default  # type: ignore[return-value]
        if val in ("y","yes"): return True
        if val in ("n","no"): return False
        print("  • Please enter y or n.")

# ---------- intake flow ----------
def collect_intake_interactive() -> dict:
    print("\n=== Better App Cost Estimator — Intake (Step 1) ===\n")

    # IMPORTANT: ask NAME FIRST (used to create the run folder)
    name = prompt_str("Company/User name", required=True)

    app_brief = prompt_str("Describe the app you want to build", required=True)

    users = prompt_int("Estimated monthly active users (MAU)", default=100, min_val=0)
    qpm = prompt_int("Queries per user per month (QPM)", default=30, min_val=0)

    availability = prompt_choice("Availability target", choices=["99.0","99.9"], default="99.0")
    monitoring = prompt_choice("Monitoring tier", choices=["basic","standard","enhanced"], default="basic")
    compliance = prompt_choice("Compliance", choices=["none","gdpr","hipaa"], default="none")

    hosting = prompt_choice("Hosting preference", choices=["api","cloud","self_host"], default="api")
    fine_tune_pref = prompt_choice("Fine-tuning", choices=["none","maybe","required"], default="none")

    rag_needed = prompt_yes_no("Will you use your own knowledge base / RAG?", default=False)
    corpus_gb = embed_model = refresh_rate = avg_chunk_tokens = None
    if rag_needed:
        corpus_gb = prompt_float("Corpus size (GB)", default=5.0, min_val=0.0)
        embed_model = prompt_str("Embedding model (Enter for default)", default="text-embedding-3-large")
        refresh_rate = prompt_choice("Corpus refresh cadence", choices=["one_time","monthly","weekly","daily"], default="monthly")
        avg_chunk_tokens = prompt_int("Average chunk size (tokens)", default=350, min_val=1)

    need_translation = prompt_yes_no("Is multilingual translation required?", default=False)
    language_pairs = None
    if need_translation:
        language_pairs = prompt_str("Language pairs (e.g., auto→en, en→es)", default="auto→en")
    domain = prompt_choice("Primary domain", choices=["general","legal","medical","financial","technical"], default="general")

    agentic = prompt_yes_no("Will the app orchestrate tools (agentic workflows)?", default=False)
    num_tools = chain_depth = avg_steps = None
    if agentic:
        num_tools = prompt_int("Number of tools", default=2, min_val=1)
        chain_depth = prompt_int("Max chain depth (steps in a path)", default=4, min_val=1)
        avg_steps = prompt_int("Average steps per request", default=6, min_val=1)

    payload = {
        "meta": {"name": name},
        "app_brief": app_brief,
        "traffic": {"users": users, "qpm": qpm},
        "ops": {
            "availability_target": float(availability),
            "monitoring_tier": monitoring.capitalize(),
            "compliance": compliance.upper() if compliance != "none" else "None",
        },
        "preferences": {"hosting": hosting, "fine_tuning": fine_tune_pref, "domain": domain},
        "rag": {
            "enabled": rag_needed,
            "corpus_gb": corpus_gb,
            "embed_model": embed_model,
            "refresh_rate": refresh_rate,
            "avg_chunk_tokens": avg_chunk_tokens,
        },
        "i18n": {"translation_required": need_translation, "language_pairs": language_pairs},
        "agentic": {"enabled": agentic, "num_tools": num_tools, "chain_depth": chain_depth, "avg_steps": avg_steps},
        "detected_tasks": [],
    }
    return payload

def main() -> int:
    # 1) Interactive intake (NAME FIRST, then fields)
    intake = collect_intake_interactive()
    name = intake["meta"]["name"]

    # 2) Create run directory under data/out using the provided name
    run_dir = make_run_dir(name)  # e.g., data/out/acme-ltd-20251018_143012

    # Optional: write run metadata (handy for a future UI)
    write_run_meta(run_dir, name, extra={"phase": "intake"})

    # 3) Write intake_<ts>.json into that run dir
    out_path = next_out_path(run_dir, "intake", "json")  # uses a fresh timestamp
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(intake, f, indent=2, ensure_ascii=False)

    print("\n✅ Intake saved:")
    print(f"   {out_path}")
    print(f"   Run directory: {run_dir}\n")
    print("Next (manual for now):")
    print(f"  python -m src.app.core.task_detect --run-dir \"{run_dir}\"")
    print(f"  python -m src.app.core.plan_and_cost --run-dir \"{run_dir}\"")
    return 0

if __name__ == "__main__":
    sys.exit(main())