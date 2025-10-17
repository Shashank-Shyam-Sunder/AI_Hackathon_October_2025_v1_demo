#!/usr/bin/env python3
"""
cli_intake.py — Step 1: Collect global inputs for AI Cost Estimator (CLI)
Writes: <PROJECT_ROOT>/data/raw/intake_YYYYMMDD_HHMMSS.json
"""

import json
import os
import sys
from datetime import datetime

# ---------- path constants (root-anchored) ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# ---------- helpers ----------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def prompt_str(label: str, default: str | None = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        val = input(f"{label}{suffix}: ").strip()
        if not val and default is not None:
            val = default
        if required and not val:
            print("  • This field is required. Please enter a value.")
            continue
        return val

def prompt_float(label: str, default: float | None = None, min_val: float | None = None) -> float | None:
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

def prompt_int(label: str, default: int | None = None, min_val: int | None = None) -> int | None:
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

def prompt_choice(label: str, choices: list[str], default: str | None = None) -> str:
    ch_str = "/".join(choices)
    while True:
        suffix = f" [{default}]" if default is not None else ""
        val = input(f"{label} ({ch_str}){suffix}: ").strip().lower()
        if not val and default is not None:
            val = default
        if val in choices:
            return val
        print(f"  • Choose one of: {', '.join(choices)}")

def prompt_yes_no(label: str, default: bool | None = None) -> bool:
    def show_default(d):
        if d is True: return "y"
        if d is False: return "n"
        return None
    d = show_default(default)
    while True:
        suffix = f" [{d}]" if d is not None else ""
        val = input(f"{label} (y/n){suffix}: ").strip().lower()
        if not val and d is not None:
            return default
        if val in ("y","yes"): return True
        if val in ("n","no"): return False
        print("  • Please enter y or n.")

# ---------- intake flow ----------
def collect_intake() -> dict:
    print("\n=== Better App Cost Estimator — Intake (Step 1) ===\n")

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
        embed_model = prompt_str("Embedding model (for defaults type Enter)", default="text-embedding-3-large")
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
    ensure_dir(RAW_DIR)
    payload = collect_intake()

    out_path = os.path.join(RAW_DIR, f"intake_{ts()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n✅ Intake saved:")
    print(f"   {out_path}\n")
    print("Next steps:")
    print("  1) Run task detection on this intake JSON (src/task_detect.py).")
    print("  2) Auto-generate follow-up questions per task using the catalog (200-row table).")
    print("  3) Build MIN/MAX plans and compute cost.\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
