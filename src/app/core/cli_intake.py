#!/usr/bin/env python3
"""
cli_intake.py — Step 1 (interactive)
- Prompts only the fields we can price in the MVP.
- Creates a per-run folder under data/out:  data/out/<slug>-YYYYMMDD_HHMMSS/
- Writes: intake_<timestamp>.json inside that run folder.
"""

from __future__ import annotations
import json
import sys
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# Use only what exists in io_paths
from ..utils.io_paths import OUT_DIR, next_out_path, write_json

# ---------- tiny local run-dir helpers (do NOT modify io_paths) ----------
def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _slugify(value: str) -> str:
    """
    Simplified slugify: ascii, lower, keep alnum and dashes.
    """
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\-]+", "-", value.lower()).strip("-")
    value = re.sub(r"-{2,}", "-", value)
    return value or "run"

def make_run_dir(owner_name: str) -> Path:
    """
    Create a fresh run directory under data/out:
      data/out/<slug>-YYYYMMDD_HHMMSS
    Also writes a small run_meta.json for convenience.
    """
    slug = _slugify(owner_name)
    rd = OUT_DIR / f"{slug}-{_timestamp()}"
    rd.mkdir(parents=True, exist_ok=True)
    # minimal run meta
    write_json(
        {"owner": owner_name, "slug": slug, "created_at": _timestamp(), "phase": "intake"},
        rd / "run_meta.json",
        indent=2,
    )
    return rd

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

def prompt_choice(label: str, choices: List[str], default: Optional[str] = None, help_text: Optional[str] = None) -> str:
    ch_str = "/".join(choices)
    if help_text:
        print(help_text)
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

# ---------- helpers: natural-language language-pair parsing ----------
_SPLIT_ARROW = re.compile(r"\s*->\s*|\s*→\s*|\s+")  # '->' or unicode arrow or whitespace

def _normalize_lang_token(tok: str) -> str:
    return tok.strip().strip(",;")

def parse_language_pairs(raw: str) -> Tuple[str, List[dict]]:
    """
    Accepts user-friendly inputs like:
      "auto English, English French"
    and stricter forms like:
      "auto->en, en->fr"
    Returns:
      normalized_string  e.g. "auto->English, English->French"
      parsed_list        e.g. [{"source":"auto","target":"English"}, ...]
    """
    if not raw:
        return "", []
    pairs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in _SPLIT_ARROW.split(chunk) if p]
        if len(parts) == 1:
            src, tgt = "auto", _normalize_lang_token(parts[0])
        elif len(parts) >= 2:
            src, tgt = _normalize_lang_token(parts[0]), _normalize_lang_token(parts[1])
        else:
            continue
        if src and tgt:
            pairs.append({"source": src, "target": tgt})
    normalized = ", ".join(f"{p['source']}->{p['target']}" for p in pairs)
    return normalized, pairs

# ---------- RAG chunk granularity mapping ----------
CHUNK_GRANULARITY: Dict[str, int] = {
    "small": 350,   # ~350 tokens per chunk (high granularity; more chunks)
    "medium": 650,  # ~650 tokens (balanced)
    "large": 900,   # ~900 tokens (fewer, larger chunks)
}

def describe_granularity() -> str:
    return (
        "    small  → ~350 tokens per chunk (more precise search, more chunks)\n"
        "    medium → ~650 tokens (balanced)\n"
        "    large  → ~900 tokens (fewer chunks, cheaper indexing)\n"
    )

# ---------- intake flow ----------
def collect_intake_interactive() -> dict:
    print("\n=== Better App Cost Estimator — Intake (Step 1) ===\n")
    print("Required:")
    print("  • Enter your Company/User name.")
    print("  • Describe the app you want to build.")
    print("\nTips:")
    print("  • For fields showing a value in [square brackets], you can press Enter to accept the default.")
    print("  • Only fields we can price in the MVP are asked here.\n")

    # Required identity & brief (readable labels)
    name = prompt_str("Company or user name", required=True)
    app_brief = prompt_str("Briefly describe the app you want to build", required=True)

    # Traffic & tokens (LLM token cost)
    users = prompt_int("Estimated monthly active users (MAU)", default=100, min_val=0)
    rpm = prompt_int("Requests per user per month (RPM)", default=30, min_val=0)
    tokens_in  = prompt_int("Average input tokens per request",  default=500, min_val=0)
    tokens_out = prompt_int("Average output tokens per request", default=300, min_val=0)

    # Compliance (flag — may add surcharge or tasks later)
    compliance = prompt_choice("Compliance requirement", choices=["none","gdpr","hipaa"], default="none")

    # Hosting preference (API or self_host) — readable label
    hosting = prompt_choice(
        "Hosting preference",
        choices=["api","self_host"],
        default="api",
        help_text="    api       → use model provider APIs (priced in MVP)\n"
                  "    self_host → run your own model servers (priced later if GPU rates available)\n"
    )

    # RAG (embedding/indexing cost block)
    rag_needed = prompt_yes_no("Will you use your own knowledge base / RAG?", default=False)
    corpus_gb = refresh_rate = chunk_granularity = None
    if rag_needed:
        corpus_gb = prompt_float("Knowledge base size (GB)", default=5.0, min_val=0.0)
        refresh_rate = prompt_choice(
            "How often will you refresh the knowledge base?",
            choices=["one_time","monthly","weekly","daily"],
            default="monthly"
        )
        print("Choose chunk granularity (used for splitting documents for search):")
        print(describe_granularity())
        chunk_granularity = prompt_choice("Granularity", choices=["small","medium","large"], default="medium")

    # Translation (keep simple and friendly)
    need_translation = prompt_yes_no("Do you need translation?", default=False)
    language_pairs_raw = language_pairs_norm = None
    language_pairs_list: List[dict] = []
    if need_translation:
        language_pairs_raw = prompt_str(
            "Enter language pairs (e.g., 'auto English, English French')",
            default="auto English"
        )
        language_pairs_norm, language_pairs_list = parse_language_pairs(language_pairs_raw)

    # Build payload
    avg_chunk_tokens = CHUNK_GRANULARITY.get(chunk_granularity, None) if rag_needed else None

    payload: Dict[str, Any] = {
        "meta": {"name": name},
        "app_brief": app_brief,

        # Core: LLM token pricing
        "traffic": {
            "users": users,
            "rpm": rpm,
            "qpm": rpm,  # back-compat alias
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        },

        # Ops — monitoring/availability internal in MVP
        "ops": {
            "availability_target": 99.0,        # internal default (not asked)
            "monitoring_tier": "Standard",      # internal label (no tiers in MVP)
            "compliance": compliance.upper() if compliance != "none" else "None",
        },

        # Hosting: API (priced) vs self_host (priced only if GPU rates available)
        "preferences": {"hosting": hosting},

        # RAG cost inputs (embedding/indexing)
        "rag": {
            "enabled": rag_needed,
            "corpus_gb": corpus_gb,
            "embed_model": None,            # internal default used in planner
            "refresh_rate": refresh_rate,
            "avg_chunk_tokens": avg_chunk_tokens,      # derived from granularity choice
            "chunk_granularity": chunk_granularity,    # keep the human-readable label too
        },

        # Translation
        "i18n": {
            "translation_required": need_translation,
            "language_pairs_raw": language_pairs_raw,      # free-form user text (for audit)
            "language_pairs": language_pairs_norm,         # normalized "a->b, c->d"
            "language_pairs_parsed": language_pairs_list,  # structured list for downstream scaling
        },

        # Internals we compute later (no prompts)
        "internals": {
            "monitoring": {
                "provider": "langsmith",
                "use_tiers": False
            },
            "guardrails_enabled": True,
            "embedding_model_default": "text-embedding-3-large"
        },

        # Placeholder to be filled by task_detect
        "detected_tasks": [],
    }
    return payload

def main() -> int:
    intake = collect_intake_interactive()
    name = intake["meta"]["name"]

    run_dir = make_run_dir(name)  # e.g., data/out/acme-ltd-20251018_143012

    out_path = next_out_path(run_dir, "intake", "json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(intake, f, indent=2, ensure_ascii=False)

    print("\n✅ Intake saved:")
    print(f"   {out_path}")
    print(f"   Run directory: {run_dir}\n")
    print("Next:")
    print(f"  python -m src.app.core.task_detect --run-dir \"{run_dir}\"")
    print(f"  python -m src.app.core.plan_and_cost --run-dir \"{run_dir}\"")
    return 0

if __name__ == "__main__":
    sys.exit(main())
