#!/usr/bin/env python3
"""
plan_and_cost.py — Step 3: Cost computation & reporting
Clean, business-style report:
  - Top summary (Requests/mo, RAG one-time & monthly, Monthly MIN/MAX)
  - Category rollups for MIN and MAX
  - MIN Plan / MAX Plan tables (human-readable; no stage/group noise)

Run:
  python -m src.app.core.plan_and_cost
  # or specify a run dir:
  python -m src.app.core.plan_and_cost --run-dir "data/out/<run-folder>"

Inputs (from latest run directory):
  • intake_*.json
  • selected_tasks_*.json
Catalog:
  • AI_Cost_Estimator_Catalog_UPDATED.xlsx (root or data/raw/)
Outputs:
  • cost_breakdown_*.json  (per-task details + rollups + summary)
  • cost_summary_*.md      (clean report)
"""

from __future__ import annotations
import argparse, json, collections
from pathlib import Path
from datetime import datetime
import pandas as pd

# Correct relative imports
from ..utils.io_paths import resolve_run_dir, require_latest_in, next_out_path
from .pricing_dict import PRICING_DICT
from .formulas import compute_token_cost, merged_group_input_tokens_per_task, compute_embedding_cost

# ==============================
# Tunables / constants (MVP)
# ==============================

# Monitoring (LangSmith) — flat "average" monthly
LANGSMITH_DEFAULT = {
    "traces_spans":     {"volume": 300_000, "unit_cost": 0.0003},
    "trace_storage_gb": {"volume": 5,       "unit_cost": 0.20},
    "error_tracking":   {"volume": 1_000,   "unit_cost": 0.001},
    "metrics_ingest":   {"volume": 200_000, "unit_cost": 0.00005},
}

# Guardrails (NVIDIA NIM) — teammate formula
# cost = max( R * ( (I + O) * 0.0000015 + 0.0001 ), 50 )
def compute_guardrails_cost(monthly_requests: int, tokens_in: int, tokens_out: int) -> float:
    R = max(0, int(monthly_requests or 0))
    I = float(tokens_in or 0)
    O = float(tokens_out or 0)
    if R <= 0:
        return 50.0
    return round(max(R * ((I + O) * 1.5e-6 + 0.0001), 50.0), 2)

def compute_langsmith_cost() -> float:
    total = 0.0
    for vals in LANGSMITH_DEFAULT.values():
        total += vals["volume"] * vals["unit_cost"]
    return round(total, 2)  # ≈ 102.00

# RAG surfaced even if not present as tasks
# We keep it simple like the old report:
#  - One-time embedding cost (reported, not added to monthly total)
#  - Monthly storage cost (added to monthly)
# Feel free to tweak the constants as your teammate refines them.
RAG_EMBED_COST_PER_GB = 11.70   # one-time $ per GB (MVP average)
RAG_STORAGE_COST_PER_GB = 0.48  # monthly $ per GB (MVP average)

# Merging parameters (input-side only)
MERGE_MODE = "proportional"  # "none" | "proportional"
MERGE_ADDON_RATIO = 0.0
MERGE_ALPHA = 0.25
MERGE_SEP_TOKENS = 5

# Cheaper model fallbacks for MIN when explicit min band missing
MIN_MODEL_FALLBACKS = {
    "google/gemini-1.5-pro": "google/gemini-1.5-flash",
    "mistral/mistral-large-mini": "mistral/mistral-medium",
    "openai/gpt-4o-mini": "openai/gpt-4.1-mini",
}
def _min_model_for(max_model: str, explicit_min: str | None) -> str:
    if explicit_min and str(explicit_min).strip():
        return str(explicit_min).strip()
    m = (max_model or "").strip()
    return MIN_MODEL_FALLBACKS.get(m, m)

# ==============================
# Helpers
# ==============================

def _fmt_money(x) -> str:
    return f"${x:.2f}" if isinstance(x, (int, float)) else ("—" if x is None else str(x))

def _resolve_catalog_path(catalog_path: Path) -> Path:
    if catalog_path.exists():
        return catalog_path
    alt = Path("data/raw") / catalog_path.name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Catalog not found at {catalog_path} or {alt}")

def _price_for_model(model_key: str) -> dict:
    """Return price dict for model; fallback if unavailable/zero."""
    if not model_key:
        return PRICING_DICT.get("openai/gpt-4o-mini")
    p = PRICING_DICT.get(model_key)
    if not p or (p.get("input_per_1M", 0) == 0 and p.get("output_per_1M", 0) == 0):
        if "mistral" in (model_key or ""):
            return PRICING_DICT["mistral/mistral-medium"]
        return PRICING_DICT["openai/gpt-4o-mini"]
    return p

def _apply_compliance(cost: float, compliance: str) -> float:
    if not compliance:
        return cost
    c = str(compliance).lower()
    if c == "gdpr":
        return cost * 1.17
    if c == "hipaa":
        return cost * 1.40
    return cost

def _translation_multiplier(intake: dict) -> float:
    """Single-dir ×1.0; bi-dir ×2.0; auto-detect ×1.2; default ×1.0 when enabled but no arrow."""
    i18n = intake.get("i18n", {}) or {}
    if not i18n.get("translation_required", False):
        return 1.0
    raw = str(i18n.get("language_pairs_raw", "")).lower()
    if "<->" in raw or "↔" in raw:
        return 2.0
    if "auto" in raw:
        return 1.2
    if "->" in raw:
        return 1.0
    return 1.0

def _pick(colnames, options):
    for opt in options:
        if opt in colnames:
            return opt
    return None

def _bucket_from_row(r: dict) -> str:
    """Use parent_category when present; otherwise infer from task name."""
    cat = (r.get("parent_category") or "").strip()
    if cat:
        return cat
    name = (r.get("task") or r.get("task_name") or "").lower()
    if any(k in name for k in ("translation", "tone", "language")):
        return "Translation & Language Adaptation"
    if any(k in name for k in ("corpus embedding", "vector", "upsert", "retrieval", "index", "compaction")):
        return "RAG Retrieval & Indexing"
    if any(k in name for k in ("chatbot", "generation", "text", "tweet", "post", "write")):
        return "Text & Content Generation"
    if any(k in name for k in ("summary", "summarization", "key findings", "research paper", "reasoning")):
        return "Summarization & Reasoning"
    if any(k in name for k in ("monitor", "slo", "latency", "cost", "evaluation", "metrics")):
        return "Evaluation & Monitoring"
    return "Other"

# ==============================
# Core computation
# ==============================

def compute_costs(run_dir: Path, catalog_path: Path) -> dict:
    # Intake + tasks
    intake_path = require_latest_in(run_dir, prefix="intake_", suffix=".json")
    with open(intake_path, "r", encoding="utf-8") as f:
        intake = json.load(f)

    tasks_path = require_latest_in(run_dir, prefix="selected_tasks_", suffix=".json")
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks_json = json.load(f)
    tasks = pd.DataFrame(tasks_json.get("selected_tasks", []))

    # Track whether selected_tasks already has parent_category
    selected_has_category = "parent_category" in tasks.columns and tasks["parent_category"].notna().any()

    # Catalog: load & resolve columns robustly
    cat_path = _resolve_catalog_path(catalog_path)
    xl = pd.ExcelFile(cat_path)
    sheet = "Sheet1" if "Sheet1" in xl.sheet_names else ("Sheet 1" if "Sheet 1" in xl.sheet_names else xl.sheet_names[0])
    cat_df = xl.parse(sheet)
    cat_df.rename(columns=lambda x: str(x).strip(), inplace=True)
    if "id" in cat_df.columns and "task_id" not in cat_df.columns:
        cat_df.rename(columns={"id": "task_id"}, inplace=True)

    cols = set(cat_df.columns)
    parent_col = _pick(cols, [
        "parent_category", "Parent Category", "parent category", "category",
        "parent_cat", "ParentCategory"
    ])
    min_model_col = _pick(cols, [
        "model_band_min_key", "model_band_min", "min_model_key", "min_model",
        "model_min_key"
    ])

    # Normalize join keys to string
    if "task_id" not in tasks.columns:
        raise SystemExit("selected_tasks JSON missing 'task_id'. Re-run task_detect.")
    tasks["task_id"] = tasks["task_id"].astype(str)
    if "task_id" not in cat_df.columns:
        raise SystemExit("Catalog missing 'task_id'/'id'. Check Excel headers.")
    cat_df["task_id"] = cat_df["task_id"].astype(str)

    # Merge (bring in category only if JSON didn't have it; always bring min-model if present)
    merge_cols = ["task_id"]
    rename_map = {}
    if not selected_has_category and parent_col:
        merge_cols.append(parent_col)
        rename_map[parent_col] = "parent_category"
    if min_model_col:
        merge_cols.append(min_model_col)
        rename_map[min_model_col] = "model_band_min_key"

    if len(merge_cols) > 1:
        tasks = tasks.merge(cat_df[merge_cols], on="task_id", how="left")
        if rename_map:
            tasks.rename(columns=rename_map, inplace=True)

    if "parent_category" not in tasks.columns:
        tasks["parent_category"] = None
    if "model_band_min_key" not in tasks.columns:
        tasks["model_band_min_key"] = tasks.get("pricing_model_key", None)

    # Intake-derived variables
    users = int(intake.get("traffic", {}).get("users", 0))
    rpm = int(intake.get("traffic", {}).get("rpm", intake.get("traffic", {}).get("qpm", 0)))
    tokens_in = int(intake.get("traffic", {}).get("tokens_in", 0))
    tokens_out = int(intake.get("traffic", {}).get("tokens_out", 0))
    num_requests = users * rpm
    compliance = intake.get("ops", {}).get("compliance", "")
    translation_mult = _translation_multiplier(intake)

    # Merging: effective per-task input tokens by parent_category (token-basis)
    token_mask = tasks["pricing_basis"].astype(str).str.lower() == "token"
    token_tasks = tasks[token_mask].copy()
    merged_input_by_taskid = {}
    merged_group_size_by_category = {}

    if len(token_tasks) > 0:
        # group by parent_category (fall back to "(uncategorized)" string for NaN)
        for cat, group in token_tasks.groupby(token_tasks["parent_category"].astype(str).replace({"None": "(uncategorized)"})):
            N = len(group)
            merged_group_size_by_category[cat] = N
            I_base = float(tokens_in)  # baseline per-task input (from intake)
            I_eff = merged_group_input_tokens_per_task(
                baseline_input_tokens_per_task=I_base,
                N=N,
                mode=MERGE_MODE,
                addon_ratio=MERGE_ADDON_RATIO,
                alpha=MERGE_ALPHA,
                separator_tokens=MERGE_SEP_TOKENS,
            )
            for _, row in group.iterrows():
                merged_input_by_taskid[str(row["task_id"])] = I_eff

    # Per-task computation (min/max)
    per_task_breakdown = []
    total_min = total_max = 0.0

    for _, r in tasks.iterrows():
        tid = str(r.get("task_id") or "")
        tname = r.get("task_name")
        basis = str(r.get("pricing_basis") or "").lower()
        parent_cat = str(r.get("parent_category") or "")

        if basis not in {"token", "embedding", "api_call"}:
            continue

        if basis == "api_call":
            per_task_breakdown.append({
                "task_id": tid,
                "task": tname, "parent_category": parent_cat or "(uncategorized)", "basis": basis,
                "model_min": "-", "model_max": "-",
                "min_cost": None, "max_cost": None,
                "notes": "API-call pricing TBD"
            })
            continue

        model_max = str(r.get("pricing_model_key") or "").strip()
        model_min = _min_model_for(model_max, r.get("model_band_min_key"))
        price_min = _price_for_model(model_min)
        price_max = _price_for_model(model_max)

        in_tokens_eff = merged_input_by_taskid.get(tid, float(tokens_in))
        out_tokens_eff = float(tokens_out)

        cmin = compute_token_cost(num_requests, in_tokens_eff, out_tokens_eff,
                                  price_min["input_per_1M"], price_min["output_per_1M"])
        cmax = compute_token_cost(num_requests, in_tokens_eff, out_tokens_eff,
                                  price_max["input_per_1M"], price_max["output_per_1M"])

        cmin *= translation_mult
        cmax *= translation_mult

        total_min += cmin
        total_max += cmax

        notes = []
        if translation_mult != 1.0:
            notes.append(f"×{translation_mult} translation factor")
        if tid in merged_input_by_taskid:
            # N for that category (if recorded)
            N = merged_group_size_by_category.get(parent_cat or "(uncategorized)", 1)
            notes.append(f"[MERGED x{N}]")

        per_task_breakdown.append({
            "task_id": tid,
            "task": tname,
            "parent_category": parent_cat or "(uncategorized)",
            "basis": basis,
            "model_min": model_min or "-",
            "model_max": model_max or "-",
            "tokens_in_eff": int(in_tokens_eff),
            "tokens_out_eff": int(out_tokens_eff),
            "min_cost": round(cmin, 2),
            "max_cost": round(cmax, 2),
            "notes": "; ".join(notes)
        })

    # Monitoring (flat monthly)
    mon_cost = compute_langsmith_cost()
    per_task_breakdown.append({
        "task_id": "monitoring",
        "task": "Monitoring (LangSmith average)",
        "parent_category": "Evaluation & Monitoring",
        "basis": "flat",
        "model_min": "-",
        "model_max": "-",
        "tokens_in_eff": 0, "tokens_out_eff": 0,
        "min_cost": mon_cost,
        "max_cost": mon_cost,
        "notes": "Traces+Storage+Errors+Metrics, internal averages"
    })
    total_min += mon_cost
    total_max += mon_cost

    # Guardrails (NVIDIA NIM)
    nim_cost = compute_guardrails_cost(num_requests, tokens_in, tokens_out)
    per_task_breakdown.append({
        "task_id": "guardrails",
        "task": "Guardrails (NVIDIA NIM)",
        "parent_category": "Evaluation & Monitoring",
        "basis": "per-request",
        "model_min": "-",
        "model_max": "-",
        "tokens_in_eff": tokens_in, "tokens_out_eff": tokens_out,
        "min_cost": nim_cost,
        "max_cost": nim_cost,
        "notes": "Formula: max(R*((I+O)*1.5e-6 + 0.0001), 50)"
    })
    total_min += nim_cost
    total_max += nim_cost

    # RAG surfaced (from intake), independent of selected tasks
    rag = intake.get("rag", {}) or {}
    rag_enabled = bool(rag.get("enabled", False))
    rag_one_time = 0.0
    rag_monthly_storage = 0.0
    if rag_enabled:
        corpus_gb = float(rag.get("corpus_gb") or 0.0)
        rag_one_time = round(corpus_gb * RAG_EMBED_COST_PER_GB, 2)
        rag_monthly_storage = round(corpus_gb * RAG_STORAGE_COST_PER_GB, 2)
        # We DO NOT add one-time to monthly totals (display only). Monthly storage is added.
        total_min += rag_monthly_storage
        total_max += rag_monthly_storage

    # -------- Category roll-ups (for report) --------
    roll_min = collections.defaultdict(float)
    roll_max = collections.defaultdict(float)
    for r in per_task_breakdown:
        bkt = _bucket_from_row(r)
        vmin = r.get("min_cost"); vmax = r.get("max_cost")
        if isinstance(vmin, (int, float)): roll_min[bkt] += float(vmin)
        if isinstance(vmax, (int, float)): roll_max[bkt] += float(vmax)
    # Add RAG storage monthly to the RAG bucket, if enabled
    if rag_enabled and rag_monthly_storage > 0:
        roll_min["RAG Retrieval & Indexing"] += rag_monthly_storage
        roll_max["RAG Retrieval & Indexing"] += rag_monthly_storage

    # Compliance on totals
    total_min_after = _apply_compliance(total_min, compliance)
    total_max_after = _apply_compliance(total_max, compliance)

    summary = {
        "requests_per_month": num_requests,
        "rag_one_time_embedding": rag_one_time,
        "rag_monthly_storage": rag_monthly_storage,
        "total_min_before_compliance": round(total_min, 2),
        "total_max_before_compliance": round(total_max, 2),
        "compliance": compliance or "None",
        "total_min_after_compliance": round(total_min_after, 2),
        "total_max_after_compliance": round(total_max_after, 2),
        "translation_multiplier": translation_mult,
        "monitoring_cost": mon_cost,
        "guardrails_cost": nim_cost,
    }
    rollups = {"min": dict(roll_min), "max": dict(roll_max)}
    return {
        "breakdown": per_task_breakdown,
        "rollups": rollups,
        "summary": summary,
        "meta": {
            "merge": {"mode": MERGE_MODE, "alpha": MERGE_ALPHA, "sep_tokens": MERGE_SEP_TOKENS},
            "rag_enabled": rag_enabled,
        }
    }

# ==============================
# CLI
# ==============================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="Run directory created by task_detect (default: latest under data/out)")
    ap.add_argument("--catalog", default="AI_Cost_Estimator_Catalog_UPDATED.xlsx",
                    help="Path to catalog for min/max model lookup (root or data/raw)")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    catalog_path = Path(args.catalog)

    result = compute_costs(run_dir, catalog_path)

    out_json = next_out_path(run_dir, "cost_breakdown", "json")
    out_md = next_out_path(run_dir, "cost_summary", "md")

    # -------- JSON (full per-task breakdown + rollups + summary) --------
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # -------- Markdown (clean, business-style) --------
    s = result["summary"]
    roll_min = result["rollups"]["min"]
    roll_max = result["rollups"]["max"]

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Cost Report — {datetime.now().strftime('%Y-%m-%d')}\n\n")

        # Top summary — like the old report
        f.write(f"- Requests/month: **{s['requests_per_month']:,}**\n")
        f.write(f"- RAG one-time embeddings: **{_fmt_money(s['rag_one_time_embedding'])}**\n")
        f.write(f"- RAG monthly storage: **{_fmt_money(s['rag_monthly_storage'])}**\n")
        f.write(f"- Monthly MIN: **{_fmt_money(s['total_min_before_compliance'])}**\n")
        f.write(f"- Monthly MAX: **{_fmt_money(s['total_max_before_compliance'])}**\n\n")

        # Service rollups — MIN
        f.write("### Services & Cost (MIN)\n\n")
        f.write("| Service | Monthly |\n|---|---:|\n")
        for cat in sorted(roll_min.keys()):
            f.write(f"| {cat} | {_fmt_money(roll_min[cat])} |\n")
        f.write("\n")

        # Service rollups — MAX
        f.write("### Services & Cost (MAX)\n\n")
        f.write("| Service | Monthly |\n|---|---:|\n")
        for cat in sorted(roll_max.keys()):
            f.write(f"| {cat} | {_fmt_money(roll_max[cat])} |\n")
        f.write("\n")

        # MIN plan table
        f.write("### MIN Plan\n\n")
        f.write("| Category | Task | TokIn | TokOut | Monthly |\n|---|---|---:|---:|---:|\n")
        for r in result["breakdown"]:
            if isinstance(r.get("min_cost"), (int, float)):
                f.write(f"| {_bucket_from_row(r)} | {r['task']} {(' ' + r['notes']) if r.get('notes') else ''} | "
                        f"{r.get('tokens_in_eff', 0)} | {r.get('tokens_out_eff', 0)} | {_fmt_money(r['min_cost'])} |\n")
        f.write("\n")

        # MAX plan table
        f.write("### MAX Plan\n\n")
        f.write("| Category | Task | TokIn | TokOut | Monthly |\n|---|---|---:|---:|---:|\n")
        for r in result["breakdown"]:
            if isinstance(r.get("max_cost"), (int, float)):
                f.write(f"| {_bucket_from_row(r)} | {r['task']} {(' ' + r['notes']) if r.get('notes') else ''} | "
                        f"{r.get('tokens_in_eff', 0)} | {r.get('tokens_out_eff', 0)} | {_fmt_money(r['max_cost'])} |\n")
        f.write("\n")

        # Totals & footnotes
        f.write("### Totals\n\n")
        if s['compliance'] and str(s['compliance']).lower() != "none":
            f.write(f"- Before compliance: {_fmt_money(s['total_min_before_compliance'])} – {_fmt_money(s['total_max_before_compliance'])}\n")
            f.write(f"- After {s['compliance']} surcharge: **{_fmt_money(s['total_min_after_compliance'])} – {_fmt_money(s['total_max_after_compliance'])}**\n")
        else:
            f.write(f"- Final total: **{_fmt_money(s['total_min_after_compliance'])} – {_fmt_money(s['total_max_after_compliance'])}**\n")
        f.write(f"- Monitoring (LangSmith): {_fmt_money(s['monitoring_cost'])}/month\n")
        f.write(f"- Guardrails (NVIDIA NIM): {_fmt_money(s['guardrails_cost'])}/month\n")
        if s["translation_multiplier"] != 1.0:
            f.write(f"- Translation multiplier applied: ×{s['translation_multiplier']}\n")
        f.write("\n> Notes: Category costs reflect merged prompts within each category. One-time RAG embedding cost is not added to monthly totals.\n")

    print("✅ Wrote:")
    print(f"   {out_json}")
    print(f"   {out_md}")
    print(f"   Run directory: {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
