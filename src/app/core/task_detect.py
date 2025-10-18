#!/usr/bin/env python3
"""
task_detect.py — Step 2: Detect candidate tasks from the intake brief using the 200-row catalog.

Inputs (default behavior):
  - data/out/<run>/intake_*.json   (latest in the chosen run dir if not provided)
  - data/raw/AI_Cost_Estimator_Task_Catalog_200rows_v2.csv

Outputs:
  - data/out/<run>/detected_<timestamp>.json

Stdlib only (csv/json). Relative imports only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# 🔧 relative, UI-agnostic helpers (run folders + paths)
from ..utils.io_paths import (
    RAW_DIR,                 # data/raw (catalog)
    resolve_run_dir,         # pick a run dir (explicit or latest)
    require_latest_in,       # get latest intake_* in that run
    next_out_path,           # build detected_<ts>.json in same run
)

CATALOG_FILENAME = "AI_Cost_Estimator_Task_Catalog_200rows_v2.csv"

STOPWORDS = {
    "the","a","an","and","or","of","to","for","with","on","in","by","at","from","that","this","it","its",
    "be","is","are","as","into","over","while","when","then","than","we","our","your","their","you","i",
    "app","application","build","create","develop","make","do"
}

# Fields we consider "globally satisfied" by intake (don’t ask again)
GLOBAL_FIELDS = {"users","qpm"}

# Per-merge_group fields we may ask if missing in the row AND not covered by intake
GROUP_REQUIRED_FIELDS = {
    "S0_clean":        ["tokens_in","tokens_out"],
    "S0_translate":    ["tokens_in","tokens_out","language_pairs"],
    "S0_ie":           ["tokens_in","tokens_out","classes","schema_fields"],
    "S1_retrieve":     ["corpus_gb","embed_model","refresh_rate","avg_chunk_tokens"],
    "S1_db":           ["num_queries","avg_query_tokens","source_type"],
    "S2_textgen":      ["tokens_in","tokens_out"],
    "S2_code":         ["tokens_in","tokens_out","complexity_level"],
    "S3_analysis":     ["tokens_in","tokens_out","classes"],
    "S3_summary":      ["tokens_in","tokens_out"],
    "S3_domain":       ["tokens_in","tokens_out","domain_flag"],
    "S4_agentic":      ["num_tools","chain_depth","avg_steps","tokens_in","tokens_out"],
    "S4_eval":         ["eval_freq","metrics","tokens_in","tokens_out"],
}

# Signals derived from intake that should boost certain groups
INTAKE_SIGNALS = [
    ("rag.enabled", True,  ("S1_retrieve", 3.0)),
    ("i18n.translation_required", True, ("S0_translate", 2.5)),
    ("agentic.enabled", True, ("S4_agentic", 3.0)),
]

# ---------------- core helpers ----------------

def load_intake(intake_path: Path) -> Dict[str, Any]:
    with intake_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_catalog() -> List[Dict[str, Any]]:
    cat_path = RAW_DIR / CATALOG_FILENAME
    if not cat_path.exists():
        raise SystemExit(f"Catalog not found: {cat_path}")
    rows: List[Dict[str, Any]] = []
    with cat_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9\+\-→_]+", text)
    return [t for t in tokens if t and t not in STOPWORDS]

def row_keywords(row: Dict[str, Any]) -> List[str]:
    parts = f"{row.get('task_name','')} {row.get('parent_category','')} {row.get('merge_group','')}"
    return tokenize(parts)

def score_row(app_tokens: List[str], row_tokens: List[str]) -> float:
    hits = 0.0
    row_set = set(row_tokens)
    for t in app_tokens:
        if t in row_set:
            hits += 1.0
    return hits / (len(row_set) + 1e-6)

def get_field(d: Dict[str, Any], path: str, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def apply_intake_signals(intake: Dict[str, Any], base_boosts: Dict[str, float]) -> Dict[str, float]:
    boosts = dict(base_boosts)
    for path, val, (merge_group, weight) in INTAKE_SIGNALS:
        if get_field(intake, path, None) == val:
            boosts[merge_group] = boosts.get(merge_group, 0.0) + weight
    return boosts

def propose_tasks(intake: Dict[str, Any], catalog: List[Dict[str, Any]], top_k_per_group: int = 3) -> List[Dict[str, Any]]:
    app_brief = intake.get("app_brief", "")
    app_tokens = tokenize(app_brief)

    scored: List[Tuple[float, Dict[str, Any], List[str]]] = []
    for row in catalog:
        rk = row_keywords(row)
        s = score_row(app_tokens, rk)
        reasons = []
        if s > 0:
            reasons.append("keyword overlap: " + ", ".join(sorted(set(app_tokens).intersection(rk))[:6]))
        scored.append((s, row, reasons))

    by_group: Dict[str, List[Tuple[float, Dict[str, Any], List[str]]]] = {}
    for s, row, reasons in scored:
        mg = row.get("merge_group", "")
        by_group.setdefault(mg, []).append((s, row, reasons))

    boosts = apply_intake_signals(intake, {})
    proposals: List[Dict[str, Any]] = []
    for mg, lst in by_group.items():
        lst.sort(key=lambda x: x[0], reverse=True)
        chosen = lst[:top_k_per_group]
        for s, row, reasons in chosen:
            conf = s + boosts.get(mg, 0.0)
            proposals.append({
                "id": int(row.get("id", 0)) if str(row.get("id","")).isdigit() else row.get("id"),
                "stage": row.get("stage"),
                "parent_category": row.get("parent_category"),
                "task_name": row.get("task_name"),
                "merge_group": mg,
                "merge_compatible": str(row.get("merge_compatible","")).lower() in ("true","1","yes"),
                "confidence": round(conf, 3),
                "reasons": reasons,
                "model_profile": {
                    "key": row.get("model_profile.key"),
                    "tier": row.get("model_profile.tier"),
                    "context_window": row.get("model_profile.context_window"),
                }
            })

    proposals = [p for p in proposals if p["confidence"] > 0]

    seen = set()
    unique: List[Dict[str, Any]] = []
    for p in sorted(proposals, key=lambda z: (-z["confidence"], z["stage"], z["task_name"] or "")):
        key = (p["merge_group"], p["task_name"])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique

def missing_fields_for_task(intake: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    mg = row.get("merge_group")
    needed = GROUP_REQUIRED_FIELDS.get(mg, [])
    missing = []
    intake_map = {
        "users": get_field(intake, "traffic.users"),
        "qpm": get_field(intake, "traffic.qpm"),
        "language_pairs": get_field(intake, "i18n.language_pairs"),
        "corpus_gb": get_field(intake, "rag.corpus_gb"),
        "embed_model": get_field(intake, "rag.embed_model"),
        "refresh_rate": get_field(intake, "rag.refresh_rate"),
        "avg_chunk_tokens": get_field(intake, "rag.avg_chunk_tokens"),
        "num_tools": get_field(intake, "agentic.num_tools"),
        "chain_depth": get_field(intake, "agentic.chain_depth"),
        "avg_steps": get_field(intake, "agentic.avg_steps"),
    }
    for f in needed:
        if f in GLOBAL_FIELDS:
            continue
        if intake_map.get(f) not in (None, "", []):
            continue
        val = row.get(f)
        if val is None or str(val).strip() == "" or str(val).lower() == "nan":
            missing.append(f)
    return missing

def attach_missing_fields(intake: Dict[str, Any], catalog: List[Dict[str, Any]], proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(r.get("id")): r for r in catalog}
    enriched = []
    for p in proposals:
        rid = str(p.get("id"))
        row = by_id.get(rid, {})
        p = dict(p)
        p["ask_for"] = missing_fields_for_task(intake, row)
        enriched.append(p)
    return enriched

# ---------------- main ----------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detect tasks from an intake JSON (per-run)")
    parser.add_argument("--run-dir", help="Run directory (e.g., data/out/acme-20251018_123456)")
    parser.add_argument("-i", "--input", help="Explicit path to intake_*.json")
    parser.add_argument("--topk", type=int, default=2, help="Top-k rows per group")
    args = parser.parse_args(argv)

    # pick a run dir (explicit or latest)
    run_dir: Path = resolve_run_dir(args.run_dir)

    # pick intake file: explicit path, else latest intake_* in run_dir
    if args.input:
        intake_path = Path(args.input)
        if not intake_path.exists():
            raise SystemExit(f"Intake file not found: {intake_path}")
    else:
        intake_path = require_latest_in(run_dir, "intake_", ".json",
                                        f"No intake_*.json in {run_dir}. Run cli_intake first.")

    intake = load_intake(intake_path)
    catalog = load_catalog()

    proposals = propose_tasks(intake, catalog, top_k_per_group=args.topk)
    proposals = attach_missing_fields(intake, catalog, proposals)

    # write detected into the SAME run folder, reusing intake timestamp for easy grouping
    out_path = next_out_path(run_dir, "detected", "json", reuse_ts_from=intake_path)
    merged = dict(intake)
    merged["detected_tasks"] = proposals
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # Console summary
    print("\n=== Proposed Tasks (grouped) ===")
    by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for p in proposals:
        by_stage.setdefault(p["stage"], []).append(p)
    for stage in sorted(by_stage.keys()):
        print(f"\n{stage}:")
        for p in by_stage[stage]:
            mg = p["merge_group"]
            print(f"  - [{mg}] {p['task_name']}  (conf={p['confidence']})  ask_for={p['ask_for']}")

    print(f"\n✅ Saved merged intake + proposals → {out_path}\n")
    print("Next:")
    print(f"  • Review/confirm tasks, or prune/add tasks manually in: {out_path}")
    print(f"  • Then run planning & costing on this run dir.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
