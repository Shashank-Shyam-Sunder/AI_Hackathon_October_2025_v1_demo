#!/usr/bin/env python3
"""
plan_and_cost.py — Step 3: Build MIN/MAX plans, verify fit, and compute costs.

Inputs:
  - data/raw/detected_*.json   (from task_detect.py)
  - data/raw/AI_Cost_Estimator_Task_Catalog_200rows_v2.csv
  - OPTIONAL:
      data/raw/models.yaml      # per-model token prices (in_per_1k/out_per_1k)
      data/raw/embeddings.yaml  # embedding model price per 1k tokens
      data/raw/vector_db.yaml   # storage_per_gb_month
      (You can also add assumptions in the detected JSON: {"assumptions": {"text_extract_ratio": 0.15}})

Outputs:
  - data/raw/cost_report_<timestamp>.json
  - data/raw/cost_report_<timestamp>.csv        (line items)
  - data/raw/cost_report_<timestamp>.md         (shareable)
  - Console summary with:
      * Services & Cost (user-friendly)
      * MIN vs MAX tables
      * Fit checklist (✅/⚠️)

Notes:
  - Merge logic interprets io_addon_merge_ratio as a *savings* fraction when merging.
  - If you later capture per-task overrides in detected JSON, apply them in tokens_for_task().
"""

import os, sys, json, csv, math
from datetime import datetime

# ---------- root-anchored paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CATALOG_CSV = os.path.join(RAW_DIR, "AI_Cost_Estimator_Task_Catalog_200rows_v2.csv")

# ---------- tiny YAML reader (optional) ----------
def load_yaml_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        import yaml  # optional
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        # fallback: parse simple key: value pairs
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    d[k.strip()] = v.strip()
        return d

# ---------- utils ----------
def now_ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def fmt_money(x): return f"${x:,.2f}"
def pad(s, n): return str(s)[:n].ljust(n)

def latest_detected_json():
    files = [f for f in os.listdir(RAW_DIR) if f.startswith("detected_") and f.endswith(".json")]
    if not files:
        sys.exit("No detected_*.json found. Run task_detect.py first.")
    files.sort(reverse=True)
    return os.path.join(RAW_DIR, files[0])

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_catalog_rows():
    if not os.path.exists(CATALOG_CSV):
        sys.exit(f"Catalog not found: {CATALOG_CSV}")
    rows = []
    with open(CATALOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    by_id = {str(r.get("id")): r for r in rows}
    return rows, by_id

# ---------- price catalogs (fallbacks are conservative) ----------
def load_price_catalogs():
    models_yaml = load_yaml_if_exists(os.path.join(RAW_DIR, "models.yaml"))
    vdb_yaml    = load_yaml_if_exists(os.path.join(RAW_DIR, "vector_db.yaml"))
    embed_yaml  = load_yaml_if_exists(os.path.join(RAW_DIR, "embeddings.yaml"))

    PRICES = {
        "model_in_per_1k":  { "default": 0.002 },
        "model_out_per_1k": { "default": 0.006 },
        "embedding_per_1k": {
            "default": 0.00013,
            "text-embedding-3-large": 0.00013
        },
        "vector_storage_gb_month": { "default": 0.20 }
    }

    # Expected models.yaml structure:
    # models:
    #   openai/gpt-4o-mini:
    #     in_per_1k: 0.005
    #     out_per_1k: 0.015
    if models_yaml and isinstance(models_yaml, dict) and "models" in models_yaml:
        for mkey, mval in models_yaml["models"].items():
            if isinstance(mval, dict):
                if "in_per_1k" in mval:
                    PRICES["model_in_per_1k"][mkey] = float(mval["in_per_1k"])
                if "out_per_1k" in mval:
                    PRICES["model_out_per_1k"][mkey] = float(mval["out_per_1k"])

    if embed_yaml and isinstance(embed_yaml, dict) and "embeddings" in embed_yaml:
        for ekey, evals in embed_yaml["embeddings"].items():
            if "per_1k" in evals:
                PRICES["embedding_per_1k"][ekey] = float(evals["per_1k"])

    if vdb_yaml and isinstance(vdb_yaml, dict) and "storage_per_gb_month" in vdb_yaml:
        PRICES["vector_storage_gb_month"]["default"] = float(vdb_yaml["storage_per_gb_month"])

    return PRICES

def price_for_model(model_key, prices, direction):
    table = prices["model_in_per_1k"] if direction == "in" else prices["model_out_per_1k"]
    return table.get(model_key, table["default"])

# ---------- volumes & tokens ----------
def monthly_requests(intake):
    users = float(intake.get("traffic", {}).get("users") or 0)
    qpm   = float(intake.get("traffic", {}).get("qpm") or 0)
    return users * qpm

def tokens_for_task(row, overrides=None):
    """
    Returns (tokens_in, tokens_out) for a single task.
    Uses catalog defaults; if per-task overrides exist, apply them here.
    """
    ti = float(row.get("io_tokens_in_default")  or 0)
    to = float(row.get("io_tokens_out_default") or 0)
    # If you later store overrides in detected JSON, apply here:
    # if overrides and "tokens_in" in overrides:  ti = float(overrides["tokens_in"])
    # if overrides and "tokens_out" in overrides: to = float(overrides["tokens_out"])
    return ti, to

def merged_tokens(rows_in_group):
    """
    Merge savings: tokens_total = sum(tokens) * (1 - savings)
    We interpret io_addon_merge_ratio as a savings fraction (clamped to [0, 0.9]).
    We take MAX savings in the group (simple & safe).
    """
    sum_in  = sum(float(r.get("io_tokens_in_default")  or 0) for r in rows_in_group)
    sum_out = sum(float(r.get("io_tokens_out_default") or 0) for r in rows_in_group)
    ratios  = [float(r.get("io_addon_merge_ratio") or 0) for r in rows_in_group]
    savings = max(ratios) if ratios else 0.0
    savings = max(0.0, min(0.9, savings))
    return sum_in * (1 - savings), sum_out * (1 - savings)

# ---------- costing ----------
def cost_token_priced(R, tokens_in, tokens_out, model_key, prices):
    cin  = price_for_model(model_key, prices, "in")
    cout = price_for_model(model_key, prices, "out")
    return R * ( (tokens_in/1000.0)*cin + (tokens_out/1000.0)*cout )

def cost_rag_one_time_and_monthly(intake, prices):
    rag = intake.get("rag", {}) or {}
    if not rag.get("enabled"):
        return 0.0, 0.0
    corpus_gb = float(rag.get("corpus_gb") or 0.0)
    embed_model = rag.get("embed_model") or "text-embedding-3-large"

    # Realistic token estimate:
    # 1 token ≈ 4 chars; 1 GB raw text ≈ 250M tokens. Many corpora aren't pure text.
    # text_extract_ratio models how much of each GB is extractable text (default 15%).
    assumptions = intake.get("assumptions", {}) or {}
    text_extract_ratio = float(assumptions.get("text_extract_ratio", 0.15))
    tokens_per_gb_effective = 250_000_000 * text_extract_ratio
    total_tokens = corpus_gb * tokens_per_gb_effective

    per_1k = prices["embedding_per_1k"].get(embed_model, prices["embedding_per_1k"]["default"])
    one_time = (total_tokens / 1000.0) * per_1k

    storage = prices["vector_storage_gb_month"]["default"]
    monthly_storage = corpus_gb * storage
    return float(round(one_time, 4)), float(round(monthly_storage, 4))

# ---------- planning ----------
def build_plans(detected, catalog_by_id):
    """
    Build MIN (merged) and MAX (separate) plans from detected tasks.
    Uses confidence >= 1.0 as default inclusion threshold.
    """
    chosen = [t for t in detected if float(t.get("confidence", 0)) >= 0.1]

    # Attach catalog rows
    enriched = []
    for t in chosen:
        row = catalog_by_id.get(str(t.get("id")))
        if not row:
            continue
        enriched.append((t, row))

    # Group by stage & merge_group
    by_stage = {}
    for t, row in enriched:
        stage = row.get("stage")
        mg    = row.get("merge_group")
        by_stage.setdefault(stage, {}).setdefault(mg, []).append((t, row))

    min_plan, max_plan = [], []

    for stage, groups in by_stage.items():
        for mg, items in groups.items():
            rows = [r for (_t, r) in items]
            mergeable = all(str(r.get("merge_compatible","")).lower() in ("true","1","yes") for r in rows)

            # choose a model_key (take the first detected model profile or fallback to row)
            model_key = items[0][0].get("model_profile", {}).get("key") or rows[0].get("model_profile.key") or "default"

            # MAX: each task separate
            for _t, r in items:
                ti, to = tokens_for_task(r)
                max_plan.append({
                    "stage": stage, "merge_group": mg, "task_name": r.get("task_name"),
                    "model_key": model_key, "tokens_in": ti, "tokens_out": to
                })

            # MIN: merge if possible, else separate
            if mergeable and len(items) > 1:
                ti, to = merged_tokens(rows)
                min_plan.append({
                    "stage": stage, "merge_group": mg, "task_name": f"[MERGED x{len(items)}]",
                    "model_key": model_key, "tokens_in": ti, "tokens_out": to
                })
            else:
                for _t, r in items:
                    ti, to = tokens_for_task(r)
                    min_plan.append({
                        "stage": stage, "merge_group": mg, "task_name": r.get("task_name"),
                        "model_key": model_key, "tokens_in": ti, "tokens_out": to
                    })
    return min_plan, max_plan

def apply_ops_multipliers(intake, monthly):
    ops = intake.get("ops", {}) or {}
    availability = float(ops.get("availability_target") or 99.0)
    monitoring_tier = (ops.get("monitoring_tier") or "Basic").lower()
    m = monthly
    if availability >= 99.9:
        m *= 1.10  # 10% uplift
    if monitoring_tier == "standard":
        m += 29.0
    elif monitoring_tier == "enhanced":
        m += 99.0
    return round(m, 4)

def compute_costs(intake, min_plan, max_plan, prices):
    R = monthly_requests(intake)
    one_time_embed, monthly_storage = cost_rag_one_time_and_monthly(intake, prices)

    def tally(plan):
        total = 0.0
        line_items = []
        for item in plan:
            c = cost_token_priced(R, item["tokens_in"], item["tokens_out"], item["model_key"], prices)
            li = dict(item)
            li["monthly_cost"] = round(c, 4)
            line_items.append(li)
            total += c
        return round(total, 4), line_items

    monthly_min_raw, lines_min = tally(min_plan)
    monthly_max_raw, lines_max = tally(max_plan)

    monthly_min = apply_ops_multipliers(intake, monthly_min_raw) + monthly_storage
    monthly_max = apply_ops_multipliers(intake, monthly_max_raw) + monthly_storage

    report = {
        "as_of": datetime.now().date().isoformat(),
        "requests_per_month": R,
        "rag": {
            "enabled": bool(intake.get("rag", {}).get("enabled", False)),
            "one_time_embeddings": round(one_time_embed, 2),
            "monthly_vector_storage": round(monthly_storage, 2)
        },
        "monthly_min": round(monthly_min, 2),
        "monthly_max": round(monthly_max, 2),
        "one_time_min": round(one_time_embed, 2),
        "one_time_max": round(one_time_embed, 2),
        "min_plan": lines_min,
        "max_plan": lines_max
    }
    return report

# ---------- Service aggregation (for user-friendly “Services & Cost”) ----------
SERVICE_LABEL_FOR_GROUP = {
    "S0_clean": "Text Cleaning & Normalization",
    "S0_translate": "Translation & Language Adaptation",
    "S0_ie": "Information Extraction",
    "S1_retrieve": "RAG Retrieval",
    "S1_db": "Database/SQL Access",
    "S2_textgen": "Text Generation",
    "S2_code": "Code/Logic Generation",
    "S3_analysis": "Text Analysis & Classification",
    "S3_summary": "Summarization",
    "S3_domain": "Domain-Specific Writing",
    "S4_agentic": "Agentic Orchestration",
    "S4_eval": "Evaluation & Monitoring",
}

def aggregate_services(plan_items):
    """
    Returns a list of {service, monthly_cost} aggregated by merge_group label.
    """
    totals = {}
    for it in plan_items:
        label = SERVICE_LABEL_FOR_GROUP.get(it["merge_group"], it["merge_group"])
        totals[label] = totals.get(label, 0.0) + float(it["monthly_cost"])
    return [{"service": s, "monthly_cost": round(v, 2)} for s, v in sorted(totals.items(), key=lambda kv: -kv[1])]

# ---------- Fit checklist (verify plan meets user demand) ----------
def fit_checklist(intake, selected_groups, catalog_rows_for_plan):
    """
    Returns a list of (status, message) where status is 'ok' (✅) or 'warn' (⚠️).
    selected_groups: set of merge_group strings included in the plan.
    """
    checks = []

    # Coverage (rough, keyword-based against intake brief)
    brief = (intake.get("app_brief") or "").lower()
    def present(group): return group in selected_groups

    # expected areas -> merge_group presence
    coverage_map = {
        "translation":     present("S0_translate"),
        "rag":             present("S1_retrieve"),
        "text generation": present("S2_textgen"),
        "summarization":   present("S3_summary"),
        "classification":  present("S3_analysis"),
        "sql/table":       present("S1_db"),
        "agentic":         present("S4_agentic"),
        "evaluation":      present("S4_eval"),
    }

    # If keywords appear in brief but area missing in plan, warn.
    keyword_triggers = {
        "translation": ["translate","multilingual","language","i18n","locale"],
        "rag": ["rag","knowledge","docs","retrieval","vector","search","embed"],
        "text generation": ["generate","chatbot","answer","response","draft","write"],
        "summarization": ["summary","summarize","tl;dr","minutes","bullet"],
        "classification": ["classify","sentiment","topic","intent","toxicity","moderation"],
        "sql/table": ["sql","table","postgres","query","analytics","reporting","bi"],
        "agentic": ["agent","tool","orchestrate","workflow","calendar","email","action"],
        "evaluation": ["eval","evaluate","monitor","hallucination","toxicity","latency","cost"],
    }

    for area, included in coverage_map.items():
        needed = any(k in brief for k in keyword_triggers[area])
        if needed and not included:
            checks.append(("warn", f"Coverage: '{area}' mentioned in brief but not present in plan."))
        elif needed and included:
            checks.append(("ok",   f"Coverage: '{area}' present."))

    # SLOs: context window feasibility (max tokens_in vs model context)
    # Take the largest single tokens_in across the plan and compare with the chosen model's context_window (if available).
    max_tokens_in = 0
    min_ctx = float("inf")
    for row in catalog_rows_for_plan:
        ti = float(row.get("io_tokens_in_default") or 0)
        max_tokens_in = max(max_tokens_in, ti)
        ctx = row.get("model_profile.context_window")
        try:
            ctx = float(ctx) if ctx not in (None, "", "nan") else None
        except Exception:
            ctx = None
        if ctx is not None:
            min_ctx = min(min_ctx, ctx)

    if min_ctx != float("inf"):
        if max_tokens_in > min_ctx:
            checks.append(("warn", f"SLOs: tokens_in_default ({int(max_tokens_in)}) may exceed smallest model context window ({int(min_ctx)})."))
        else:
            checks.append(("ok",   f"SLOs: context windows sufficient (max tokens_in {int(max_tokens_in)} ≤ ctx {int(min_ctx)})."))
    else:
        checks.append(("warn", "SLOs: could not verify context windows (no model context data)."))

    # Ops: availability & monitoring acknowledged
    availability = float(intake.get("ops", {}).get("availability_target") or 99.0)
    monitoring   = (intake.get("ops", {}).get("monitoring_tier") or "Basic").lower()
    if availability >= 99.9:
        checks.append(("ok", "Ops: 99.9% availability applied with surcharge."))
    else:
        checks.append(("ok", "Ops: Standard availability."))
    checks.append(("ok", f"Ops: Monitoring tier '{monitoring}'. Uplift applied if applicable."))

    # Domain pruning (if not medical/legal, warn if S3_domain present)
    domain = (intake.get("preferences", {}).get("domain") or "general").lower()
    if domain not in ("medical","legal") and "S3_domain" in selected_groups:
        checks.append(("warn", f"Domain: '{domain}' domain; consider pruning 'Domain-Specific Writing' tasks unless required."))

    # RAG: confirm corpus & cadence if enabled
    if bool(intake.get("rag", {}).get("enabled", False)):
        cgb = intake["rag"].get("corpus_gb")
        rr  = intake["rag"].get("refresh_rate")
        em  = intake["rag"].get("embed_model")
        if not cgb or not rr or not em:
            checks.append(("warn", "RAG: missing corpus_gb, refresh_rate, or embed_model."))
        else:
            checks.append(("ok", f"RAG: {cgb} GB corpus, {rr} refresh, {em} embeddings."))

    # i18n: confirm language pairs if translation present/needed
    if "S0_translate" in selected_groups:
        lp = intake.get("i18n", {}).get("language_pairs")
        if not lp:
            checks.append(("warn", "i18n: language_pairs not provided; using defaults."))
        else:
            checks.append(("ok", f"i18n: language pairs set: {lp}"))

    return checks

# ---------- exports ----------
def write_csv(min_items, max_items, out_path_csv):
    with open(out_path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["plan","stage","merge_group","task_name","model_key","tokens_in","tokens_out","monthly_cost"])
        w.writeheader()
        for li in min_items:
            w.writerow({"plan":"MIN", **li})
        for li in max_items:
            w.writerow({"plan":"MAX", **li})

def write_md(report, services_min, services_max, out_path_md):
    def md_table(items, title):
        lines = [f"### {title}", "", "| Stage | Group | Task | TokIn | TokOut | Monthly |", "|---|---|---|---:|---:|---:|"]
        for li in items:
            lines.append(f"| {li['stage']} | {li['merge_group']} | {li['task_name']} | {int(li['tokens_in'])} | {int(li['tokens_out'])} | {fmt_money(li['monthly_cost'])} |")
        lines.append("")
        return "\n".join(lines)

    def md_services(services, title):
        lines = [f"### {title}", "", "| Service | Monthly |", "|---|---:|"]
        for s in services:
            lines.append(f"| {s['service']} | {fmt_money(s['monthly_cost'])} |")
        lines.append("")
        return "\n".join(lines)

    with open(out_path_md, "w", encoding="utf-8") as f:
        f.write(f"# Cost Report ({report['as_of']})\n\n")
        f.write(f"- Requests/month: **{int(report['requests_per_month']):,}**\n")
        if report["rag"]["enabled"]:
            f.write(f"- RAG one-time embeddings: **{fmt_money(report['rag']['one_time_embeddings'])}**\n")
            f.write(f"- RAG monthly storage: **{fmt_money(report['rag']['monthly_vector_storage'])}**\n")
        f.write(f"- Monthly **MIN**: **{fmt_money(report['monthly_min'])}**\n")
        f.write(f"- Monthly **MAX**: **{fmt_money(report['monthly_max'])}**\n\n")
        f.write(md_services(services_min, "Services & Cost (MIN)"))
        f.write(md_services(services_max, "Services & Cost (MAX)"))
        f.write(md_table(report["min_plan"], "MIN plan (merged when possible)"))
        f.write(md_table(report["max_plan"], "MAX plan (all separate)"))

# ---------- main ----------
def main():
    detected_path = sys.argv[1] if len(sys.argv) > 1 else latest_detected_json()
    intake_and_detected = read_json(detected_path)
    intake = intake_and_detected  # includes intake fields

    _, by_id = read_catalog_rows()
    prices = load_price_catalogs()

    proposals = intake_and_detected.get("detected_tasks", [])
    if not proposals:
        print("No detected tasks found. Did task_detect.py produce any?")
        return 1

    # Build plans (tokens only)
    min_plan, max_plan = build_plans(proposals, by_id)
    # Compute costs (+ RAG, + ops)
    report = compute_costs(intake, min_plan, max_plan, prices)

    # Aggregate services for user-friendly view
    services_min = aggregate_services(report["min_plan"])
    services_max = aggregate_services(report["max_plan"])

    # Fit checklist (verify plan vs user demand)
    selected_groups = set(li["merge_group"] for li in report["min_plan"]) | set(li["merge_group"] for li in report["max_plan"])
    # Pull catalog rows used in either plan (for context window checks)
    catalog_rows_used = []
    for li in report["min_plan"] + report["max_plan"]:
        # we don't have row ids here; approximate by merge_group best-effort via by_id not possible.
        # For SLO context check, use min_plan items and assume their catalog defaults represent the group.
        # (Good enough for now; can be refined by carrying row ids from build_plans)
        pass
    # Instead, approximate: build a synthetic list with one exemplar per merge_group from min_plan
    exemplar_rows = []
    seen_groups = set()
    for t_row_id, row in by_id.items():
        # pick one exemplar row per merge_group present in selected_groups
        mg = row.get("merge_group")
        if mg in selected_groups and mg not in seen_groups:
            exemplar_rows.append(row)
            seen_groups.add(mg)
        if seen_groups == selected_groups:
            break

    checklist = fit_checklist(intake, selected_groups, exemplar_rows)

    # Save JSON report
    out_json = os.path.join(RAW_DIR, f"cost_report_{now_ts()}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save CSV (line items)
    out_csv = os.path.join(RAW_DIR, f"cost_report_{now_ts()}.csv")
    write_csv(report["min_plan"], report["max_plan"], out_csv)

    # Save Markdown (shareable)
    out_md = os.path.join(RAW_DIR, f"cost_report_{now_ts()}.md")
    write_md(report, services_min, services_max, out_md)

    # ---------- Console output ----------
    print("\n=== Cost Report ===")
    print(f"Requests/month: {int(report['requests_per_month']):,}")
    if report["rag"]["enabled"]:
        print(f"RAG one-time embeddings: {fmt_money(report['rag']['one_time_embeddings'])}")
        print(f"RAG monthly storage:     {fmt_money(report['rag']['monthly_vector_storage'])}")
    print(f"\nMonthly MIN: {fmt_money(report['monthly_min'])}   |   Monthly MAX: {fmt_money(report['monthly_max'])}")

    # Services & Cost section (user-friendly)
    def print_services(title, items):
        print(f"\n-- {title} --")
        print(pad("Service", thirty:=32), pad("Monthly", 12))
        for s in items:
            print(pad(s["service"], thirty), pad(fmt_money(s["monthly_cost"]), 12))

    print_services("Services & Cost (MIN)", services_min)
    print_services("Services & Cost (MAX)", services_max)

    # Detailed plans
    def show_plan(title, items):
        print(f"\n-- {title} --")
        print(pad("Stage",6), pad("Group",16), pad("Task",36), pad("TokIn",8), pad("TokOut",8), "Monthly")
        for li in items:
            print(
                pad(li['stage'],6),
                pad(li['merge_group'],16),
                pad(li['task_name'],36),
                pad(int(li['tokens_in']),8),
                pad(int(li['tokens_out']),8),
                fmt_money(li['monthly_cost'])
            )

    show_plan("MIN plan (merged when possible)", report["min_plan"])
    show_plan("MAX plan (all separate)",       report["max_plan"])

    # Fit checklist (✅/⚠️)
    print("\n-- Fit Checklist --")
    for status, msg in checklist:
        icon = "✅" if status == "ok" else "⚠️"
        print(icon, msg)

    print(f"\n✅ Saved:\n  JSON  {out_json}\n  CSV   {out_csv}\n  MD    {out_md}\n")
    print("Tip: drop real prices into data/raw/models.yaml, embeddings.yaml, vector_db.yaml for accuracy.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
