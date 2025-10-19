#!/usr/bin/env python3
"""
task_detect.py — Step 2 (LLM-assisted selector with deterministic fallback)

Quick start (no flags):
  python -m src.app.core.task_detect

What this does
--------------
• Auto-picks the latest run dir under data/out/ (or use --run-dir)
• Prompts: "Choose LLM provider (perplexity/openai/grok/google/none) [none]:"
  - Type a name to use that provider (keys read from .env automatically)
  - Type Enter or 'none' to use deterministic (no LLM) fuzzy-style detection
• Loads Excel catalog (default: AI_Cost_Estimator_Catalog_UPDATED.xlsx; also looks in data/raw/)
• Sends a minimal catalog to the LLM (id|name|category) — but also carries basis/include_if/model
  so deterministic fallback can rank properly
• Joins back, evaluates include_if, tags mvp_priceable
• Writes selected_tasks_*.json + explain_*.md into the run dir
"""

from __future__ import annotations
import argparse, json, ast, os, re
from pathlib import Path
from typing import Any, Dict, List, Set, Optional
from dotenv import load_dotenv

import pandas as pd

from ..utils.io_paths import resolve_run_dir, require_latest_in, next_out_path
from ..llm.selector import select_tasks
from ..llm.config import default_model_for

# ---------------- safe rule evaluation ----------------
_ALLOWED_NODES = {
    ast.Module, ast.Expr, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Subscript,
    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE,
    ast.In, ast.NotIn, ast.USub, ast.UAdd, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.Call
}
_SAFE_FUNCS = {"min": min, "max": max, "abs": abs, "round": round}

def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out

def _names_ok(node: ast.AST, allowed: Set[str]) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id not in allowed and sub.id not in _SAFE_FUNCS:
            return False
    return True

def safe_eval(expr: str, env: Dict[str, Any]) -> bool:
    """Evaluate a tiny boolean/arithmetic expression safely. Empty/invalid => True (include)."""
    if not isinstance(expr, str) or not expr.strip():
        return True
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return False
    if not all(type(n) in _ALLOWED_NODES for n in ast.walk(node)):
        return False
    if not _names_ok(node, set(env.keys())):
        return False
    try:
        return bool(eval(compile(node, "<include_if>", "eval"), {"__builtins__": {}, **_SAFE_FUNCS}, env))
    except Exception:
        return False

# ---------------- helpers: intake, catalog, env derivation ----------------

def _derive_env(intake: Dict[str, Any]) -> Dict[str, Any]:
    users = int(intake.get("traffic", {}).get("users", 0) or 0)
    rpm = int(intake.get("traffic", {}).get("rpm", intake.get("traffic", {}).get("qpm", 0)) or 0)
    tokens_in = int(intake.get("traffic", {}).get("tokens_in", 0) or 0)
    tokens_out = int(intake.get("traffic", {}).get("tokens_out", 0) or 0)
    rag_on = bool(intake.get("rag", {}).get("enabled", False))
    translation_on = bool(intake.get("i18n", {}).get("translation_required", False))
    hosting = intake.get("preferences", {}).get("hosting", "api")
    compliance = intake.get("ops", {}).get("compliance", "None")
    corpus_gb = intake.get("rag", {}).get("corpus_gb", None)
    avg_chunk_tokens = intake.get("rag", {}).get("avg_chunk_tokens", None)
    refresh_rate = intake.get("rag", {}).get("refresh_rate", None)

    return {
        "users": users,
        "rpm": rpm,
        "qpm": rpm,
        "num_requests": users * rpm,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "rag_enabled": rag_on,
        "translation_enabled": translation_on,
        "hosting": hosting,
        "compliance": compliance,
        "corpus_gb": corpus_gb,
        "avg_chunk_tokens": avg_chunk_tokens,
        "refresh_rate": refresh_rate,
    }

def _load_catalog(catalog_path: Path) -> pd.DataFrame:
    """
    Load the catalog and normalize column names to what task_detect expects.
    - Falls back to data/raw/<filename> if needed
    - Handles both Sheet1 and "Sheet 1"
    - Renames 'id' -> 'task_id'
    """
    # Resolve path (try data/raw/<name> if not found)
    if not catalog_path.exists():
        alt_path = Path("data/raw") / catalog_path.name
        if alt_path.exists():
            catalog_path = alt_path
        else:
            raise FileNotFoundError(
                f"❌ Catalog file not found.\nTried:\n - {catalog_path}\n - {alt_path}"
            )

    # Load Excel/CSV
    suf = catalog_path.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls", ".xltx"):
        xl = pd.ExcelFile(catalog_path)
        sheet = "Sheet1" if "Sheet1" in xl.sheet_names else ("Sheet 1" if "Sheet 1" in xl.sheet_names else xl.sheet_names[0])
        df = xl.parse(sheet)
    else:
        df = pd.read_csv(catalog_path)

    # Strip whitespace in headers
    df.columns = [str(c).strip() for c in df.columns]

    # Canonicalize: id -> task_id
    if "task_id" not in df.columns and "id" in df.columns:
        df.rename(columns={"id": "task_id"}, inplace=True)

    # Ensure required columns exist
    for required in ("task_id", "task_name", "parent_category"):
        if required not in df.columns:
            df[required] = None

    return df

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = None

# ---------------- deterministic fallback (fuzzy-ish) ----------------

_STOPWORDS = {
    "the","a","an","and","or","of","to","for","with","on","in","by","at","from","that","this","it","its",
    "be","is","are","as","into","over","while","when","then","than","we","our","your","their","you","i",
    "app","application","build","create","develop","make","do"
}

def _tok(text: str) -> list[str]:
    text = (text or "").lower()
    toks = re.findall(r"[a-z0-9\+\-_→]+", text)
    return [t for t in toks if t and t not in _STOPWORDS]

def _row_keywords(row) -> list[str]:
    parts = " ".join(str(x or "") for x in [
        row.get("task_name", ""), row.get("parent_category", ""), row.get("merge_group", "")
    ])
    return _tok(parts)

def _score_row(app_tokens: list[str], row_tokens: list[str]) -> float:
    if not row_tokens:
        return 0.0
    rs = set(row_tokens)
    hits = sum(1.0 for t in app_tokens if t in rs)
    return hits / (len(rs) + 1e-6)

def _deterministic_select_ids(intake: dict, df: pd.DataFrame) -> list[str]:
    """Keyword-based, group-aware selector (no LLM) — adapted to behave like your old fuzzy version."""
    app_tokens = _tok(intake.get("app_brief", ""))

    # score each row
    scored = []
    for _, r in df.iterrows():
        tid = str(r.get("task_id") or "").strip()
        tname = str(r.get("task_name") or "").strip()
        if not tid or not tname:
            continue
        rk = _row_keywords(r)
        s = _score_row(app_tokens, rk)
        scored.append((s, r))

    # bucket by merge_group (if present)
    by_group: Dict[str, List[tuple[float, Any]]] = {}
    for s, r in scored:
        mg = str(r.get("merge_group") or "")
        by_group.setdefault(mg, []).append((s, r))

    # simple boosts from intake signals (RAG/translation)
    boosts: Dict[str, float] = {}
    if bool(intake.get("rag", {}).get("enabled", False)):
        boosts["S1_retrieve"] = boosts.get("S1_retrieve", 0.0) + 3.0
    if bool(intake.get("i18n", {}).get("translation_required", False)):
        boosts["S0_translate"] = boosts.get("S0_translate", 0.0) + 2.5

    # pick top K per group (K=3), sort by score+boost
    chosen_rows: List[tuple[float, Any]] = []
    for mg, lst in by_group.items():
        lst.sort(key=lambda z: z[0], reverse=True)
        top = lst[:3]
        for s, r in top:
            conf = s + boosts.get(mg, 0.0)
            chosen_rows.append((conf, r))

    # global sort; keep unique by (merge_group, task_name)
    seen = set()
    uniq = []
    for conf, r in sorted(chosen_rows, key=lambda z: (-z[0], str(z[1].get("stage") or ""), str(z[1].get("task_name") or ""))):
        key = (str(r.get("merge_group") or ""), str(r.get("task_name") or ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    # return task_ids (clip to 15)
    out: List[str] = []
    for r in uniq:
        tid = str(r.get("task_id") or "").strip()
        if tid:
            out.append(tid)
        if len(out) >= 15:
            break
    return out

# ---------------- simple provider prompt ----------------
def _prompt_provider_simple() -> tuple[Optional[str], Optional[str]]:
    """
    Single-line prompt with explanatory note for 'none'.
    Valid: perplexity, openai, grok, google, none
    """
    print(
        "\nAvailable LLM providers:\n"
        "  • perplexity\n"
        "  • openai\n"
        "  • grok\n"
        "  • google\n"
        "  • none  — performs deterministic, rule-based task identification (no LLM used).\n"
        "without using any LLM.\n"
    )
    valid = {"perplexity", "openai", "grok", "google", "none"}
    raw = input("Choose LLM provider (perplexity/openai/grok/google/none) [none]: ").strip().lower()
    if raw == "" or raw == "none":
        return None, None
    provider = raw
    if provider not in valid:
        print("Unknown provider. Using deterministic fallback (none).")
        return None, None
    model = default_model_for(provider)
    return provider, model


# ---------------- core detection ----------------

def detect(
    intake: Dict[str, Any],
    catalog_path: Path,
    llm_provider: Optional[str],
    llm_model: Optional[str]
) -> Dict[str, Any]:
    derived = _derive_env(intake)
    env = {**_flatten(intake), **derived,
           "rag": derived["rag_enabled"], "translation": derived["translation_enabled"]}

    df = _load_catalog(catalog_path)
    needed_cols = [
        "task_id", "task_name", "parent_category", "stage",
        "pricing_basis", "include_if", "pricing_model_key",
        "model_profile.key", "model_band_min_key", "merge_group"
    ]
    _ensure_cols(df, needed_cols)

    # Minimal candidates for LLM: id | name | category
    # Also pass through basis/include_if/model so deterministic fallback can rank if needed.
    candidates_for_llm: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        tid = str(r.get("task_id") or "").strip()
        tname = str(r.get("task_name") or "").strip()
        pcat = str(r.get("parent_category") or "").strip()
        if not tid or not tname:
            continue
        candidates_for_llm.append({
            "task_id": tid,
            "task_name": tname,
            "parent_category": pcat,
            "pricing_basis": str(r.get("pricing_basis") or "").strip().lower(),
            "include_if": str(r.get("include_if") or "").strip(),
            "pricing_model_key": str(
                r.get("pricing_model_key")
                or r.get("model_profile.key")
                or r.get("model_band_min_key")
                or ""
            ).strip(),
        })

    # Selection path
    if (llm_provider is None) or (str(llm_provider).lower() == "none"):
        # Pure deterministic selection (no LLM) — behaves like your old fuzzy detector
        selected_ids = _deterministic_select_ids(intake, df)
    else:
        # LLM path (selector internally falls back to deterministic if keys are missing or LLM returns nothing)
        selected_ids = select_tasks(
            candidates=candidates_for_llm,
            intake=intake,
            provider_override=llm_provider,
            model_override=llm_model
        )

    selected_set = set(selected_ids)

    # Join back to full catalog
    df_sel = df[df["task_id"].astype(str).isin(selected_set)].copy()

    def resolve_model_key(row) -> str | None:
        for col in ("pricing_model_key", "model_profile.key", "model_band_min_key"):
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
        return None

    included: List[Dict[str, Any]] = []
    excluded_after_check: List[Dict[str, Any]] = []

    for _, r in df_sel.iterrows():
        tid = str(r.get("task_id", "")).strip()
        tname = str(r.get("task_name", "")).strip()
        stage = r.get("stage", None)
        basis = str(r.get("pricing_basis", "") or "").strip().lower()
        parent_cat = r.get("parent_category", None)
        include_if = str(r.get("include_if", "") or "").strip()
        pricing_key = resolve_model_key(r)

        ok = safe_eval(include_if, env)
        mvp_priceable = basis in {"token", "api_call", "embedding"}

        rec = {
            "task_id": tid,
            "task_name": tname,
            "stage": stage,
            "pricing_basis": basis or None,
            "pricing_model_key": pricing_key,
            "parent_category": parent_cat,
            "include_if": include_if or None,
            "mvp_priceable": mvp_priceable,
        }

        if ok:
            included.append(rec)
        else:
            excluded_after_check.append({
                "task": tname or tid,
                "reason": f"include_if evaluated False: {include_if}" if include_if else "include_if empty/false",
            })

    provenance = {
        "llm_selection": {
            "provider": (llm_provider or "none"),
            "model": (llm_model or "(default)"),
            "mode": "full_catalog_minimal_fields",
            "candidate_count": len(candidates_for_llm),
        },
        "monitoring": {"provider": "langsmith", "tiering_used": False},
        "pricing": {"source": "Excel/PRICING_DICT (models priced later in plan_and_cost)", "version": "current-run"},
        "catalog": {"path": str(catalog_path), "sheet": "Sheet1", "row_count": int(len(df))}
    }

    return {"included": included, "excluded_after_rule": excluded_after_check, "derived_env": derived, "provenance": provenance}

# ---------------- CLI ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="Run directory created by cli_intake (default: latest under data/out)")
    ap.add_argument("--catalog", default="AI_Cost_Estimator_Catalog_UPDATED.xlsx", help="Path to Excel/CSV catalog")
    ap.add_argument("--llm-provider", default=None, help="perplexity | openai | grok | google | none")
    ap.add_argument("--llm-model", default=None, help="Optional model override for the chosen provider")
    args = ap.parse_args()

    # Load .env so API keys are available automatically
    load_dotenv()

    # Resolve run dir (auto-picks latest under data/out if not passed)
    run_dir = resolve_run_dir(args.run_dir)

    # Provider: if not supplied, ask simply
    llm_provider = args.llm_provider.lower() if args.llm_provider else None
    llm_model = args.llm_model
    if llm_provider is None:
        prov, model = _prompt_provider_simple()
        llm_provider, llm_model = prov, model

    # Load intake json (latest in run dir)
    intake_path = require_latest_in(
        run_dir, prefix="intake_", suffix=".json",
        missing_msg=f"Could not find an intake_*.json in {run_dir}."
    )
    with open(intake_path, "r", encoding="utf-8") as f:
        intake = json.load(f)

    result = detect(
        intake=intake,
        catalog_path=Path(args.catalog),
        llm_provider=llm_provider,
        llm_model=llm_model
    )

    # Write artifacts
    out_tasks = next_out_path(run_dir, "selected_tasks", "json")
    with open(out_tasks, "w", encoding="utf-8") as f:
        json.dump({
            "selected_tasks": result["included"],
            "derived": result["derived_env"],
            "provenance": result["provenance"]
        }, f, indent=2, ensure_ascii=False)

    out_md = next_out_path(run_dir, "explain", "md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Task Detection — Explanation\n\n")
        f.write(f"**LLM provider/model:** {result['provenance']['llm_selection']['provider']} / "
                f"{result['provenance']['llm_selection']['model']}\n")
        f.write(f"**Mode:** {result['provenance']['llm_selection']['mode']}\n")
        f.write(f"**Candidates seen:** {result['provenance']['llm_selection']['candidate_count']}\n")
        f.write(f"**Catalog:** {result['provenance']['catalog']['path']} (rows: {result['provenance']['catalog']['row_count']})\n\n")

        f.write(f"**Included tasks:** {len(result['included'])}\n\n")
        for rec in result["included"][:50]:
            priced = "✓" if rec.get("mvp_priceable") else "—"
            f.write(f"- {rec.get('task_name')}  "
                    f"(id: {rec.get('task_id')}, basis: {rec.get('pricing_basis')}, "
                    f"model: {rec.get('pricing_model_key')}, mvp_priceable: {priced})\n")
        if len(result["included"]) > 50:
            f.write(f"...and {len(result['included'])-50} more\n")

        f.write("\n**Excluded after rule checks (first 50):**\n")
        for rec in result["excluded_after_rule"][:50]:
            f.write(f"- {rec['task']}: {rec['reason']}\n")
        if len(result["excluded_after_rule"]) > 50:
            f.write(f"...and {len(result['excluded_after_rule'])-50} more\n")

        f.write("\n**Derived environment used for rules:**\n")
        for k, v in result["derived_env"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n**Assumptions:**\n")
        f.write("- Monitoring provider assumed: LangSmith (no tiering in MVP).\n")
        f.write("- Non-MVP priceable tasks are listed but not costed; the planner will surface them as 'not costed'.\n")

    print("✅ Wrote:")
    print(f"   {out_tasks}")
    print(f"   {out_md}")
    print(f"   Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
