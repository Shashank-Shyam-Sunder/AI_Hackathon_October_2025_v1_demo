# src/app/llm/selector.py
from __future__ import annotations
from typing import List, Dict, Any, Set, Optional
from .config import build_config, is_llm_enabled, default_model_for  # default_model_for not strictly needed but fine

def _load_provider(cfg):
    if cfg.provider == "perplexity":
        from .providers.perplexity import PerplexitySelector
        return PerplexitySelector(cfg)
    if cfg.provider == "openai":
        from .providers.openai import OpenAISelector
        return OpenAISelector(cfg)
    if cfg.provider == "grok":
        from .providers.grok import GrokSelector
        return GrokSelector(cfg)
    if cfg.provider == "google":
        from .providers.google import GoogleSelector
        return GoogleSelector(cfg)
    raise ValueError(f"Unsupported LLM provider: {cfg.provider}")

def _validate_and_clip_ids(proposed_ids: List[str], candidate_id_set: Set[str], max_return: int) -> List[str]:
    seen, out = set(), []
    for tid in proposed_ids:
        tid_s = str(tid)
        if tid_s in candidate_id_set and tid_s not in seen:
            out.append(tid_s)
            seen.add(tid_s)
            if len(out) >= max_return:
                break
    return out

def _fallback_deterministic(cands: List[Dict[str, Any]], max_return: int) -> List[str]:
    """
    Deterministic selection when LLM is unavailable or returns nothing.
    Prefer priceable bases, but NEVER return empty (no hard filtering).
    """
    basis_rank = {"token": 0, "embedding": 1, "api_call": 2}
    def key(c):
        basis = str(c.get("pricing_basis", "")).lower()
        return (
            basis_rank.get(basis, 9),
            str(c.get("parent_category", "")).lower(),
            str(c.get("task_name", "")).lower(),
            str(c.get("task_id", "")).lower(),
        )
    seen, out = set(), []
    for c in sorted(cands, key=key):
        tid = str(c.get("task_id") or "")
        if tid and tid not in seen:
            out.append(tid)
            seen.add(tid)
            if len(out) >= max_return:
                break
    return out

def select_tasks(
    candidates: List[Dict[str, Any]],
    intake: Dict[str, Any],
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None
) -> List[str]:
    """
    Unified selector interface used by core.task_detect.
    Uses build_config() to create provider-specific config with defaults.
    Guarantees a non-empty result (falls back if LLM fails or returns nothing).
    """
    cfg = build_config(provider_override, model_override)
    short = candidates[: cfg.max_candidates]
    candidate_id_set = {str(c.get("task_id")) for c in short if c.get("task_id")}

    # If no API key for the chosen provider, fallback immediately
    if not is_llm_enabled(cfg):
        return _fallback_deterministic(short, cfg.max_return)

    provider_impl = _load_provider(cfg)
    try:
        proposed_ids = provider_impl.select(short, intake)
    except Exception:
        return _fallback_deterministic(short, cfg.max_return)

    # Fallback if LLM returned nothing or non-list
    if not proposed_ids:
        return _fallback_deterministic(short, cfg.max_return)

    clean_ids = _validate_and_clip_ids([str(x) for x in proposed_ids], candidate_id_set, cfg.max_return)
    return clean_ids or _fallback_deterministic(short, cfg.max_return)
