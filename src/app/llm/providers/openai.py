# src/app/llm/providers/openai.py
from __future__ import annotations
import json, re, os
from typing import List, Dict, Any
from ..config import LLMConfig

_JSON_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)

def _render_candidates_block(cands: List[Dict[str, Any]]) -> str:
    lines = []
    for c in cands:
        lines.append(
            f"- id={c.get('task_id')} | name={c.get('task_name')} | "
            f"cat={c.get('parent_category')} | basis={c.get('pricing_basis')} | "
            f"include_if={c.get('include_if') or ''} | model={c.get('pricing_model_key') or ''}"
        )
    return "\n".join(lines)

def _build_prompt(intake: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    translation = bool(intake.get("i18n", {}).get("translation_required", False))
    rag = bool(intake.get("rag", {}).get("enabled", False))
    app_brief = str(intake.get("app_brief", "")).strip()
    return f"""
You are a task selector. Select tasks ONLY from the candidate list below.
Return JSON with this exact shape:

{{
  "selected_task_ids": ["TASK_ID_1", "TASK_ID_2", "..."],
  "notes": "Very short reason (<=50 words)."
}}

Rules:
- Only select tasks with pricing_basis in ["token","api_call","embedding"].
- Do not invent tasks outside the candidate list.
- If unsure, return an empty list.
- Avoid tasks whose include_if would be false given the intake.

Intake gates:
- translation_required = {translation}
- rag_enabled = {rag}
- app_brief = "{app_brief}"

Candidates:
{_render_candidates_block(candidates)}
""".strip()

def _parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("LLM did not return JSON.")
    return json.loads(m.group(0))

class OpenAISelector:
    """
    OpenAI via LangChain.
      pip install langchain langchain-openai
      export OPENAI_API_KEY=...
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing OpenAI LangChain integration. Install with "
                "'pip install langchain langchain-openai'") from e

        # Optional: allow base_url override via env (useful for Azure/OpenAI-compatible endpoints)
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

        self._client = ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            base_url=base_url
        )

    def select(self, candidates: List[Dict[str, Any]], intake: Dict[str, Any]) -> List[str]:
        prompt = _build_prompt(intake, candidates)
        try:
            resp = self._client.invoke(prompt)
            content = getattr(resp, "content", None) or str(resp)
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}") from e

        data = _parse_json_response(content)
        selected = data.get("selected_task_ids", [])
        if not isinstance(selected, list):
            raise ValueError("selected_task_ids is not a list.")
        return [str(x) for x in selected if x is not None]
