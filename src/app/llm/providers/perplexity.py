# src/app/llm/providers/perplexity.py
from __future__ import annotations
import os
from typing import List, Dict, Any
from ..config import LLMConfig

class PerplexitySelector:
    """
    Perplexity via official langchain-perplexity integration.

    Setup:
      pip install -U langchain-perplexity
      export PPLX_API_KEY=...

    Notes:
      - ChatPerplexity now lives in `langchain_perplexity`
      - Non-default params like `top_p` must be passed in model_kwargs
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        # Ensure dependency is present
        try:
            from langchain_perplexity import ChatPerplexity  # NEW import location
        except Exception as e:
            raise RuntimeError(
                "Perplexity provider requires `langchain-perplexity`.\n"
                "Install with: pip install -U langchain-perplexity"
            ) from e

        api_key = os.getenv("PPLX_API_KEY")
        if not api_key:
            raise RuntimeError("PPLX_API_KEY not found in environment/.env")

        # Perplexity’s ChatPerplexity now expects non-default args in model_kwargs
        self._client = ChatPerplexity(
            model=self.cfg.model or "sonar",
            temperature=self.cfg.temperature,
            # top_p moved into model_kwargs to silence the LangChain warning
            model_kwargs={
                "top_p": self.cfg.top_p,
                # You can pass any other non-defaults here as needed
            },
            api_key=api_key,
        )

    def _build_prompt(self, candidates: List[Dict[str, Any]], intake: Dict[str, Any]) -> str:
        # Keep your existing prompt format; example minimal prompt:
        items = "\n".join(f"- [{c['task_id']}] {c['task_name']} ({c.get('parent_category','')})" for c in candidates)
        app = intake.get("app_brief", "")
        return (
            "You are a task selector. From the list, pick the most relevant task IDs for the app.\n"
            f"App: {app}\n"
            "Tasks:\n" + items + "\n"
            "Return a JSON array of task_id strings only."
        )

    def select(self, candidates: List[Dict[str, Any]], intake: Dict[str, Any]) -> List[str]:
        prompt = self._build_prompt(candidates, intake)

        # You can keep your existing LC call structure; this is generic:
        from langchain_core.messages import HumanMessage
        resp = self._client.invoke([HumanMessage(content=prompt)])

        # Extract IDs (assuming model returns a JSON array or plaintext ids)
        text = getattr(resp, "content", "") if hasattr(resp, "content") else str(resp)
        # Very tolerant parse:
        import json, re
        ids: List[str] = []
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                ids = [str(x) for x in arr]
        except Exception:
            # fallback: pull bracketed IDs like ["12","34"] or plain [12, 34]
            m = re.search(r"\[.*?\]", text, flags=re.S)
            if m:
                try:
                    arr = json.loads(m.group(0))
                    if isinstance(arr, list):
                        ids = [str(x) for x in arr]
                except Exception:
                    pass
            if not ids:
                # last resort: grab bracketed [ID] lines from the prompt echo
                ids = re.findall(r"\[(\d+)\]", text)

        return ids
