# src/app/llm/config.py
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class LLMConfig:
    """
    Minimal, deterministic LLM config for MVP.
    - Provider is chosen at runtime (e.g., --llm-provider openai).
    - Model is fixed per provider by default but can be overridden via --llm-model.
    - Decoding params are fixed for determinism; not exposed to users.
    """
    provider: str = (os.getenv("LLM_PROVIDER", "perplexity").strip().lower() or "perplexity")
    model: str = ""          # if empty, we'll fill from default_model_for(provider)

    # Deterministic decoding (internal, not user-exposed)
    temperature: float = 0.0
    top_p: float = 1.0
    max_candidates: int = 30
    max_return: int = 15
    seed: int = 42           # best-effort; some providers ignore it

def default_model_for(provider: str) -> str:
    """
    Single place to define default model names per provider.
    Adjust here later if your account uses different labels.
    """
    p = (provider or "").lower()
    if p == "openai":
        return "gpt-4o-mini"
    if p == "grok":
        return "grok-3-mini"
    if p == "google":
        return "gemini-1.5-flash"
    if p == "perplexity":
        return "sonar"  # Per your note: use 'sonar' (no 'small chat' variant)
    # Fallback (shouldn't happen if provider is validated upstream)
    return "sonar"

def is_llm_enabled(cfg: LLMConfig) -> bool:
    """
    Returns True if a usable API key exists for the chosen provider.
    """
    p = cfg.provider
    if p == "perplexity":
        return bool(os.getenv("PPLX_API_KEY"))
    if p == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if p == "grok":
        return bool(os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY"))
    if p == "google":
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY"))
    return False

def build_config(provider_override: str | None = None, model_override: str | None = None) -> LLMConfig:
    """
    Helper to construct a finalized config from optional runtime overrides.
    Ensures model is populated with a sane default for the chosen provider.
    """
    base = LLMConfig()
    provider = (provider_override or base.provider or "perplexity").lower()
    model = (model_override or base.model or "").strip() or default_model_for(provider)
    # dataclasses are frozen; rebuild with updated fields
    return LLMConfig(
        provider=provider,
        model=model,
        temperature=base.temperature,
        top_p=base.top_p,
        max_candidates=base.max_candidates,
        max_return=base.max_return,
        seed=base.seed,
    )
