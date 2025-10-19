# src/app/core/formulas.py
"""
Reusable cost formulas for the MVP.

Implements:
- compute_token_cost(): monthly LLM token cost
- merged-input estimation for tasks in the same category (prompt merge)
- embedding helpers (placeholders)

Merging logic is based on your teammate's sketch:
  T_sep = N * I
  T_merge_raw = α * I + N * ((1 - α) * I + s)
  Optionally adjust by (1 + addon_ratio) when mode == "proportional"

Where:
  I      = avg input tokens per request (baseline, per task)
  N      = number of tasks merged together (same category)
  α      = shared fraction of tokens (default 0.25)
  s      = separator tokens between merged prompts (default 5)
"""

from __future__ import annotations

# --------- Core token costing ---------

def compute_token_cost(
    monthly_requests: int,
    avg_input_tokens_per_request: float,
    avg_output_tokens_per_request: float,
    price_input_per_1M: float,
    price_output_per_1M: float,
) -> float:
    if monthly_requests <= 0:
        return 0.0

    ti_millions = (avg_input_tokens_per_request or 0.0) / 1_000_000.0
    to_millions = (avg_output_tokens_per_request or 0.0) / 1_000_000.0

    per_request_cost = ti_millions * (price_input_per_1M or 0.0) + \
                       to_millions * (price_output_per_1M or 0.0)
    return float(monthly_requests) * per_request_cost


# --------- Merging (prompt grouping) ---------

def merged_group_input_tokens_per_task(
    baseline_input_tokens_per_task: float,
    N: int,
    mode: str = "none",
    addon_ratio: float = 0.0,
    alpha: float = 0.25,
    separator_tokens: int = 5,
) -> float:
    """
    Returns the effective *per-task* input tokens after merging N tasks in one category.

    Baseline (no merge): per-task input tokens = I
    Separate total:      T_sep = N * I
    Merge raw total:     T_merge_raw = α*I + N * ((1 - α)*I + s)
    Mode "proportional": T_merge = T_merge_raw * (1 + addon_ratio)
    Other modes:         T_merge = T_merge_raw

    Effective per-task input tokens after merging = T_merge / N
    """
    I = float(baseline_input_tokens_per_task or 0.0)
    if N <= 1 or I <= 0:
        return I

    T_sep = N * I
    T_merge_raw = alpha * I + N * ((1.0 - alpha) * I + separator_tokens)

    if mode == "proportional":
        T_merge = T_merge_raw * (1.0 + float(addon_ratio or 0.0))
    else:
        # "none", "fixed", "separate_only", or unknown -> just use raw merge formula
        T_merge = T_merge_raw

    # Safety: never exceed the separate baseline per-task input significantly
    per_task_after = max(1.0, T_merge / N)
    # If merge is worse than separate (can happen with large s), just use baseline I
    if per_task_after > I:
        return I
    return per_task_after


def merge_token_savings(
    baseline_input_tokens_per_task: float,
    N: int,
    mode: str = "none",
    addon_ratio: float = 0.0,
    alpha: float = 0.25,
    separator_tokens: int = 5,
) -> float:
    """
    Returns *total* savings in input tokens for the whole group vs separate,
    primarily for diagnostics. Not required by plan_and_cost, but useful.

    savings = T_sep - T_merge
    """
    I = float(baseline_input_tokens_per_task or 0.0)
    if N <= 1 or I <= 0:
        return 0.0

    T_sep = N * I
    T_merge_raw = alpha * I + N * ((1.0 - alpha) * I + separator_tokens)

    if mode == "proportional":
        T_merge = T_merge_raw * (1.0 + float(addon_ratio or 0.0))
    else:
        T_merge = T_merge_raw

    return max(0.0, T_sep - T_merge)


# --------- Embedding placeholder helpers ---------

def compute_embedding_cost(
    total_tokens_to_embed: int,
    price_input_per_1M: float,
) -> float:
    if total_tokens_to_embed <= 0:
        return 0.0
    return (total_tokens_to_embed / 1_000_000.0) * (price_input_per_1M or 0.0)
