# src/app/core/pricing_dict.py
# Cost per 1M tokens (input/output) for each model

PRICING_DICT = {
    "anthropic/claude-3.5-haiku": {"input_per_1M": 0.8, "output_per_1M": 4},
    "anthropic/claude-3.5-sonnet-mini": {"input_per_1M": 0.15, "output_per_1M": 0.6},
    "cohere/command-r-plus-mini": {"input_per_1M": 0.0025, "output_per_1M": 0.01},
    "google/gemini-1.5-flash": {"input_per_1M": 0.075, "output_per_1M": 0.3},
    "google/gemini-1.5-pro": {"input_per_1M": 1.25, "output_per_1M": 5},
    "mistral/mistral-medium": {"input_per_1M": 0.4, "output_per_1M": 2},
    "mistral/mistral-large-mini": {"input_per_1M": 0, "output_per_1M": 0},  # auto-replaced if 0
    "openai/gpt-4o-mini": {"input_per_1M": 0.48, "output_per_1M": 6.3},
    "openai/gpt-4.1-mini": {"input_per_1M": 0.15, "output_per_1M": 6},
}
