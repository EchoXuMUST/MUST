from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Runtime configuration for a DeepResearch-like Qwen3-VL framework."""

    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    api_base: str = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "EMPTY")

    api_timeout_s: float = float(os.getenv("API_TIMEOUT_S", "120"))
    api_retries: int = int(os.getenv("API_RETRIES", "3"))
    api_retry_backoff_s: float = float(os.getenv("API_RETRY_BACKOFF_S", "2"))
    api_max_tokens: int = int(os.getenv("API_MAX_TOKENS", "128"))

    max_rounds: int = int(os.getenv("MAX_RESEARCH_ROUNDS", "3"))
    max_snippets_per_round: int = int(os.getenv("MAX_SNIPPETS_PER_ROUND", "5"))
    max_actions_per_question: int = int(os.getenv("MAX_ACTIONS_PER_QUESTION", "4"))
    max_evidence_items: int = int(os.getenv("MAX_EVIDENCE_ITEMS", "12"))
    max_tool_logs: int = int(os.getenv("MAX_TOOL_LOGS", "20"))

    search_api_url: str | None = os.getenv("SEARCH_API_URL")
    search_api_key: str | None = os.getenv("SEARCH_API_KEY")


settings = Settings()
