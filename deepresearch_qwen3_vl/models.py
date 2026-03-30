from __future__ import annotations

import json
import time
from typing import Any

from openai import APIConnectionError, NotFoundError, OpenAI

from config import settings


class QwenVLClient:
    """OpenAI-compatible client wrapper for Qwen3-VL models."""

    def __init__(self) -> None:
        self.client = OpenAI(base_url=settings.api_base, api_key=settings.api_key, timeout=settings.api_timeout_s)
        self.model = self._resolve_model_name(settings.model_name)

    def _resolve_model_name(self, preferred: str) -> str:
        """Pick an available model when preferred model ID does not exist on the server."""
        try:
            model_list = self.client.models.list()
            available = [m.id for m in model_list.data if getattr(m, "id", None)]
        except Exception:
            return preferred

        if not available:
            return preferred
        if preferred in available:
            return preferred

        preferred_tail = preferred.split("/")[-1].lower()
        candidates = [
            m
            for m in available
            if preferred_tail in m.lower() or m.lower() in preferred_tail or "qwen" in m.lower()
        ]
        chosen = candidates[0] if candidates else available[0]
        print(
            f"[QwenVLClient] configured model '{preferred}' not found; fallback to '{chosen}'. "
            f"available={available}"
        )
        return chosen

    def _get_available_models(self) -> list[str]:
        try:
            model_list = self.client.models.list()
            return [m.id for m in model_list.data if getattr(m, "id", None)]
        except Exception:
            return []

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        last_error: Exception | None = None
        for i in range(max(settings.api_retries, 1)):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max(settings.api_max_tokens, 1),
                )
                return completion.choices[0].message.content or ""
            except NotFoundError as exc:
                # model id mismatch between local vLLM served-model-name and configured MODEL_NAME
                last_error = exc
                available = self._get_available_models()
                if available:
                    new_model = available[0]
                    if new_model != self.model:
                        print(
                            f"[QwenVLClient] model '{self.model}' not found at runtime; "
                            f"retry with '{new_model}'. available={available}"
                        )
                        self.model = new_model
                        continue
                raise
            except APIConnectionError as exc:
                last_error = exc
                if i < settings.api_retries - 1:
                    time.sleep(settings.api_retry_backoff_s * (i + 1))
                    continue
                raise
        if last_error:
            raise last_error
        return ""

    def chat_json(self, messages: list[dict[str, Any]], fallback: Any, temperature: float = 0.1) -> Any:
        raw = self.chat(messages=messages, temperature=temperature).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        if "```" in raw:
            chunks = raw.split("```")
            for chunk in chunks:
                text = chunk.strip()
                if text.startswith("json"):
                    text = text[4:].strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue
        return fallback

    def chat_with_image(self, prompt: str, image_url: str, temperature: float = 0.1) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        return self.chat(messages=messages, temperature=temperature)
