"""OpenAI-compatible model providers used by attack-region selection."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMConfigurationError(RuntimeError):
    """Raised when the selected provider has incomplete configuration."""


class BaseLLMProvider(ABC):
    """Minimal provider contract used by :mod:`perch.region_recommender`."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return one model response as text."""


@dataclass(frozen=True)
class LLMConfig:
    """Environment-backed configuration for attack-region reasoning."""

    provider: str = "openai"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.3
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("PERCH_LLM_PROVIDER", "openai").strip().lower()
        if provider == "local":
            return cls(
                provider=provider,
                api_key=_first_env(
                    "PERCH_LOCAL_API_KEY",
                    default="not-needed",
                ),
                base_url=_first_env(
                    "PERCH_LOCAL_BASE_URL",
                    default="http://localhost:11434/v1",
                ),
                model=_first_env(
                    "PERCH_LOCAL_MODEL",
                    default="qwen2.5:7b",
                ),
                temperature=_env_float("PERCH_LLM_TEMPERATURE", 0.3),
                timeout_s=_env_float("PERCH_LLM_TIMEOUT_S", 30.0),
            )
        return cls(
            provider=provider,
            api_key=_first_env(
                "PERCH_OPENAI_API_KEY",
                "OPENAI_API_KEY",
                default="",
            ),
            base_url=_first_env(
                "PERCH_OPENAI_BASE_URL",
                "OPENAI_BASE_URL",
                default="https://api.deepseek.com",
            ),
            model=_first_env(
                "PERCH_OPENAI_MODEL",
                "OPENAI_MODEL",
                default="deepseek-chat",
            ),
            temperature=_env_float("PERCH_LLM_TEMPERATURE", 0.3),
            timeout_s=_env_float("PERCH_LLM_TIMEOUT_S", 30.0),
        )


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for OpenAI, DeepSeek, Qwen, Ollama, and vLLM endpoints."""

    def __init__(self, config: LLMConfig) -> None:
        if config.provider not in {"openai", "local"}:
            raise LLMConfigurationError(
                f"Unsupported PERCH_LLM_PROVIDER: {config.provider}"
            )
        if not config.api_key:
            raise LLMConfigurationError(
                "PERCH_OPENAI_API_KEY is required for LLM region selection"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise LLMConfigurationError(
                "The openai package is required for LLM region selection"
            ) from exc

        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_s,
        )
        self._model = config.model
        self._temperature = config.temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Attack-region LLM returned an empty response")
        return str(content)


def create_llm_provider(config: LLMConfig | None = None) -> BaseLLMProvider:
    """Build the configured provider without importing SDKs at module import."""
    return OpenAICompatibleProvider(config or LLMConfig.from_env())


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default
