"""Central configuration. Everything tunable lives here, sourced from env vars."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings.

    Provider selection is deliberately explicit: the app must never silently fall
    back to a different model than the operator configured.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- LLM ----
    llm_provider: Literal["gemini", "groq", "openai", "none"] = Field(
        default="none", alias="LLM_PROVIDER"
    )
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_fast_model: str = Field(default="", alias="LLM_FAST_MODEL")
    llm_timeout_s: float = Field(default=180.0, alias="LLM_TIMEOUT_S")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    # Applies to Gemini 2.x, Groq and OpenAI. Gemini 3.x deprecated sampling
    # parameters in favour of thinking levels, so it is ignored there.
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    # Gemini 3.x only: minimal | low | medium | high.
    llm_thinking_level: Literal["minimal", "low", "medium", "high"] = Field(
        default="high", alias="LLM_THINKING_LEVEL"
    )
    # Comma-separated models tried in order when the preferred one is out of quota.
    llm_fallback_models: str = Field(default="", alias="LLM_FALLBACK_MODELS")
    llm_max_retry_wait_s: float = Field(default=60.0, alias="LLM_MAX_RETRY_WAIT_S")

    # ---- ingestion limits ----
    max_upload_mb: float = Field(default=200.0, alias="MAX_UPLOAD_MB")
    max_files_per_session: int = Field(default=10, alias="MAX_FILES_PER_SESSION")
    profile_sample_rows: int = Field(default=50_000, alias="PROFILE_SAMPLE_ROWS")

    # ---- query guard ----
    max_result_rows: int = Field(default=5_000, alias="MAX_RESULT_ROWS")
    query_timeout_s: float = Field(default=30.0, alias="QUERY_TIMEOUT_S")

    # ---- agent ----
    max_agent_steps: int = Field(default=8, alias="MAX_AGENT_STEPS")
    history_turns: int = Field(default=6, alias="HISTORY_TURNS")

    # ---- cache ----
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl_s: int = Field(default=3600, alias="CACHE_TTL_S")
    cache_max_entries: int = Field(default=256, alias="CACHE_MAX_ENTRIES")

    # ---- api ----
    api_base_url: str = Field(default="http://localhost:8000", alias="API_BASE_URL")
    api_key: str = Field(default="", alias="APP_API_KEY")

    # ---- observability ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    @property
    def default_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "gemini": "gemini-3.6-flash",
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o-mini",
            "none": "",
        }[self.llm_provider]

    @property
    def fast_model(self) -> str:
        if self.llm_fast_model:
            return self.llm_fast_model
        return {
            "gemini": "gemini-3.5-flash-lite",
            "groq": "llama-3.1-8b-instant",
            "openai": "gpt-4o-mini",
            "none": "",
        }[self.llm_provider]

    @property
    def model_chain(self) -> list[str]:
        """Preferred model first, then any configured fallbacks, de-duplicated."""
        chain = [self.default_model]
        for name in self.llm_fallback_models.split(","):
            name = name.strip()
            if name and name not in chain:
                chain.append(name)
        return [m for m in chain if m]

    @property
    def llm_configured(self) -> bool:
        return self.llm_provider != "none" and bool(self.llm_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
