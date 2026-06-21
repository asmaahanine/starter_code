"""
config.py
=========

Typed application configuration via Pydantic ``BaseSettings``.

Loads from environment variables and an optional ``.env`` file, validates types,
and fails fast at startup with a clear error if something required is missing.
This is the pattern that avoids the classic "KeyError deep inside a job at 3am"
problem — misconfiguration surfaces immediately, on load.

Requires: pydantic>=2, pydantic-settings

Example
-------
    from config import Settings

    settings = Settings()                  # reads env + .env
    print(settings.llm_model)
    print(settings.sharepoint_site_url)

    # Override at construction (handy in tests):
    settings = Settings(log_level="DEBUG")
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """
    Central application settings.

    Every field maps to an env var of the same name (case-insensitive).
    ``SecretStr`` fields are masked in logs/reprs so secrets don't leak.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # ignore unrelated env vars rather than erroring
    )

    # -- general ---------------------------------------------------------
    app_name: str = "ds-toolkit"
    log_level: LogLevel = LogLevel.INFO
    environment: str = Field(default="dev", description="dev | staging | prod")

    # -- LLM -------------------------------------------------------------
    llm_model: str = "mistral-large-latest"
    llm_timeout: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    mistral_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # -- integrations (e.g. SharePoint / Graph) --------------------------
    sharepoint_site_url: str | None = None
    sharepoint_folder_path: str = "DSI/CTI"
    graph_tenant_id: str | None = None
    graph_client_id: str | None = None
    graph_client_secret: SecretStr | None = None

    @field_validator("environment")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        allowed = {"dev", "staging", "prod"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


if __name__ == "__main__":
    # Loads whatever is in the current environment; prints with secrets masked.
    settings = Settings()
    print(settings.model_dump())  # SecretStr fields show as '**********'
