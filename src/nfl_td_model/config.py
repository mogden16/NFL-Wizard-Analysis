"""Phase 0 settings. Secrets are read from the environment or .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    odds_api_key: str | None = None
    prediction_lead_minutes: int = 60
    odds_regions: str = "us"
