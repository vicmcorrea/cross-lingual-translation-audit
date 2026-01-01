"""Project paths and private settings kept outside Hydra."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
DATA_ROOT = Path(os.getenv("TRANSLATION_AUDIT_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
ARTIFACT_ROOT = (
    Path(os.getenv("TRANSLATION_AUDIT_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")).expanduser().resolve()
)
ENV_FILE = PROJECT_ROOT / ".env"


def load_project_env() -> None:
    """Load local secrets without exposing them to Hydra composition."""
    load_dotenv(ENV_FILE, override=False)


class PrivateRuntimeSettings(BaseSettings):
    """Secrets available only to clients that explicitly request them."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    runpod_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
