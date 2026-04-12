from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Vital Lens rPPG"
    app_version: str = "1.0.0"
    debug: bool = False

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_relative_path: str = "../models/BP4D_BigSmall_Multitask_Fold2.pth"
    model_device: str = "cpu"
    model_strict_loading: bool = False

    frontend_dir: str = "../frontend"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def backend_dir(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def model_path(self) -> Path:
        return (self.backend_dir / self.model_relative_path).resolve()

    @property
    def frontend_path(self) -> Path:
        return (self.backend_dir / self.frontend_dir).resolve()


@lru_cache

def get_settings() -> Settings:
    return Settings()
