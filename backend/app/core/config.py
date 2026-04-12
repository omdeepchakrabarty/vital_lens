import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    app_name: str = "Vital Lens rPPG"
    app_version: str = "1.0.0"
    debug: bool = False

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_relative_path: str = "models/BP4D_BigSmall_Multitask_Fold2.pth"
    model_device: str = "cpu"
    model_strict_loading: bool = False

    frontend_dir: str = "frontend"
    project_root: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @staticmethod
    def _is_project_root(path: Path) -> bool:
        required_entries = [path / "backend", path / "frontend", path / "models"]
        has_required_layout = all(entry.exists() for entry in required_entries)
        has_sentinel = any((path / sentinel).exists() for sentinel in (".git", "README.md", "Dockerfile"))
        return has_required_layout and has_sentinel

    def _discover_project_root(self) -> tuple[Path, str]:
        env_root = self.project_root or os.getenv("PROJECT_ROOT")
        if env_root:
            candidate = Path(env_root).expanduser().resolve()
            if self._is_project_root(candidate):
                return candidate, "PROJECT_ROOT"
            logger.warning(
                "PROJECT_ROOT is set to '%s' but does not match expected repository layout.",
                candidate,
            )

        module_file = Path(__file__).resolve()
        for parent in module_file.parents:
            if self._is_project_root(parent):
                return parent, f"sentinel-from-module:{module_file}"

        cwd = Path.cwd().resolve()
        for parent in (cwd, *cwd.parents):
            if self._is_project_root(parent):
                return parent, f"cwd-contract:{cwd}"

        raise RuntimeError(
            "Unable to resolve project root. Set PROJECT_ROOT to the repository root "
            "(the directory containing backend/, frontend/, and models/)."
        )

    @property
    def project_root_path(self) -> Path:
        root, source = self._discover_project_root()
        logger.info("Resolved project root: %s (source=%s)", root, source)
        return root

    @property
    def backend_dir(self) -> Path:
        backend_path = (self.project_root_path / "backend").resolve()
        if not backend_path.exists():
            logger.warning("Resolved backend directory does not exist: %s", backend_path)
        return backend_path

    @property
    def model_path(self) -> Path:
        model_path = (self.project_root_path / self.model_relative_path).resolve()
        if not model_path.exists():
            logger.warning("Resolved model path does not exist yet: %s", model_path)
        return model_path

    @property
    def frontend_path(self) -> Path:
        frontend_path = (self.project_root_path / self.frontend_dir).resolve()
        if not frontend_path.exists():
            logger.warning("Resolved frontend path does not exist: %s", frontend_path)
        return frontend_path


@lru_cache
def get_settings() -> Settings:
    return Settings()
