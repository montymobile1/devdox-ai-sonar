"""
Configuration settings for the DevDox AI Sonar
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""

    VERSION: str = "0.0.6"

    CONFIG_DIR: Path = Field(
        default_factory=lambda: Path.home() / "devdox",
        description="Configuration directory",
    )

    # Computed paths
    @property
    def config_file_path(self) -> Path:
        return self.CONFIG_DIR / "config.toml"

    @property
    def auth_file_path(self) -> Path:
        return self.CONFIG_DIR / "auth.json"

    API_KEY: str = ""  # Fallback for backward compatibility
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    LLM_PROVIDER: str = "LLM_PROVIDER"
    LLM_MODEL: str = "gemini-2.5-flash"

    SONAR_TOKEN: str = Field(default="")
    SONAR_ORGANIZATION: str = Field(default="")
    SONAR_PROJECT_KEY: str = Field(default="")

    MAX_FIXES_LIMIT: int = 20
    DEFAULT_MAX_FIXES: int = 5

    PROJECT_PATH: Path = Path(".")

    EXC_INFO: bool = False

    class Config:
        """Pydantic config class."""

        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Initialize settings instance
settings = Settings()
