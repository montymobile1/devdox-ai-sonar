"""
Configuration settings for the DevDox AI Sonar
"""

from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""

    VERSION: str = "0.0.1"

    API_KEY: str = ""  # Fallback for backward compatibility
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    LLM_PROVIDER: str = "LLM_PROVIDER"
    LLM_MODEL: str = "gemini-2.5-flash"

    SONAR_TOKEN:str=""
    SONAR_ORGANIZATION:str=""
    SONAR_PROJECT_KEY:str=""
    PROJECT_PATH: Path = Path("/your/project/path")

    class Config:
        """Pydantic config class."""

        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Initialize settings instance
settings = Settings()
