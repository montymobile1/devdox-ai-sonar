"""Application-wide settings loaded from ``.env`` / environment variables.

The :class:`Settings` class declares every environment-sourced value the
CLI or library code reads. It inherits from ``pydantic_settings.BaseSettings``,
which automatically reads values from (in order of precedence):

1. Fields passed explicitly to ``Settings(...)`` (used in tests).
2. Process environment variables.
3. A ``.env`` file in the working directory.
4. The field's declared default.

LLM precedence across the application
-------------------------------------
The LLM configuration specifically has a *further* precedence chain that
spans multiple config sources, resolved at CLI runtime (not here):

1. CLI flags (``--llm-model`` etc.) -- highest.
2. Environment variables (``LLM_MODEL``, ``LLM_API_KEY``, ``LLM_BASE_URL``)
   read via this :class:`Settings` object.
3. The ``default_profile`` entry in ``~/devdox/config.toml``.

Fields in this module cover only layer 2. Layer 1 lives in :mod:`cli`;
layer 3 lives in :mod:`llm.profile_store`.

Under DEVDOX-63 there are no provider-specific environment variables.
``LLM_MODEL`` carries the provider via its prefix (``"gemini/..."`` /
``"openai/..."`` etc.), and the single ``LLM_API_KEY`` covers whichever
provider that is.
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""

    VERSION: str = "0.0.7"

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

    # LLM configuration (see module docstring for precedence details).
    # An empty LLM_MODEL means "no env-level default; fall back to
    # ~/devdox/config.toml's default_profile".
    LLM_MODEL: str = ""
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""

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
