"""One shared configuration; architecture-specific options live in extras."""

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import dotenv_values


class AppError(Exception):
    """An actionable application error safe to show to a student."""


@dataclass(frozen=True)
class Settings:
    corpus: Path = Path("data/corpus")
    collection: str = "magenta_course_pdf_v1"
    host: str = "127.0.0.1"
    port: int = 8004
    model: str = "claude-haiku-4-5-20251001"
    architecture: str = "classic_rag"
    extras: dict = field(default_factory=dict)
    top_k: int = 6
    env_file: Path = Path(".env")
    temperature: float | None = None

    def __post_init__(self):
        if not self.collection.strip() or not self.model.startswith("claude-"):
            raise AppError("Select a collection and an explicit Anthropic claude-* model.")
        if not self.architecture.strip() or ":" in self.architecture:
            raise AppError("Use a nonempty architecture name without a colon.")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 20:
            raise AppError("Use top-k 1–20.")
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise AppError("Temperature must be between 0 and 1.")
        if not 1 <= self.port <= 65535:
            raise AppError("Chroma port must be between 1 and 65535.")

    def env(self, name, *, required=False):
        shell_value = (os.getenv(name) or "").strip()
        if shell_value:
            return shell_value
        if self.env_file != Path(".env") and not self.env_file.is_file():
            raise AppError("The selected env file does not exist.")
        values = dotenv_values(self.env_file) if self.env_file.is_file() else {}
        value = (values.get(name) or "").strip()
        if required and not value:
            raise AppError(f"Set {name} in the shell or {self.env_file}.")
        return value

    def api_key(self):
        return self.env("ANTHROPIC_API_KEY", required=True)
