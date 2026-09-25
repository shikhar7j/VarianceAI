from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    base_dir: str

    @property
    def live_mode(self) -> bool:
        """True once a usable OpenAI key is configured."""
        return bool(self.openai_api_key)

    @property
    def budget_actuals_path(self) -> str:
        return os.path.join(self.base_dir, "sample_data", "budget_actuals.csv")

    @property
    def financial_statement_path(self) -> str:
        return os.path.join(self.base_dir, "sample_data", "financial_statement.txt")


def load_settings() -> Settings:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        base_dir=base_dir,
    )


settings = load_settings()