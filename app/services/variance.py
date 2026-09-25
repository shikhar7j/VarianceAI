from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import pandas as pd

from app import llm_client
from app.config import settings

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"line_item", "department", "budget", "actual"}
REVENUE_LINE_ITEMS = {"revenue", "sales"}


class VarianceDataError(ValueError):
    """Raised when the input CSV is missing required columns or is malformed."""


@dataclass(frozen=True)
class VarianceResult:
    line_item: str
    department: str
    budget: float
    actual: float
    variance: float
    variance_pct: float
    direction: str  # "over" | "under"
    commentary: str
    mode: str  # "live" | "fallback"

    def to_dict(self) -> dict:
        return asdict(self)


class VarianceAnalyzer:
    """Computes variances from a budget/actuals CSV and attaches commentary."""

    def __init__(self, csv_path: str):
        self._csv_path = csv_path

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self._csv_path)
        except FileNotFoundError as exc:
            raise VarianceDataError(f"Data file not found: {self._csv_path}") from exc

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise VarianceDataError(f"CSV is missing required column(s): {sorted(missing)}")

        return df

    @staticmethod
    def _compute(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["variance"] = df["actual"] - df["budget"]
        df["variance_pct"] = (df["variance"] / df["budget"].replace(0, pd.NA) * 100).fillna(0).round(1)
        df["direction"] = df["variance"].apply(lambda v: "over" if v > 0 else "under")
        return df

    @staticmethod
    def _is_unfavorable(line_item: str, direction: str) -> bool:
        is_revenue = line_item.strip().lower() in REVENUE_LINE_ITEMS
        if is_revenue:
            return direction == "under"
        return direction == "over"

    def _fallback_commentary(self, row: pd.Series) -> str:
        pct = abs(row["variance_pct"])

        if pct == 0:
            return (
                f"{row['line_item']} ({row['department']}) came in exactly on budget "
                f"for the period. No variance to investigate."
            )

        severity = "significantly" if pct >= 25 else "moderately" if pct >= 10 else "slightly"
        unfavorable = self._is_unfavorable(row["line_item"], row["direction"])
        tone = "unfavorable" if unfavorable else "favorable"
        article = "an" if tone == "unfavorable" else "a"

        return (
            f"{row['line_item']} ({row['department']}) came in {severity} {row['direction']} budget "
            f"by {pct}% (${abs(row['variance']):,.0f}), {article} {tone} variance. "
            f"Recommend reviewing the {row['department'].lower()} team's drivers for this period."
        )

    def _llm_commentary(self, row: pd.Series) -> str:
        prompt = (
            "You are an FP&A analyst writing month-end variance commentary for a finance "
            "close pack. Be concise (2 sentences max), professional, and specific.\n\n"
            f"Line item: {row['line_item']}\n"
            f"Department: {row['department']}\n"
            f"Budget: ${row['budget']:,.0f}\n"
            f"Actual: ${row['actual']:,.0f}\n"
            f"Variance: ${row['variance']:,.0f} ({row['variance_pct']}%, {row['direction']} budget)\n\n"
            "Write the variance commentary:"
        )
        try:
            return llm_client.complete(prompt, max_tokens=120, temperature=0.4)
        except Exception:
            logger.warning("LLM commentary failed for %s, falling back to template", row["line_item"])
            return self._fallback_commentary(row)

    def run(self) -> list[VarianceResult]:
        df = self._compute(self._load())
        mode = "live" if settings.live_mode else "fallback"

        results: list[VarianceResult] = []
        for _, row in df.iterrows():
            commentary = self._llm_commentary(row) if settings.live_mode else self._fallback_commentary(row)
            results.append(
                VarianceResult(
                    line_item=row["line_item"],
                    department=row["department"],
                    budget=float(row["budget"]),
                    actual=float(row["actual"]),
                    variance=float(row["variance"]),
                    variance_pct=float(row["variance_pct"]),
                    direction=row["direction"],
                    commentary=commentary,
                    mode=mode,
                )
            )
        return results