r"""
data_quality.py
================

A lightweight, dependency-light data quality framework for pandas DataFrames.

Define expectations declaratively, run them against a DataFrame, and get a
structured report you can log, assert on in a pipeline, or serialize to JSON.

Design goals
------------
- No heavy dependencies (just pandas).
- Each check is a small, composable object — easy to extend.
- Results are structured (not just printed) so they slot into CI/monitoring.
- A non-zero failure count can fail a job; severities let you warn vs. block.

Example
-------
    import pandas as pd
    from data_quality import (
        DataQualitySuite, NotNull, Unique, InRange, InSet, MatchesRegex, RowCount,
    )

    df = pd.DataFrame({
        "id":     [1, 2, 3, 3],
        "age":    [25, 130, 40, 30],
        "email":  ["a@x.com", "bad", "c@x.com", "d@x.com"],
        "status": ["active", "active", "frozen", "active"],
    })

    suite = DataQualitySuite([
        RowCount(min_rows=1),
        NotNull("id"),
        Unique("id"),
        InRange("age", min_value=0, max_value=120),
        MatchesRegex("email", r"[^@]+@[^@]+\.[^@]+"),
        InSet("status", {"active", "frozen", "closed"}),
    ])

    report = suite.run(df)
    print(report.to_text())
    report.raise_if_failed()        # raises DataQualityError if any blocking check failed
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable

import pandas as pd


class Severity(str, Enum):
    """Whether a failing check should block a pipeline or merely warn."""
    BLOCK = "block"
    WARN = "warn"


@dataclass
class CheckResult:
    """Outcome of a single check against a DataFrame."""
    name: str
    column: str | None
    passed: bool
    severity: Severity
    failed_count: int
    total_count: int
    message: str
    examples: list[Any] = field(default_factory=list)

    @property
    def failed_fraction(self) -> float:
        return self.failed_count / self.total_count if self.total_count else 0.0


class Check(ABC):
    """Base class for all data quality checks."""

    def __init__(self, severity: Severity = Severity.BLOCK) -> None:
        self.severity = severity

    @abstractmethod
    def run(self, df: pd.DataFrame) -> CheckResult:
        """Evaluate the check against ``df`` and return a structured result."""
        raise NotImplementedError

    # -- helpers shared by subclasses ------------------------------------
    def _require_column(self, df: pd.DataFrame, column: str) -> CheckResult | None:
        """Return a failing result if the column is missing, else None."""
        if column not in df.columns:
            return CheckResult(
                name=type(self).__name__,
                column=column,
                passed=False,
                severity=self.severity,
                failed_count=len(df),
                total_count=len(df),
                message=f"Column '{column}' not found in DataFrame.",
            )
        return None

    @staticmethod
    def _examples(series: pd.Series, mask: pd.Series, limit: int = 5) -> list[Any]:
        return series[mask].head(limit).tolist()


class RowCount(Check):
    """Assert the DataFrame has a row count within optional bounds."""

    def __init__(self, min_rows: int | None = None, max_rows: int | None = None,
                 severity: Severity = Severity.BLOCK) -> None:
        super().__init__(severity)
        self.min_rows = min_rows
        self.max_rows = max_rows

    def run(self, df: pd.DataFrame) -> CheckResult:
        n = len(df)
        ok = True
        if self.min_rows is not None and n < self.min_rows:
            ok = False
        if self.max_rows is not None and n > self.max_rows:
            ok = False
        bounds = f"[{self.min_rows}, {self.max_rows}]"
        return CheckResult(
            name="RowCount",
            column=None,
            passed=ok,
            severity=self.severity,
            failed_count=0 if ok else 1,
            total_count=n,
            message=f"Row count {n} {'within' if ok else 'outside'} bounds {bounds}.",
        )


class NotNull(Check):
    """Assert a column has no null/NaN values."""

    def __init__(self, column: str, severity: Severity = Severity.BLOCK) -> None:
        super().__init__(severity)
        self.column = column

    def run(self, df: pd.DataFrame) -> CheckResult:
        if (missing := self._require_column(df, self.column)) is not None:
            return missing
        mask = df[self.column].isna()
        failed = int(mask.sum())
        return CheckResult(
            name="NotNull",
            column=self.column,
            passed=failed == 0,
            severity=self.severity,
            failed_count=failed,
            total_count=len(df),
            message=f"{failed} null value(s) in '{self.column}'.",
            examples=df.index[mask].tolist()[:5],
        )


class Unique(Check):
    """Assert a column (or combination of columns) has no duplicate values."""

    def __init__(self, columns: str | Iterable[str],
                 severity: Severity = Severity.BLOCK) -> None:
        super().__init__(severity)
        self.columns = [columns] if isinstance(columns, str) else list(columns)

    def run(self, df: pd.DataFrame) -> CheckResult:
        for col in self.columns:
            if (missing := self._require_column(df, col)) is not None:
                return missing
        mask = df.duplicated(subset=self.columns, keep=False)
        failed = int(mask.sum())
        label = ", ".join(self.columns)
        return CheckResult(
            name="Unique",
            column=label,
            passed=failed == 0,
            severity=self.severity,
            failed_count=failed,
            total_count=len(df),
            message=f"{failed} duplicate row(s) on ({label}).",
            examples=df[mask][self.columns].head(5).to_dict("records"),
        )


class InRange(Check):
    """Assert a numeric column's values fall within [min_value, max_value]."""

    def __init__(self, column: str, min_value: float | None = None,
                 max_value: float | None = None,
                 severity: Severity = Severity.BLOCK) -> None:
        super().__init__(severity)
        self.column = column
        self.min_value = min_value
        self.max_value = max_value

    def run(self, df: pd.DataFrame) -> CheckResult:
        if (missing := self._require_column(df, self.column)) is not None:
            return missing
        series = pd.to_numeric(df[self.column], errors="coerce")
        mask = pd.Series(False, index=df.index)
        if self.min_value is not None:
            mask |= series < self.min_value
        if self.max_value is not None:
            mask |= series > self.max_value
        mask |= series.isna()  # non-numeric / missing counts as out of range
        failed = int(mask.sum())
        return CheckResult(
            name="InRange",
            column=self.column,
            passed=failed == 0,
            severity=self.severity,
            failed_count=failed,
            total_count=len(df),
            message=f"{failed} value(s) in '{self.column}' outside "
                    f"[{self.min_value}, {self.max_value}].",
            examples=self._examples(df[self.column], mask),
        )


class InSet(Check):
    """Assert all values of a column belong to an allowed set."""

    def __init__(self, column: str, allowed: set[Any],
                 severity: Severity = Severity.BLOCK) -> None:
        super().__init__(severity)
        self.column = column
        self.allowed = set(allowed)

    def run(self, df: pd.DataFrame) -> CheckResult:
        if (missing := self._require_column(df, self.column)) is not None:
            return missing
        mask = ~df[self.column].isin(self.allowed)
        failed = int(mask.sum())
        return CheckResult(
            name="InSet",
            column=self.column,
            passed=failed == 0,
            severity=self.severity,
            failed_count=failed,
            total_count=len(df),
            message=f"{failed} value(s) in '{self.column}' not in allowed set.",
            examples=self._examples(df[self.column], mask),
        )


class MatchesRegex(Check):
    """Assert all non-null values of a column match a regular expression."""

    def __init__(self, column: str, pattern: str,
                 severity: Severity = Severity.BLOCK) -> None:
        super().__init__(severity)
        self.column = column
        self.pattern = re.compile(pattern)

    def run(self, df: pd.DataFrame) -> CheckResult:
        if (missing := self._require_column(df, self.column)) is not None:
            return missing
        as_str = df[self.column].astype("string")
        mask = ~as_str.str.fullmatch(self.pattern) | as_str.isna()
        failed = int(mask.sum())
        return CheckResult(
            name="MatchesRegex",
            column=self.column,
            passed=failed == 0,
            severity=self.severity,
            failed_count=failed,
            total_count=len(df),
            message=f"{failed} value(s) in '{self.column}' do not match "
                    f"/{self.pattern.pattern}/.",
            examples=self._examples(df[self.column], mask),
        )


class DataQualityError(AssertionError):
    """Raised when one or more blocking checks fail."""


@dataclass
class DataQualityReport:
    """Aggregated results of running a suite of checks."""
    results: list[CheckResult]

    @property
    def passed(self) -> bool:
        """True only if no *blocking* check failed (warnings don't count)."""
        return all(r.passed for r in self.results if r.severity is Severity.BLOCK)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def to_text(self) -> str:
        lines = ["Data Quality Report", "=" * 40]
        for r in self.results:
            icon = "PASS" if r.passed else ("WARN" if r.severity is Severity.WARN else "FAIL")
            col = f" [{r.column}]" if r.column else ""
            lines.append(f"[{icon}] {r.name}{col} — {r.message}")
            if not r.passed and r.examples:
                lines.append(f"        e.g. {r.examples}")
        status = "PASSED" if self.passed else "FAILED"
        lines += ["=" * 40, f"Overall: {status} ({self.n_failed} check(s) flagged)"]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_failed": self.n_failed,
            "results": [asdict(r) for r in self.results],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), default=str, **kwargs)

    def raise_if_failed(self) -> None:
        """Raise DataQualityError if any blocking check failed."""
        if not self.passed:
            failed = [r for r in self.results
                      if not r.passed and r.severity is Severity.BLOCK]
            details = "; ".join(f"{r.name}({r.column}): {r.message}" for r in failed)
            raise DataQualityError(f"{len(failed)} blocking check(s) failed: {details}")


class DataQualitySuite:
    """A collection of checks run together against a DataFrame."""

    def __init__(self, checks: list[Check]) -> None:
        self.checks = checks

    def run(self, df: pd.DataFrame) -> DataQualityReport:
        return DataQualityReport([check.run(df) for check in self.checks])


if __name__ == "__main__":
    demo = pd.DataFrame({
        "id":     [1, 2, 3, 3],
        "age":    [25, 130, 40, 30],
        "email":  ["a@x.com", "bad", "c@x.com", "d@x.com"],
        "status": ["active", "active", "frozen", "active"],
    })
    suite = DataQualitySuite([
        RowCount(min_rows=1),
        NotNull("id"),
        Unique("id"),
        InRange("age", min_value=0, max_value=120),
        MatchesRegex("email", r"[^@]+@[^@]+\.[^@]+"),
        InSet("status", {"active", "frozen", "closed"}, severity=Severity.WARN),
    ])
    print(suite.run(demo).to_text())
