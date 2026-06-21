"""
spark_data_quality.py
======================

Data quality checks for large datasets using PySpark.

Same idea as the pandas ``data_quality`` module, but every check is expressed
as a Spark aggregation so it runs distributed and never pulls the full dataset
to the driver. Each check contributes one or more aggregate expressions; they
are evaluated together in a *single pass* over the data, then assembled into a
report.

Why a single pass: running each check as its own ``.filter().count()`` would
scan the data N times. Building a list of aggregate columns and calling
``df.agg(*exprs)`` once is dramatically cheaper on big data.

Requires: pyspark>=3

Example
-------
    from pyspark.sql import SparkSession
    from spark_data_quality import (
        SparkDQSuite, NotNull, InRange, InSet, DistinctRatio,
    )

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet("s3://.../events")

    suite = SparkDQSuite([
        NotNull("user_id"),
        InRange("age", 0, 120),
        InSet("status", ["active", "frozen", "closed"]),
        DistinctRatio("user_id", min_ratio=0.99),   # near-unique
    ])
    report = suite.run(df)
    print(report.to_text())
    report.raise_if_failed()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Column


@dataclass
class SparkCheckResult:
    name: str
    column: str | None
    passed: bool
    failed_count: int
    total_count: int
    message: str


class SparkCheck(ABC):
    """A check that contributes aggregate expressions evaluated in one pass."""

    @abstractmethod
    def agg_exprs(self) -> list[Column]:
        """Aliased aggregate columns this check needs (e.g. count of bad rows)."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, row: dict, total: int) -> SparkCheckResult:
        """Turn the collected aggregate row + total count into a result."""
        raise NotImplementedError


class NotNull(SparkCheck):
    def __init__(self, column: str) -> None:
        self.column = column
        self._key = f"__nn_{column}"

    def agg_exprs(self) -> list[Column]:
        return [F.sum(F.col(self.column).isNull().cast("long")).alias(self._key)]

    def evaluate(self, row: dict, total: int) -> SparkCheckResult:
        failed = int(row.get(self._key) or 0)
        return SparkCheckResult(
            name="NotNull", column=self.column, passed=failed == 0,
            failed_count=failed, total_count=total,
            message=f"{failed} null value(s) in '{self.column}'.",
        )


class InRange(SparkCheck):
    def __init__(self, column: str, min_value: float | None = None,
                 max_value: float | None = None) -> None:
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self._key = f"__rng_{column}"

    def agg_exprs(self) -> list[Column]:
        c = F.col(self.column)
        bad = c.isNull()
        if self.min_value is not None:
            bad = bad | (c < self.min_value)
        if self.max_value is not None:
            bad = bad | (c > self.max_value)
        return [F.sum(bad.cast("long")).alias(self._key)]

    def evaluate(self, row: dict, total: int) -> SparkCheckResult:
        failed = int(row.get(self._key) or 0)
        return SparkCheckResult(
            name="InRange", column=self.column, passed=failed == 0,
            failed_count=failed, total_count=total,
            message=f"{failed} value(s) in '{self.column}' outside "
                    f"[{self.min_value}, {self.max_value}].",
        )


class InSet(SparkCheck):
    def __init__(self, column: str, allowed: list) -> None:
        self.column = column
        self.allowed = allowed
        self._key = f"__set_{column}"

    def agg_exprs(self) -> list[Column]:
        c = F.col(self.column)
        bad = ~c.isin(self.allowed) | c.isNull()
        return [F.sum(bad.cast("long")).alias(self._key)]

    def evaluate(self, row: dict, total: int) -> SparkCheckResult:
        failed = int(row.get(self._key) or 0)
        return SparkCheckResult(
            name="InSet", column=self.column, passed=failed == 0,
            failed_count=failed, total_count=total,
            message=f"{failed} value(s) in '{self.column}' not in allowed set.",
        )


class DistinctRatio(SparkCheck):
    """Assert (distinct count / total) >= min_ratio — e.g. near-unique keys."""

    def __init__(self, column: str, min_ratio: float = 1.0) -> None:
        self.column = column
        self.min_ratio = min_ratio
        self._key = f"__dist_{column}"

    def agg_exprs(self) -> list[Column]:
        # approx_count_distinct is far cheaper than exact at scale
        return [F.approx_count_distinct(F.col(self.column)).alias(self._key)]

    def evaluate(self, row: dict, total: int) -> SparkCheckResult:
        distinct = int(row.get(self._key) or 0)
        ratio = distinct / total if total else 0.0
        passed = ratio >= self.min_ratio
        return SparkCheckResult(
            name="DistinctRatio", column=self.column, passed=passed,
            failed_count=0 if passed else total - distinct, total_count=total,
            message=f"distinct ratio {ratio:.4f} "
                    f"({'>=' if passed else '<'} {self.min_ratio}).",
        )


@dataclass
class SparkDQReport:
    results: list[SparkCheckResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_text(self) -> str:
        lines = ["Spark Data Quality Report", "=" * 40]
        for r in self.results:
            icon = "PASS" if r.passed else "FAIL"
            lines.append(f"[{icon}] {r.name} [{r.column}] — {r.message}")
        lines += ["=" * 40,
                  f"Overall: {'PASSED' if self.passed else 'FAILED'} "
                  f"(rows scanned: {self.results[0].total_count if self.results else 0})"]
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.passed:
            bad = [r for r in self.results if not r.passed]
            raise AssertionError(
                f"{len(bad)} check(s) failed: "
                + "; ".join(f"{r.name}({r.column})" for r in bad))


class SparkDQSuite:
    """Runs all checks in a single distributed pass over the DataFrame."""

    def __init__(self, checks: list[SparkCheck]) -> None:
        self.checks = checks

    def run(self, df: DataFrame) -> SparkDQReport:
        # Cache so the count + agg don't recompute upstream transformations twice.
        df = df.cache()
        total = df.count()
        exprs: list[Column] = []
        for check in self.checks:
            exprs.extend(check.agg_exprs())
        row = df.agg(*exprs).collect()[0].asDict() if exprs else {}
        results = [check.evaluate(row, total) for check in self.checks]
        return SparkDQReport(results)
