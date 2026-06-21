"""
Tests for the ds-toolkit modules.

Run with:  pytest -v
These cover the dependency-light modules (data_quality, llm_client, rag_retriever).
The Spark module is excluded here since it needs a SparkSession.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_quality import (  # noqa: E402
    DataQualitySuite, DataQualityError, Severity,
    NotNull, Unique, InRange, InSet, MatchesRegex, RowCount,
)
from llm_client import (  # noqa: E402
    LLMClient, LLMProvider, LLMResponse, AllProvidersFailedError,
)
from rag_retriever import (  # noqa: E402
    RAGRetriever, InMemoryVectorStore, HashingEmbedder, chunk_text,
)


# --------------------------------------------------------------------------
# data_quality
# --------------------------------------------------------------------------
@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "age": [25, 40, 30],
        "status": ["active", "frozen", "closed"],
    })


def test_clean_data_passes(clean_df):
    suite = DataQualitySuite([
        NotNull("id"), Unique("id"),
        InRange("age", 0, 120),
        InSet("status", {"active", "frozen", "closed"}),
    ])
    report = suite.run(clean_df)
    assert report.passed
    assert report.n_failed == 0


def test_detects_duplicates():
    df = pd.DataFrame({"id": [1, 1, 2]})
    result = Unique("id").run(df)
    assert not result.passed
    assert result.failed_count == 2


def test_detects_out_of_range():
    df = pd.DataFrame({"age": [25, 200, -5]})
    result = InRange("age", 0, 120).run(df)
    assert not result.passed
    assert result.failed_count == 2


def test_missing_column_fails_gracefully(clean_df):
    result = NotNull("does_not_exist").run(clean_df)
    assert not result.passed
    assert "not found" in result.message


def test_regex_check():
    df = pd.DataFrame({"email": ["a@b.com", "nope"]})
    result = MatchesRegex("email", r"[^@]+@[^@]+\.[^@]+").run(df)
    assert result.failed_count == 1


def test_warn_severity_does_not_block():
    df = pd.DataFrame({"status": ["weird"]})
    suite = DataQualitySuite([
        InSet("status", {"active"}, severity=Severity.WARN),
    ])
    report = suite.run(df)
    assert report.passed          # warn-only failure doesn't flip overall status
    report.raise_if_failed()      # should NOT raise


def test_raise_if_failed_raises():
    df = pd.DataFrame({"id": [1, 1]})
    report = DataQualitySuite([Unique("id")]).run(df)
    with pytest.raises(DataQualityError):
        report.raise_if_failed()


def test_report_serializes_to_json(clean_df):
    report = DataQualitySuite([NotNull("id")]).run(clean_df)
    js = report.to_json()
    assert '"passed": true' in js


def test_rowcount_bounds():
    df = pd.DataFrame({"x": [1, 2]})
    assert RowCount(min_rows=1).run(df).passed
    assert not RowCount(min_rows=5).run(df).passed


# --------------------------------------------------------------------------
# llm_client
# --------------------------------------------------------------------------
class _AlwaysFails(LLMProvider):
    name = "fail"

    def _call(self, prompt, system, **kwargs):
        raise RuntimeError("boom")


class _AlwaysWorks(LLMProvider):
    name = "ok"

    def _call(self, prompt, system, **kwargs):
        return LLMResponse(text="ok", provider=self.name, model=self.model)


def test_fallback_to_second_provider():
    client = LLMClient([
        _AlwaysFails(model="a", max_retries=0),
        _AlwaysWorks(model="b"),
    ])
    resp = client.complete("hi")
    assert resp.provider == "ok"


def test_all_providers_failed_raises():
    client = LLMClient([_AlwaysFails(model="a", max_retries=0)])
    with pytest.raises(AllProvidersFailedError):
        client.complete("hi")


def test_empty_providers_rejected():
    with pytest.raises(ValueError):
        LLMClient([])


def test_retry_then_succeed():
    class _FlakyOnce(LLMProvider):
        name = "flaky"
        calls = 0

        def _call(self, prompt, system, **kwargs):
            type(self).calls += 1
            if self.calls < 2:
                raise RuntimeError("transient")
            return LLMResponse(text="recovered", provider=self.name, model=self.model)

    client = LLMClient([_FlakyOnce(model="x", max_retries=2, backoff_base=0.001)])
    assert client.complete("hi").text == "recovered"


# --------------------------------------------------------------------------
# rag_retriever
# --------------------------------------------------------------------------
def test_chunking_with_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=4, overlap=1)
    assert chunks[0] == "abcd"
    assert chunks[1][0] == "d"   # overlap preserved


def test_chunk_validation():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=4, overlap=4)


def test_index_and_retrieve():
    rag = RAGRetriever(HashingEmbedder(dim=128), InMemoryVectorStore())
    n = rag.index(["spark runs on a cluster", "paris is in france"])
    assert n == 2
    hits = rag.retrieve("cluster spark", k=1)
    assert hits and "spark" in hits[0].text


def test_empty_store_returns_nothing():
    rag = RAGRetriever(HashingEmbedder(), InMemoryVectorStore())
    assert rag.retrieve("anything") == []


def test_build_context_respects_max_chars():
    rag = RAGRetriever(HashingEmbedder(), InMemoryVectorStore())
    rag.index(["a" * 100, "b" * 100, "c" * 100])
    ctx = rag.build_context("a", k=3, max_chars=150)
    assert len(ctx) <= 200  # a couple chunks + separators, not all three
