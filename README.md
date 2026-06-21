# ds-toolkit

Reusable, self-contained building blocks for data science / AI engineering work.
Each module under `src/` stands alone — drop it into a project and import what you
need. No framework lock-in, minimal dependencies, typed and documented throughout.

## Modules

| Module | What it gives you | Key dependency |
|---|---|---|
| `data_quality.py` | Declarative data quality checks for pandas (`NotNull`, `Unique`, `InRange`, `InSet`, `MatchesRegex`, `RowCount`) with a structured, serializable report and block/warn severities. | pandas |
| `spark_data_quality.py` | The same idea at scale: every check is a Spark aggregation, evaluated in a **single distributed pass** over the data. | pyspark |
| `llm_client.py` | Provider-agnostic LLM client with automatic **fallback chain** (e.g. Mistral → Gemini → Claude), per-provider retry + exponential backoff, uniform response object. | vendor SDKs (lazy) |
| `rag_retriever.py` | Minimal RAG loop — chunk → embed → store → search — behind swappable `Embedder` / `VectorStore` interfaces. Ships with an in-memory cosine store that runs with zero infra. | numpy |
| `config.py` | Typed app configuration via Pydantic `BaseSettings`: loads from env + `.env`, validates at startup, masks secrets. | pydantic, pydantic-settings |

## Quick start

```bash
pip install -r requirements.txt
python src/data_quality.py      # runnable demo
python src/llm_client.py        # runnable demo (uses a stub, no API key needed)
python src/rag_retriever.py     # runnable demo
pytest -v                       # run the test suite
```

## Examples

**Data quality gate in a pipeline**

```python
from data_quality import DataQualitySuite, NotNull, Unique, InRange

suite = DataQualitySuite([
    NotNull("id"),
    Unique("id"),
    InRange("age", 0, 120),
])
report = suite.run(df)
print(report.to_text())
report.raise_if_failed()        # fail the job on a blocking violation
```

**LLM call with provider fallback**

```python
from llm_client import LLMClient, MistralProvider, AnthropicProvider

client = LLMClient(providers=[
    MistralProvider(model="mistral-large-latest"),
    AnthropicProvider(model="claude-sonnet-4-6"),
])
resp = client.complete("Summarize CVE-2024-1234 in one sentence.")
print(resp.text, "—", resp.provider)
```

**Retrieval for RAG**

```python
from rag_retriever import RAGRetriever, InMemoryVectorStore, HashingEmbedder

rag = RAGRetriever(HashingEmbedder(dim=256), InMemoryVectorStore())
rag.index(my_documents)
context = rag.build_context("my question", k=4)   # prompt-ready context block
```

## Design principles

- **Interfaces over implementations.** Embedders, vector stores, and LLM
  providers sit behind small ABCs, so swapping a vendor never touches calling code.
- **Structured results, not prints.** Reports serialize to JSON for logging,
  monitoring, and CI assertions.
- **Fail fast, fail clearly.** Config validates on load; data quality gates raise
  with actionable messages.
- **Runs out of the box.** Demos and tests use stubs / in-memory stores so the
  repo works with zero external infrastructure or API keys.

## Layout

```
ds-toolkit/
├── src/
│   ├── data_quality.py
│   ├── spark_data_quality.py
│   ├── llm_client.py
│   ├── rag_retriever.py
│   └── config.py
├── tests/
│   └── test_toolkit.py
├── requirements.txt
└── .gitlab-ci.yml
```
