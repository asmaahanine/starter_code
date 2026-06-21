"""
simple_starters.py
==================

Quick, readable snippets you can copy, run, and adapt on the spot.
No classes, no frameworks — just the core of each task in plain functions.
For polished/production versions, see the ds-toolkit modules.

Run any section by calling it under `if __name__ == "__main__"` at the bottom.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# 1. DATA QUALITY — quick checks on a DataFrame
# ---------------------------------------------------------------------------
def check_data(df):
    """Print a quick health summary of a DataFrame."""
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n")

    # Nulls per column
    nulls = df.isnull().sum()
    print("Nulls per column:")
    print(nulls[nulls > 0] if nulls.any() else "  none")

    # Duplicate rows
    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes}")

    # Quick numeric overview
    print("\nNumeric summary:")
    print(df.describe())


def quick_checks(df):
    """Return a dict of simple pass/fail checks — easy to assert on."""
    return {
        "no_nulls_in_id": df["id"].notnull().all() if "id" in df else None,
        "ids_unique": df["id"].is_unique if "id" in df else None,
        "not_empty": len(df) > 0,
    }


# ---------------------------------------------------------------------------
# 2. CALL AN LLM — the simplest possible version (Anthropic)
# ---------------------------------------------------------------------------
def ask_claude(prompt):
    """Send a prompt to Claude, return the text. Needs ANTHROPIC_API_KEY set."""
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def ask_with_fallback(prompt):
    """Try Claude; if it fails, fall back to OpenAI. Minimal version."""
    try:
        return ask_claude(prompt)
    except Exception as e:
        print(f"Claude failed ({e}), trying OpenAI...")
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 3. SUPER-SIMPLE RAG — find the most relevant text, no vector DB
# ---------------------------------------------------------------------------
def simple_search(query, documents, top_k=2):
    """
    Naive keyword-overlap search. Good enough for a demo or tiny corpus.
    For real semantic search, use embeddings (see ds-toolkit/rag_retriever.py).
    """
    query_words = set(query.lower().split())
    scored = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        scored.append((overlap, doc))
    scored.sort(reverse=True)
    return [doc for score, doc in scored[:top_k] if score > 0]


def answer_from_docs(query, documents):
    """Tiny RAG: find relevant docs, stuff them into a prompt, ask the LLM."""
    relevant = simple_search(query, documents)
    context = "\n".join(relevant)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer using only the context."
    return ask_claude(prompt)


# ---------------------------------------------------------------------------
# 4. READ A FILE / CONFIG — the no-framework way
# ---------------------------------------------------------------------------
def load_config():
    """Read settings from environment with sensible defaults."""
    import os
    return {
        "model": os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Data quality demo (runs with no API key)
    df = pd.DataFrame({
        "id": [1, 2, 3, 3],
        "age": [25, 130, 40, 30],
        "city": ["Paris", None, "Lyon", "Nice"],
    })
    print("=== DATA QUALITY ===")
    check_data(df)
    print("\nQuick checks:", quick_checks(df))

    # Simple search demo (runs with no API key)
    print("\n=== SIMPLE SEARCH ===")
    docs = [
        "Spark runs computations across a cluster.",
        "Paris is the capital of France.",
        "RAG augments an LLM with retrieved context.",
    ]
    print(simple_search("what does spark do", docs))

    # LLM demos are commented out — uncomment once your API key is set:
    # print(ask_claude("Say hello in one word."))
    # print(answer_from_docs("what does spark do", docs))
