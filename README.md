[![CI](https://github.com/shadowmodder/agent-scratchpad/actions/workflows/ci.yml/badge.svg)](https://github.com/shadowmodder/agent-scratchpad/actions/workflows/ci.yml)

# agent-scratchpad

Persistent vector memory for agents. Store observations during a run, retrieve the most relevant ones by semantic similarity, persist to disk between runs. Pluggable embedding function — bring your own (Anthropic, OpenAI, sentence-transformers, TF-IDF, random for tests).

## Install

```bash
pip install -e .
```

## Quickstart

```python
import anthropic
from agentscratchpad import Scratchpad

client = anthropic.Anthropic()

def embed(text: str) -> list[float]:
    """Embed via Anthropic's voyage-3 model."""
    response = client.embeddings.create(
        model="voyage-3",
        input=[text],
    )
    return response.embeddings[0].embedding

scratchpad = Scratchpad(
    embed_fn=embed,
    persist_path="memory.json",
    max_memories=500,
)

# Store observations during an agent run
scratchpad.remember("user_pref_lang", "The user strongly prefers Python over Go.")
scratchpad.remember("api_key_note", "The API key is stored in ANTHROPIC_API_KEY env var.")
scratchpad.remember("project_goal", "Building a document QA pipeline over internal wiki.")

# Retrieve the most relevant memories at any point
results = scratchpad.recall("what programming language does the user want?", k=2)
for r in results:
    print(f"[{r['score']:.4f}] {r['key']}: {r['text']}")
```

```
[0.9341] user_pref_lang: The user strongly prefers Python over Go.
[0.6127] project_goal: Building a document QA pipeline over internal wiki.
```

Memories persist to `memory.json` automatically. On the next run, they load back in.

## API

```python
# Store
scratchpad.remember(key, text)           # overwrites if key exists
scratchpad.forget(key)                   # remove by key; no-op if missing

# Retrieve
results = scratchpad.recall(query, k=5)  # list of {key, text, score}
scratchpad.all_memories()                # list of {key, text, ts}

# Inspect
len(scratchpad)                          # count of stored memories
```

## Max memories and eviction

```python
scratchpad = Scratchpad(embed_fn=embed, max_memories=100)
```

When the memory store reaches `max_memories`, the oldest memory (by insertion timestamp) is evicted before each new `remember()` call. Recency is preferred over relevance for eviction — a highly-relevant but old memory is evicted before a recent irrelevant one, which matches typical agent behavior (recent context > stale facts).

## Bring your own embedder

The library has no mandatory ML dependency. Anything that returns `list[float]` works:

```python
# TF-IDF cosine (no API needed, good for testing)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = ["placeholder"]
vec = TfidfVectorizer()
vec.fit(corpus)

def tfidf_embed(text: str) -> list[float]:
    return vec.transform([text]).toarray()[0].tolist()

scratchpad = Scratchpad(embed_fn=tfidf_embed)
```

Cosine similarity is computed in pure Python (no NumPy required) with a fast NumPy path if available.

## Testing

```bash
pytest -q
```

```
...................
19 passed in 0.10s
```
