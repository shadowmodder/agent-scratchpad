"""Persistent vector memory for long-running agents."""
from __future__ import annotations
import json
import os
import time
from typing import Callable

from .similarity import cosine_similarity


class Scratchpad:
    """Store and retrieve agent memories by semantic similarity.

    Memories are (key, text, embedding) triples. Retrieval ranks all stored
    memories by cosine similarity to the query embedding and returns the top k.

    The embedding function is pluggable — pass any callable that maps a string
    to a list of floats::

        import anthropic
        client = anthropic.Anthropic()

        def embed(text: str) -> list[float]:
            r = client.embeddings.create(model="voyage-3", input=[text])
            return r.embeddings[0].embedding

        pad = Scratchpad(embed_fn=embed, persist_path="memory.json")
        pad.remember("project_goal", "Build a RAG pipeline for legal documents")
        results = pad.recall("what is the project about?", k=3)

    For testing without an API, pass a deterministic mock::

        def mock_embed(text: str) -> list[float]:
            # TF-style: bag of character trigrams as a fixed-dim vector
            ...

    Args:
        embed_fn:     Callable[str, list[float]] — any embedding function.
        persist_path: Optional JSON file path. If given, memories survive
                      process restarts. Loaded on construction if the file exists.
        max_memories: Cap on stored memories. When exceeded, the oldest entry
                      is evicted (FIFO). None = unlimited.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        persist_path: str | None = None,
        max_memories: int | None = None,
    ) -> None:
        self._embed = embed_fn
        self._persist_path = persist_path
        self._max = max_memories
        # Ordered list of {"key", "text", "vector", "ts"}
        self._memories: list[dict] = []
        if persist_path and os.path.exists(persist_path):
            self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def remember(self, key: str, text: str) -> None:
        """Embed ``text`` and store it under ``key``.

        If ``key`` already exists it is replaced in-place (preserving order).
        """
        vector = self._embed(text)
        for i, m in enumerate(self._memories):
            if m["key"] == key:
                self._memories[i] = {"key": key, "text": text, "vector": vector, "ts": time.time()}
                self._save()
                return
        if self._max and len(self._memories) >= self._max:
            self._memories.pop(0)  # evict oldest
        self._memories.append({"key": key, "text": text, "vector": vector, "ts": time.time()})
        self._save()

    def recall(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k most similar memories to ``query``.

        Each result is ``{"key": str, "text": str, "score": float}``, sorted
        highest-score first. Returns an empty list when no memories are stored.
        """
        if not self._memories:
            return []
        qvec = self._embed(query)
        scored = [
            (cosine_similarity(qvec, m["vector"]), m["key"], m["text"])
            for m in self._memories
        ]
        scored.sort(reverse=True)
        return [{"key": k_, "text": t, "score": round(s, 6)} for s, k_, t in scored[:k]]

    def forget(self, key: str) -> bool:
        """Remove the memory with the given key. Returns True if it existed."""
        before = len(self._memories)
        self._memories = [m for m in self._memories if m["key"] != key]
        if len(self._memories) < before:
            self._save()
            return True
        return False

    def keys(self) -> list[str]:
        return [m["key"] for m in self._memories]

    def __len__(self) -> int:
        return len(self._memories)

    def __contains__(self, key: str) -> bool:
        return any(m["key"] == key for m in self._memories)

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        if not self._persist_path:
            return
        data = [{"key": m["key"], "text": m["text"], "vector": m["vector"], "ts": m["ts"]}
                for m in self._memories]
        with open(self._persist_path, "w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        with open(self._persist_path) as f:
            content = f.read().strip()
        if not content:
            return
        data = json.loads(content)
        self._memories = [
            {"key": d["key"], "text": d["text"], "vector": d["vector"], "ts": d.get("ts", 0.0)}
            for d in data
        ]
