"""Tests for Scratchpad and cosine_similarity. No API key required."""
import json
import math
import os
import pytest
import tempfile

from agentscratchpad import Scratchpad, cosine_similarity


# ── Deterministic test embedding ──────────────────────────────────────────

def _char_embed(text: str, dim: int = 8) -> list[float]:
    """Bag-of-characters embedding: deterministic, no API needed."""
    vec = [0.0] * dim
    for ch in text.lower():
        vec[ord(ch) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@pytest.fixture
def pad():
    return Scratchpad(embed_fn=_char_embed)


# ── cosine_similarity ─────────────────────────────────────────────────────

def test_identical_vectors_score_one():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_score_neg_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


# ── Scratchpad.remember / recall / forget ─────────────────────────────────

def test_remember_and_recall(pad):
    pad.remember("fact1", "python programming language")
    results = pad.recall("python programming", k=1)
    assert len(results) == 1
    assert results[0]["key"] == "fact1"
    assert 0.0 <= results[0]["score"] <= 1.0


def test_recall_empty_returns_empty(pad):
    assert pad.recall("anything") == []


def test_recall_returns_at_most_k(pad):
    for i in range(10):
        pad.remember(f"key{i}", f"memory number {i}")
    results = pad.recall("memory", k=3)
    assert len(results) == 3


def test_recall_sorted_by_score_descending(pad):
    pad.remember("exact", "machine learning")
    pad.remember("distant", "cooking recipes")
    results = pad.recall("machine learning", k=2)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_forget_existing_key(pad):
    pad.remember("tmp", "to be deleted")
    assert pad.forget("tmp") is True
    assert "tmp" not in pad


def test_forget_missing_key(pad):
    assert pad.forget("nonexistent") is False


def test_remember_replaces_existing_key(pad):
    pad.remember("k", "first version")
    pad.remember("k", "second version")
    assert len(pad) == 1
    results = pad.recall("second version", k=1)
    assert results[0]["text"] == "second version"


def test_len(pad):
    assert len(pad) == 0
    pad.remember("a", "alpha")
    pad.remember("b", "beta")
    assert len(pad) == 2
    pad.forget("a")
    assert len(pad) == 1


def test_contains(pad):
    pad.remember("x", "content")
    assert "x" in pad
    assert "y" not in pad


def test_keys(pad):
    pad.remember("k1", "one")
    pad.remember("k2", "two")
    assert set(pad.keys()) == {"k1", "k2"}


def test_max_memories_evicts_oldest(pad):
    pad2 = Scratchpad(embed_fn=_char_embed, max_memories=3)
    for i in range(4):
        pad2.remember(f"key{i}", f"content {i}")
    assert len(pad2) == 3
    assert "key0" not in pad2  # first in, first evicted


# ── Persistence ───────────────────────────────────────────────────────────

def test_persist_and_reload():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        pad1 = Scratchpad(embed_fn=_char_embed, persist_path=path)
        pad1.remember("saved", "this was persisted")

        pad2 = Scratchpad(embed_fn=_char_embed, persist_path=path)
        assert "saved" in pad2
        assert pad2.recall("persisted", k=1)[0]["key"] == "saved"
    finally:
        os.unlink(path)


def test_no_persist_path_does_not_write(tmp_path, pad):
    pad.remember("x", "content")
    assert list(tmp_path.iterdir()) == []


def test_forget_updates_persisted_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        pad = Scratchpad(embed_fn=_char_embed, persist_path=path)
        pad.remember("keep", "stays")
        pad.remember("drop", "leaves")
        pad.forget("drop")

        pad2 = Scratchpad(embed_fn=_char_embed, persist_path=path)
        assert "keep" in pad2
        assert "drop" not in pad2
    finally:
        os.unlink(path)
