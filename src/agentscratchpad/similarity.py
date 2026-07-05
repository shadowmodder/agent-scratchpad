"""Cosine similarity — uses numpy if available, falls back to pure Python."""
from __future__ import annotations
import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns [-1, 1]."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    try:
        import numpy as np
        av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        na, nb = np.linalg.norm(av), np.linalg.norm(bv)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(av, bv) / (na * nb))
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
