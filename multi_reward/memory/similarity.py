"""
Cosine similarity computation for evidence board feature vectors.

Used by the memory system to find similar past rounds for retrieval.
"""

import math


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two feature vectors.

    Vectors are dicts mapping feature names to numeric values.
    Missing features default to 0.

    Returns:
        Similarity in [0, 1]. Higher = more similar.
    """
    all_keys = set(vec_a.keys()) | set(vec_b.keys())

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for key in all_keys:
        a = vec_a.get(key, 0.0)
        b = vec_b.get(key, 0.0)
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a < 1e-12 and norm_b < 1e-12:
        return 1.0  # both empty vectors
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return max(0.0, min(1.0, sim))


def normalize_feature_vector(fv: dict[str, float]) -> dict[str, float]:
    """Normalize a feature vector to unit length."""
    total = sum(v * v for v in fv.values())
    if total < 1e-12:
        return fv
    norm = math.sqrt(total)
    return {k: v / norm for k, v in fv.items()}
