def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def retrieval_confidence(scores: list[float], min_similarity: float) -> float:
    """Normalize cosine retrieval quality into a conservative 0..1 score."""
    if not scores:
        return 0.0

    best = max(scores)
    top_scores = sorted(scores, reverse=True)[:3]
    mean_top = sum(top_scores) / len(top_scores)

    # A threshold-level match should not look highly confident.
    denominator = max(1e-6, 1.0 - min_similarity)
    best_norm = clamp((best - min_similarity) / denominator)
    mean_norm = clamp((mean_top - min_similarity) / denominator)

    return round(clamp(0.65 * best_norm + 0.35 * mean_norm), 4)


def combined_confidence(
    retrieval_score: float,
    grounding_score: float,
    grounded: bool,
) -> float:
    if not grounded:
        return round(clamp(retrieval_score * 0.25), 4)

    return round(clamp(0.60 * retrieval_score + 0.40 * grounding_score), 4)
