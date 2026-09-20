from app.scoring import combined_confidence, retrieval_confidence


def test_retrieval_confidence_empty_is_zero():
    assert retrieval_confidence([], 0.25) == 0.0


def test_retrieval_confidence_is_bounded():
    score = retrieval_confidence([0.91, 0.84, 0.78], 0.25)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_ungrounded_answer_is_penalized():
    grounded = combined_confidence(0.9, 0.9, True)
    not_grounded = combined_confidence(0.9, 0.0, False)
    assert grounded > not_grounded
    assert 0 <= not_grounded <= 1
