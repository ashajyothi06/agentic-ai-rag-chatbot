from app.llm import _extract_json_object


def test_extract_plain_json():
    result = _extract_json_object(
        '{"grounded": true, "score": 0.9, "reason": "supported"}'
    )
    assert result["grounded"] is True
    assert result["score"] == 0.9


def test_extract_fenced_json():
    result = _extract_json_object(
        '```json\n{"grounded": false, "score": 0.2, "reason": "no"}\n```'
    )
    assert result["grounded"] is False
