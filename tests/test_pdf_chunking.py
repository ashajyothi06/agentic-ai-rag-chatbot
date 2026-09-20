from pathlib import Path

import pytest

from app.pdf_loader import load_and_chunk_pdf


def test_missing_pdf_has_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="download_pdf"):
        load_and_chunk_pdf(tmp_path / "missing.pdf", 900, 150)
