from pathlib import Path

import requests

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    destination = Path(settings.source_pdf_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading source PDF from:\n{settings.source_pdf_url}")
    response = requests.get(settings.source_pdf_url, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        raise RuntimeError("Downloaded file does not appear to be a PDF.")

    destination.write_bytes(response.content)
    print(f"Saved: {destination} ({len(response.content):,} bytes)")


if __name__ == "__main__":
    main()
