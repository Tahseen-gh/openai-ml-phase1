from __future__ import annotations

from typing import Any


def problem(title: str, status: int, request_id: str, detail: str | None = None) -> dict[str, Any]:
    """Create a RFC 7807 Problem Details payload."""
    return {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail or title,
        "request_id": request_id,
    }
