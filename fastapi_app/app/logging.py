from __future__ import annotations

import logging
import sys

import structlog

from .config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog-based JSON logging."""
    timestamper = structlog.processors.TimeStamper(fmt="iso", key="ts")
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
    ]
    if settings.json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:  # pragma: no cover - human-friendly logs
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        force=True,
    )
