"""Make this service's own log lines reach the log.

Nothing here configured logging, and uvicorn configures only its own loggers —
so every `logger.info` in this codebase went to a root logger with no handler
and was dropped. What was lost is exactly the part worth reading: the search
terms each country was actually queried for, the five quality measures, and the
line saying the hallucination judge is not configured. The container printed
access lines and nothing else.

It passed every test, because pytest's caplog attaches its own handler — so the
assertions saw lines that production never wrote.
"""

from __future__ import annotations

import logging
import sys

from app.config import get_settings

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging() -> None:
    """Send this application's logs to stdout at the configured level.

    Applied to the root logger so every module's `getLogger(__name__)` inherits
    it, and idempotent so an import at a second entry point does not double
    every line.
    """
    level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    if not any(getattr(h, "_marketmind", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        handler._marketmind = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    # httpx logs one INFO line per request, and a run makes hundreds of tool
    # calls. Left at INFO it buries the lines this exists to surface.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
