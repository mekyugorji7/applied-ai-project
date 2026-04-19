"""Centralized logging setup for the music recommender system.

Usage in any module:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("message")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_LOG_DIR = _ROOT / "logs"
_LOG_FILE = _LOG_DIR / "app.log"

_configured = False


def _setup() -> None:
    global _configured
    if _configured:
        return

    _LOG_DIR.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("myVibe")
    root.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — DEBUG and above (full trace)
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            _LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        root.warning("Could not open log file %s; file logging disabled.", _LOG_FILE)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'myVibe' namespace."""
    _setup()
    # Strip leading 'src.' so log names are concise: recommender, rag, pipeline …
    short = name.replace("src.", "").replace("__main__", "main")
    return logging.getLogger(f"myVibe.{short}")
