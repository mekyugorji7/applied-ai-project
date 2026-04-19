"""Shared Gemini API wrapper using the google-genai SDK.

Key resolution order:
  1. GEMINI_API_KEY environment variable
  2. .env file in the project root (loaded via python-dotenv)
  3. Streamlit st.secrets["GEMINI_API_KEY"] (when running inside Streamlit)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from .logger import get_logger

log = get_logger(__name__)

# Load .env from the project root
_ROOT = Path(__file__).parent.parent
_ENV_FILE = _ROOT / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_FILE, override=False)
    log.debug("Loaded .env from %s", _ENV_FILE)
except ImportError:
    log.debug("python-dotenv not installed; skipping .env load.")


def _resolve_api_key() -> str:
    """Return the API key from env or Streamlit secrets, or empty string."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key and key != "your_api_key_here":
        return key

    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "").strip()
        if key and key != "your_api_key_here":
            log.debug("API key resolved from Streamlit secrets.")
            return key
    except Exception:
        pass

    return ""


def _parse_retry_delay(error_message: str) -> float:
    """Extract the suggested retry delay (seconds) from a 429 error message."""
    match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", str(error_message), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


class GeminiClient:
    """Thin wrapper around the google-genai SDK with logging and error guardrails."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model_name = model
        self.enabled = False
        self._client = None
        self._api_key = _resolve_api_key()

        if not self._api_key:
            log.warning("GEMINI_API_KEY not set. AI explanations will be unavailable.")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            self.enabled = True
            log.info("Gemini client ready (model=%s).", model)
        except Exception as exc:
            log.error("Failed to initialise Gemini client: %s", exc)
            self.enabled = False

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Call Gemini and return the response text.

        Handles common failure modes explicitly:
        - Not configured → clear message
        - 429 rate limit → message with retry hint
        - Other API errors → message with error detail
        """
        if not self.enabled or self._client is None:
            return "[Gemini unavailable — add your GEMINI_API_KEY to .env]"

        t0 = time.time()
        try:
            from google import genai
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                ),
            )
            elapsed = time.time() - t0
            text = response.text.strip()
            log.info(
                "Gemini call succeeded in %.2fs — %d chars returned.",
                elapsed, len(text),
            )
            return text

        except Exception as exc:
            elapsed = time.time() - t0
            err_str = str(exc)

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                delay = _parse_retry_delay(err_str)
                hint = f" (suggested retry in {delay:.0f}s)" if delay else ""
                log.warning("Gemini rate limit hit after %.2fs%s.", elapsed, hint)
                return (
                    f"[Gemini rate limit reached{hint} — "
                    "please wait a moment and try again]"
                )

            if "API_KEY_INVALID" in err_str or "401" in err_str:
                log.error("Gemini API key is invalid.")
                return "[Gemini error: invalid API key — check your GEMINI_API_KEY in .env]"

            log.error("Gemini call failed after %.2fs: %s", elapsed, exc, exc_info=False)
            return f"[Gemini error: {exc}]"
