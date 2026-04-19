"""Integrated recommendation pipeline: score → RAG retrieve → AI explain.

This is the single entry point the app, agent, and CLI all use.
Gemini is given the retrieved KB context so its explanation is grounded
in real genre/mood knowledge, not just the song list.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .logger import get_logger
from .recommender import recommend_songs
from .rag import KnowledgeBase
from .gemini_client import GeminiClient

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Input guardrails
# ---------------------------------------------------------------------------

def validate_prefs(prefs: dict) -> dict:
    """
    Sanitize user_prefs before passing to the pipeline.

    - Clamps energy, valence, danceability to [0, 1].
    - Clamps tempo_bpm to [20, 250].
    - Coerces likes_acoustic to bool | None.
    - Strips whitespace from string fields.
    - Warns about (but does not block) out-of-range values.

    Returns a cleaned copy; never mutates the input.
    """
    cleaned = dict(prefs)

    for float_key, lo, hi in [
        ("target_energy", 0.0, 1.0),
        ("target_valence", 0.0, 1.0),
        ("target_danceability", 0.0, 1.0),
    ]:
        if float_key in cleaned and cleaned[float_key] is not None:
            try:
                val = float(cleaned[float_key])
                if not (lo <= val <= hi):
                    log.warning(
                        "Guardrail: %s=%s is outside [%s, %s]; clamping.",
                        float_key, val, lo, hi,
                    )
                    val = max(lo, min(hi, val))
                cleaned[float_key] = val
            except (TypeError, ValueError):
                log.warning("Guardrail: %s=%r is not numeric; removing.", float_key, cleaned[float_key])
                del cleaned[float_key]

    if "target_tempo_bpm" in cleaned and cleaned["target_tempo_bpm"] is not None:
        try:
            bpm = float(cleaned["target_tempo_bpm"])
            if not (20.0 <= bpm <= 250.0):
                log.warning("Guardrail: target_tempo_bpm=%s clamped to [20, 250].", bpm)
                bpm = max(20.0, min(250.0, bpm))
            cleaned["target_tempo_bpm"] = bpm
        except (TypeError, ValueError):
            log.warning("Guardrail: target_tempo_bpm is not numeric; removing.")
            del cleaned["target_tempo_bpm"]

    for str_key in ("favorite_genre", "favorite_mood"):
        if str_key in cleaned and cleaned[str_key] is not None:
            cleaned[str_key] = str(cleaned[str_key]).strip().lower()

    # Coerce likes_acoustic: accept bool, "True"/"False" strings, None
    if "likes_acoustic" in cleaned:
        la = cleaned["likes_acoustic"]
        if isinstance(la, str):
            if la.lower() in ("true", "1", "yes"):
                cleaned["likes_acoustic"] = True
            elif la.lower() in ("false", "0", "no"):
                cleaned["likes_acoustic"] = False
            else:
                log.warning("Guardrail: likes_acoustic=%r unrecognised; setting None.", la)
                cleaned["likes_acoustic"] = None

    return cleaned


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    top_songs: List  # list of (song_dict, score, reasons)
    retrieved_docs: List[Dict] = field(default_factory=list)
    rag_query: str = ""
    explanation: str = ""   # RAG-grounded AI explanation (empty if Gemini unavailable)
    gemini_enabled: bool = False
    elapsed_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RecommendationPipeline:
    """
    Fuses: weighted scoring → TF-IDF RAG retrieval → Gemini explanation.

    The AI explanation is generated AFTER retrieval so Gemini receives the
    relevant genre/mood knowledge as context.  This means the explanation
    is grounded in the knowledge base, not produced from scratch.
    """

    def __init__(
        self,
        songs: List[Dict],
        kb: KnowledgeBase,
        gemini: GeminiClient,
    ):
        self.songs = songs
        self.kb = kb
        self.gemini = gemini
        log.info(
            "Pipeline ready — catalog: %d songs | KB: %d docs | Gemini: %s",
            len(songs),
            len(kb._docs),
            "enabled" if gemini.enabled else "disabled",
        )

    def run(self, user_prefs: dict, k: int = 5) -> PipelineResult:
        """Execute the full pipeline and return a structured result."""
        t0 = time.time()

        # ── 1. Validate / sanitize inputs ───────────────────────────────
        clean_prefs = validate_prefs(user_prefs)
        genre = clean_prefs.get("favorite_genre", "unknown")
        mood = clean_prefs.get("favorite_mood", "unknown")
        log.info("Pipeline.run — genre=%s, mood=%s, energy=%.2f",
                 genre, mood, clean_prefs.get("target_energy", float("nan")))

        # ── 2. Score catalog (pure Python, no API) ──────────────────────
        try:
            top_songs = recommend_songs(clean_prefs, self.songs, k=k)
            log.debug(
                "Top-%d scored: %s",
                k,
                ", ".join(f"{s['title']}({sc:.2f})" for s, sc, _ in top_songs),
            )
        except Exception as exc:
            log.error("Scoring failed: %s", exc, exc_info=True)
            return PipelineResult(top_songs=[], error=str(exc))

        # ── 3. RAG retrieval ─────────────────────────────────────────────
        query_tokens = [x for x in [genre, mood] if x != "unknown"]
        e = clean_prefs.get("target_energy", 0.5)
        if e < 0.55:
            query_tokens.append("low energy acoustic slow")
        else:
            query_tokens.append("high energy upbeat")
        if clean_prefs.get("likes_acoustic") is True:
            query_tokens.append("acoustic warm organic")
        rag_query = " ".join(query_tokens) or "music recommendation"

        try:
            retrieved_docs = self.kb.retrieve(rag_query, top_k=3)
            log.info(
                "RAG retrieved: %s",
                ", ".join(f"{d['filename']}({d['score']:.2f})" for d in retrieved_docs),
            )
        except Exception as exc:
            log.warning("RAG retrieval failed: %s; continuing without context.", exc)
            retrieved_docs = []

        # ── 4. Build RAG-grounded Gemini prompt ──────────────────────────
        if not self.gemini.enabled:
            log.info("Gemini disabled; skipping explanation.")
            elapsed = (time.time() - t0) * 1000
            return PipelineResult(
                top_songs=top_songs,
                retrieved_docs=retrieved_docs,
                rag_query=rag_query,
                explanation="",
                gemini_enabled=False,
                elapsed_ms=round(elapsed, 1),
            )

        song_lines = "\n".join(
            f"  {i}. \"{s['title']}\" by {s['artist']} "
            f"[genre={s['genre']}, mood={s['mood']}, energy={s['energy']:.2f}, "
            f"BPM={s['tempo_bpm']}, acousticness={s['acousticness']:.2f}]"
            for i, (s, _, _) in enumerate(top_songs, 1)
        )

        kb_context = "\n\n".join(
            f"[{doc['filename']}]\n{doc['content']}"
            for doc in retrieved_docs
        )

        prompt = (
            "You are an expert music recommender assistant with deep knowledge of "
            "music theory, psychoacoustics, and genre characteristics.\n\n"
            "## Listener Profile\n"
            f"- Favorite genre: {genre}\n"
            f"- Favorite mood: {mood}\n"
            f"- Target energy: {clean_prefs.get('target_energy', 'N/A')}\n"
            f"- Target BPM: {clean_prefs.get('target_tempo_bpm', 'N/A')}\n"
            f"- Target valence: {clean_prefs.get('target_valence', 'N/A')}\n"
            f"- Likes acoustic: {clean_prefs.get('likes_acoustic', 'N/A')}\n\n"
            "## Music Knowledge (use this to inform your explanation)\n"
            f"{kb_context}\n\n"
            "## Recommended Songs\n"
            f"{song_lines}\n\n"
            "## Your Task\n"
            "Using the music knowledge above, write a 3–5 sentence explanation of why "
            "these specific songs suit this listener. Reference concrete audio features "
            "(BPM range, energy level, acousticness, mood characteristics) from the "
            "knowledge documents to justify each recommendation. Do not just restate "
            "the song metadata — explain the musical reasoning."
        )

        log.debug("Gemini prompt length: %d chars", len(prompt))
        explanation = self.gemini.generate(prompt, max_tokens=400)
        log.info("Gemini explanation generated (%d chars).", len(explanation))

        elapsed = (time.time() - t0) * 1000
        log.info("Pipeline completed in %.0f ms.", elapsed)

        return PipelineResult(
            top_songs=top_songs,
            retrieved_docs=retrieved_docs,
            rag_query=rag_query,
            explanation=explanation,
            gemini_enabled=True,
            elapsed_ms=round(elapsed, 1),
        )
