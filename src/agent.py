"""Agentic workflow: 5 observable steps built on the integrated pipeline."""

from __future__ import annotations

import time
from typing import Dict, List

from .logger import get_logger
from .pipeline import RecommendationPipeline, PipelineResult, validate_prefs
from .rag import KnowledgeBase
from .gemini_client import GeminiClient

log = get_logger(__name__)


class MusicAgent:
    """
    5-step agentic music recommendation workflow.

    Steps 1–3 are diagnostic (profile analysis, KB query, catalog scoring).
    Step 4 calls the integrated RecommendationPipeline — Gemini receives the
    RAG-retrieved context, so its output is knowledge-grounded.
    Step 5 synthesises everything into a final human-readable report.
    """

    def __init__(self, songs: List[Dict], kb: KnowledgeBase, gemini: GeminiClient):
        self.songs = songs
        self.kb = kb
        self.gemini = gemini
        self._pipeline = RecommendationPipeline(songs, kb, gemini)

    def run(self, user_prefs: dict, k: int = 5) -> PipelineResult:
        """Execute all 5 steps, print each header, and return the pipeline result."""
        start = time.time()
        log.info("Agent.run started — k=%d", k)

        clean_prefs = validate_prefs(user_prefs)

        profile_summary, primary_genre, primary_mood, energy_label = (
            self._step_analyze_profile(clean_prefs)
        )
        retrieved_docs = self._step_query_kb(clean_prefs, primary_genre, primary_mood, energy_label)
        top_songs_preview = self._step_score_catalog(clean_prefs, k)

        # Step 4 — run the full integrated pipeline (RAG + Gemini together)
        result = self._step_run_pipeline(clean_prefs, k)

        self._step_synthesize(result, time.time() - start)

        log.info("Agent.run completed in %.2fs", time.time() - start)
        return result

    # ── Step 1 ────────────────────────────────────────────────────────────────
    def _step_analyze_profile(self, prefs: dict):
        print("\n[AGENT STEP 1/5] Analyzing user profile...")
        log.debug("Step 1: analyzing prefs=%s", prefs)

        genre = prefs.get("favorite_genre", "unknown")
        mood = prefs.get("favorite_mood", "unknown")
        e = prefs.get("target_energy")
        acoustic = prefs.get("likes_acoustic")

        if e is not None:
            if e < 0.35:
                e_label = f"VERY LOW ({e:.2f})"
            elif e < 0.55:
                e_label = f"LOW ({e:.2f})"
            elif e < 0.70:
                e_label = f"MODERATE ({e:.2f})"
            else:
                e_label = f"HIGH ({e:.2f})"
        else:
            e_label = "N/A"

        acoustic_label = "YES" if acoustic is True else ("NO" if acoustic is False else "N/A")
        parts = [x for x in [genre, mood] if x != "unknown"]
        if e is not None and e < 0.55:
            parts.append("low-energy")
        elif e is not None:
            parts.append("high-energy")
        if acoustic is True:
            parts.append("acoustic-leaning")
        summary = ", ".join(parts) + " listener" if parts else "general listener"

        print(f"  -> Genre: {genre} | Mood: {mood} | Energy: {e_label} | Acoustic: {acoustic_label}")
        print(f'  -> Profile summary: "{summary}"')
        return summary, genre, mood, e_label

    # ── Step 2 ────────────────────────────────────────────────────────────────
    def _step_query_kb(self, prefs: dict, genre: str, mood: str, e_label: str) -> List[Dict]:
        print("\n[AGENT STEP 2/5] Querying knowledge base...")
        log.debug("Step 2: building RAG query")

        tokens = [x for x in [genre, mood] if x != "unknown"]
        if "LOW" in e_label:
            tokens.append("low energy slow tempo acoustic")
        elif "HIGH" in e_label:
            tokens.append("high energy upbeat")
        if prefs.get("likes_acoustic") is True:
            tokens.append("acoustic")
        query = " ".join(tokens) or "music"

        print(f'  -> Query: "{query}"')
        docs = self.kb.retrieve(query, top_k=3)
        for doc in docs:
            print(f"  -> {doc['filename']} (relevance: {doc['score']:.2f})")
        return docs

    # ── Step 3 ────────────────────────────────────────────────────────────────
    def _step_score_catalog(self, prefs: dict, k: int) -> List:
        from .recommender import recommend_songs
        print(f"\n[AGENT STEP 3/5] Scoring local catalog ({len(self.songs)} songs)...")
        log.debug("Step 3: scoring catalog")

        results = recommend_songs(prefs, self.songs, k=k)
        top_names = ", ".join(f"{s['title']} ({sc:.2f})" for s, sc, _ in results[:3])
        print(f"  -> Top {k}: {top_names}{'...' if len(results) > 3 else ''}")
        return results

    # ── Step 4 ────────────────────────────────────────────────────────────────
    def _step_run_pipeline(self, prefs: dict, k: int) -> PipelineResult:
        print("\n[AGENT STEP 4/5] Running integrated pipeline (RAG → Gemini)...")
        log.info("Step 4: running RecommendationPipeline")

        result = self._pipeline.run(prefs, k=k)

        if result.error:
            print(f"  -> Pipeline error: {result.error}")
        elif not self.gemini.enabled:
            print("  -> Gemini disabled — RAG context retrieved but no AI explanation generated.")
            print(f"  -> Retrieved docs: {[d['filename'] for d in result.retrieved_docs]}")
        else:
            print(f"  -> RAG query: \"{result.rag_query}\"")
            print(f"  -> Docs used: {[d['filename'] for d in result.retrieved_docs]}")
            # Show first 200 chars of explanation as a preview
            preview = result.explanation[:200].replace("\n", " ")
            print(f"  -> AI explanation (preview): {preview}...")
        return result

    # ── Step 5 ────────────────────────────────────────────────────────────────
    def _step_synthesize(self, result: PipelineResult, elapsed: float) -> None:
        print("\n[AGENT STEP 5/5] Synthesizing final recommendations...")

        print("\n  [Local Catalog Top Songs]")
        for i, (song, score, _) in enumerate(result.top_songs, 1):
            print(f"    {i}. {song['title']:<30} | {song['artist']:<20} | Score: {score:.2f}")

        print("\n  [Knowledge Base Docs Used]")
        for doc in result.retrieved_docs:
            snippet = doc["content"][:100].replace("\n", " ")
            print(f"    {doc['filename']}: {snippet}...")

        if result.explanation and not result.explanation.startswith("["):
            print("\n  [AI Explanation — RAG-Grounded]")
            for line in result.explanation.strip().splitlines():
                print(f"    {line}")

        print(f"\n  Agent completed 5 steps in {elapsed:.2f}s.")


def run_agent(songs: List[Dict]) -> None:
    """Entry point called from main.py --agent flag."""
    print("\n" + "=" * 60)
    print("AGENTIC MUSIC RECOMMENDATION WORKFLOW")
    print("=" * 60)

    from .rag import KnowledgeBase
    from .gemini_client import GeminiClient
    from .main import TASTE_PROFILE

    kb = KnowledgeBase()
    gemini = GeminiClient()
    agent = MusicAgent(songs, kb, gemini)
    agent.run(TASTE_PROFILE)
