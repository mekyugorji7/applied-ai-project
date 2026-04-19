"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import sys

try:
    from .recommender import load_songs, recommend_songs
except ImportError:
    # `python src/main.py` — script is not executed as part of package `src`
    from recommender import load_songs, recommend_songs

# Content-based taste profiles: target values compared against each song's features.
# Maps to catalog fields: genre, mood, energy, tempo_bpm, valence, danceability, acousticness.

USER_PREFS_CHILL_LOFI = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.42,
    "target_tempo_bpm": 78.0,
    "target_valence": 0.58,
    "target_danceability": 0.61,
    "likes_acoustic": True,
}

USER_PREFS_UPBEAT_POP = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.85,
    "target_tempo_bpm": 120.0,
    "target_valence": 0.82,
    "target_danceability": 0.80,
    "likes_acoustic": False,
}

USER_PREFS_SOFT_AMBIENT = {
    "favorite_genre": "ambient",
    "favorite_mood": "chill",
    "target_energy": 0.30,
    "target_tempo_bpm": 62.0,
    "target_valence": 0.62,
    "target_danceability": 0.42,
    "likes_acoustic": True,
}

# Adversarial and edge-case profiles for probing scoring behavior.
EDGE_CONFLICTING_AFFECT = {
    "favorite_genre": "classical",
    "favorite_mood": "sad",
    "target_energy": 0.95,
    "target_tempo_bpm": 150.0,
    "target_valence": 0.10,
    "target_danceability": 0.90,
    "likes_acoustic": True,
}

EDGE_OUT_OF_RANGE = {
    "favorite_genre": "metal",
    "favorite_mood": "aggressive",
    "target_energy": 1.8,
    "target_tempo_bpm": -20.0,
    "target_valence": -0.5,
    "target_danceability": 2.0,
    "likes_acoustic": False,
}

EDGE_UNKNOWN_LABELS = {
    "favorite_genre": "hyperpop",
    "favorite_mood": "transcendent",
    "target_energy": 0.50,
    "target_tempo_bpm": 100.0,
    "target_valence": 0.50,
    "target_danceability": 0.50,
    "likes_acoustic": True,
}

EDGE_BOOL_STRING_TRAP = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.40,
    "target_tempo_bpm": 80.0,
    "target_valence": 0.60,
    "target_danceability": 0.60,
    "likes_acoustic": "False",
}

EDGE_NONE_ACOUSTIC = {
    "favorite_genre": "jazz",
    "favorite_mood": "relaxed",
    "target_energy": 0.35,
    "target_tempo_bpm": 90.0,
    "target_valence": 0.70,
    "target_danceability": 0.50,
    "likes_acoustic": None,
}

EDGE_SPARSE = {
    "target_energy": 0.42,
}

# Default profile used by main() (same as USER_PREFS_CHILL_LOFI).
TASTE_PROFILE = USER_PREFS_CHILL_LOFI

PROFILE_SUITE = [
    ("USER_PREFS_CHILL_LOFI", USER_PREFS_CHILL_LOFI),
    ("USER_PREFS_UPBEAT_POP", USER_PREFS_UPBEAT_POP),
    ("USER_PREFS_SOFT_AMBIENT", USER_PREFS_SOFT_AMBIENT),
    ("EDGE_CONFLICTING_AFFECT", EDGE_CONFLICTING_AFFECT),
    ("EDGE_OUT_OF_RANGE", EDGE_OUT_OF_RANGE),
    ("EDGE_UNKNOWN_LABELS", EDGE_UNKNOWN_LABELS),
    ("EDGE_BOOL_STRING_TRAP", EDGE_BOOL_STRING_TRAP),
    ("EDGE_NONE_ACOUSTIC", EDGE_NONE_ACOUSTIC),
    ("EDGE_SPARSE", EDGE_SPARSE),
]


def print_recommendations(
    profile_name: str, profile: dict, songs: list, k: int = 5, show_reasons: bool = True
) -> None:
    """Print top-k recommendations for one profile."""
    recommendations = recommend_songs(profile, songs, k=k)
    print("=" * 45)
    print(f"----- Top Recommendations: {profile_name} -----")
    print("=" * 45 + "\n")
    for idx, (song, score, reasons) in enumerate(recommendations, 1):
        print(f"{idx}. {song['title']:<30} | Score: {score:.2f}")
        if show_reasons:
            if isinstance(reasons, list):
                for reason in reasons:
                    print(f"   - {reason}")
            else:
                print(f"   - {reasons}")
        print("-" * 45 + "\n")


def run_profile_suite(songs: list, k: int = 5) -> None:
    """Print top-k recommendations for all defined profiles."""
    print("Top recommendations for all profiles\n")
    for profile_name, profile in PROFILE_SUITE:
        print_recommendations(profile_name, profile, songs, k=k, show_reasons=False)


def run_rag(songs: list) -> None:
    """RAG-enhanced recommendations with before/after metric comparison."""
    from .rag import KnowledgeBase
    from .gemini_client import GeminiClient
    import re

    print("\n" + "=" * 60)
    print("RAG-ENHANCED RECOMMENDATIONS")
    print("=" * 60)

    kb = KnowledgeBase()
    gemini = GeminiClient()
    top_songs = recommend_songs(TASTE_PROFILE, songs, k=5)

    song_list = "\n".join(
        f"- {s['title']} by {s['artist']} (score: {sc:.2f})"
        for s, sc, _ in top_songs
    )

    def _count_music_terms(text: str) -> int:
        terms = [
            "bpm", "tempo", "energy", "acousticness", "danceability", "valence",
            "timbre", "harmonic", "melody", "rhythm", "beat", "frequency",
            "acoustic", "electronic", "arousal", "circumplex", "entrainment",
            "lofi", "lo-fi", "vinyl", "reverb", "decay", "groove", "swing",
            "psychoacoustic", "cognitive", "parasympathetic",
        ]
        lower = text.lower()
        return sum(1 for t in terms if t in lower)

    def _count_numeric_refs(text: str) -> int:
        return len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:bpm|hz|db|%|ms)?\b", text, re.IGNORECASE))

    def _depth_score(text: str) -> float:
        terms = _count_music_terms(text)
        nums = _count_numeric_refs(text)
        sentences = max(1, len([s for s in re.split(r"[.!?]+", text.strip()) if s.strip()]))
        return round((terms + nums) / sentences * 3, 1)

    # Baseline call
    baseline_prompt = (
        "You are a music recommendation assistant. "
        "Explain why these songs fit the listener based on the song titles and scores alone.\n\n"
        f"Songs:\n{song_list}"
    )
    print("\nCalling Gemini (baseline — no RAG context)...")
    baseline_text = gemini.generate(baseline_prompt, max_tokens=256)
    print(baseline_text)

    # RAG-enhanced call
    context = kb.build_context(TASTE_PROFILE)
    rag_prompt = (
        "You are a music recommendation assistant with deep genre knowledge. "
        "Use the following music domain knowledge to explain why these songs fit the listener.\n\n"
        f"DOMAIN KNOWLEDGE:\n{context}\n\n"
        f"Songs:\n{song_list}\n\n"
        "Explain with references to BPM, energy, acousticness, and genre characteristics."
    )
    print("\nCalling Gemini (RAG-enhanced)...")
    rag_text = gemini.generate(rag_prompt, max_tokens=512)
    print(rag_text)

    # Metrics
    b_terms = _count_music_terms(baseline_text)
    r_terms = _count_music_terms(rag_text)
    b_nums = _count_numeric_refs(baseline_text)
    r_nums = _count_numeric_refs(rag_text)
    b_depth = _depth_score(baseline_text)
    r_depth = _depth_score(rag_text)

    print("\n" + "-" * 55)
    print(f"{'Metric':<25} | {'Baseline':>8} | {'RAG':>6} | {'Delta':>7}")
    print("-" * 55)
    print(f"{'Music domain terms':<25} | {b_terms:>8} | {r_terms:>6} | {r_terms - b_terms:>+7}")
    print(f"{'Numeric references':<25} | {b_nums:>8} | {r_nums:>6} | {r_nums - b_nums:>+7}")
    print(f"{'Technical depth score':<25} | {b_depth:>8.1f} | {r_depth:>6.1f} | {r_depth - b_depth:>+7.1f}")
    print("=" * 60)


def main() -> None:
    from .logger import get_logger
    log = get_logger(__name__)

    songs = load_songs("data/songs.csv")
    log.info("Loaded %d songs from data/songs.csv", len(songs))
    print(f"\nLoaded {len(songs)} songs.\n")

    if "--rag" in sys.argv:
        run_rag(songs)
    elif "--agent" in sys.argv:
        from .agent import run_agent
        run_agent(songs)
    elif "--fewshot" in sys.argv:
        from .fewshot import run_fewshot
        run_fewshot(songs)
    elif "--evaluate" in sys.argv:
        from scripts.evaluate import run_evaluation
        import sys as _sys
        failures = run_evaluation(songs)
        _sys.exit(0 if failures == 0 else 1)
    elif "--all-profiles" in sys.argv or "--suite" in sys.argv:
        run_profile_suite(songs, k=5)
    else:
        # Default: run the integrated pipeline (score + RAG + AI explanation)
        from .rag import KnowledgeBase
        from .gemini_client import GeminiClient
        from .pipeline import RecommendationPipeline

        kb = KnowledgeBase()
        gemini = GeminiClient()
        pipeline = RecommendationPipeline(songs, kb, gemini)

        result = pipeline.run(TASTE_PROFILE, k=5)

        if result.error:
            log.error("Pipeline error: %s", result.error)
            print(f"Error: {result.error}")
            return

        print("=" * 55)
        print("----- Top Recommendations: TASTE_PROFILE -----")
        print("=" * 55 + "\n")
        for idx, (song, score, reasons) in enumerate(result.top_songs, 1):
            print(f"{idx}. {song['title']:<30} | Score: {score:.2f}")
            for reason in reasons:
                print(f"   - {reason}")
            print("-" * 55)

        print(f"\n[RAG] Retrieved: {[d['filename'] for d in result.retrieved_docs]}")

        if result.explanation and not result.explanation.startswith("["):
            print("\n[AI Explanation — RAG-grounded]")
            print(result.explanation)
        else:
            print(f"\n[AI] {result.explanation}")

        log.info("Default pipeline run completed in %.0f ms", result.elapsed_ms)


if __name__ == "__main__":
    main()
