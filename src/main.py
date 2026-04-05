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


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"\nLoaded {len(songs)} songs.\n")
    if "--all-profiles" in sys.argv or "--suite" in sys.argv:
        run_profile_suite(songs, k=5)
    else:
        print_recommendations("TASTE_PROFILE", TASTE_PROFILE, songs, k=5, show_reasons=True)


if __name__ == "__main__":
    main()
