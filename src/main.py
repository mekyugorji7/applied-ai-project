"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

try:
    from .recommender import load_songs, recommend_songs
except ImportError:
    # `python src/main.py` — script is not executed as part of package `src`
    from recommender import load_songs, recommend_songs

# Content-based taste profile: target values compared against each song's features.
# Maps to catalog fields: genre, mood, energy, tempo_bpm, valence, danceability, acousticness.
TASTE_PROFILE = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.42,
    "target_tempo_bpm": 78.0,
    "target_valence": 0.58,
    "target_danceability": 0.61,
    "likes_acoustic": True,
}


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"\nLoaded {len(songs)} songs.\n")
    recommendations = recommend_songs(TASTE_PROFILE, songs, k=5)

    print("=" * 45)
    print("----- Top Recommendations For You -----")
    print("=" * 45 + "\n")
    for idx, (song, score, reasons) in enumerate(recommendations, 1):
        print(f"{idx}. {song['title']:<30} | Score: {score:.2f}")
        if isinstance(reasons, list):
            for reason in reasons:
                print(f"   - {reason}")
        else:
            print(f"   - {reasons}")
        print("-" * 45 + "\n")


if __name__ == "__main__":
    main()
