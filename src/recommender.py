from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Point-weighting strategy -------------------------------------------------
# Discrete matches: genre now contributes less relative to continuous similarity.
WEIGHT_GENRE_MATCH = 1.0
WEIGHT_MOOD_MATCH = 1.0

# Continuous: up to this many points when song energy exactly equals target (0–1 scale).
WEIGHT_ENERGY_SIMILARITY_MAX = 2.0

# Optional refinements when `user_prefs` includes extra keys (e.g. from TASTE_PROFILE).
WEIGHT_TEMPO_SIMILARITY_MAX = 0.5
WEIGHT_VALENCE_SIMILARITY_MAX = 0.5
WEIGHT_DANCE_SIMILARITY_MAX = 0.5
# Scale BPM error: zero tempo points when |ΔBPM| >= this value.
TEMPO_MATCH_SCALE_BPM = 80.0
WEIGHT_ACOUSTIC_ALIGN_MAX = 0.5


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # Optional refinements; when set, passed through to the same prefs dict as TASTE_PROFILE.
    target_tempo_bpm: Optional[float] = None
    target_valence: Optional[float] = None
    target_danceability: Optional[float] = None


def _song_to_dict(song: Song) -> Dict:
    """Map a Song dataclass to the dict shape expected by score_song."""
    return {
        "id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "tempo_bpm": song.tempo_bpm,
        "valence": song.valence,
        "danceability": song.danceability,
        "acousticness": song.acousticness,
    }


def score_song(song: Dict, user_prefs: Dict) -> float:
    """Total score for one song against a taste profile (higher is better)."""
    total, _ = score_song_with_explanation(song, user_prefs)
    return total


def score_song_with_explanation(song: Dict, user_prefs: Dict) -> Tuple[float, list]:
    """Score one song against prefs and return (total, list of short reason strings)."""
    score = 0.0
    reasons: List[str] = []

    fg = user_prefs.get("favorite_genre")
    if fg is not None and song.get("genre") == fg:
        score += WEIGHT_GENRE_MATCH
        reasons.append(f"genre match (+{WEIGHT_GENRE_MATCH:.1f})")

    fm = user_prefs.get("favorite_mood")
    if fm is not None and song.get("mood") == fm:
        score += WEIGHT_MOOD_MATCH
        reasons.append(f"mood match (+{WEIGHT_MOOD_MATCH:.1f})")

    te = user_prefs.get("target_energy")
    if te is not None:
        e = float(song.get("energy", 0))
        energy_pts = WEIGHT_ENERGY_SIMILARITY_MAX * max(
            0.0, 1.0 - abs(float(te) - e)
        )
        if energy_pts > 0:
            reasons.append(f"energy closeness (+{energy_pts:.2f})")
        score += energy_pts

    tt = user_prefs.get("target_tempo_bpm")
    if tt is not None:
        t = float(song.get("tempo_bpm", 0))
        tempo_delta = abs(t - float(tt))
        tempo_pts = WEIGHT_TEMPO_SIMILARITY_MAX * max(
            0.0, 1.0 - tempo_delta / TEMPO_MATCH_SCALE_BPM
        )
        if tempo_pts > 0:
            reasons.append(f"tempo closeness (+{tempo_pts:.2f})")
        score += tempo_pts

    tv = user_prefs.get("target_valence")
    if tv is not None:
        v = float(song.get("valence", 0))
        valence_pts = WEIGHT_VALENCE_SIMILARITY_MAX * max(
            0.0, 1.0 - abs(float(tv) - v)
        )
        if valence_pts > 0:
            reasons.append(f"valence closeness (+{valence_pts:.2f})")
        score += valence_pts

    td = user_prefs.get("target_danceability")
    if td is not None:
        d = float(song.get("danceability", 0))
        dance_pts = WEIGHT_DANCE_SIMILARITY_MAX * max(
            0.0, 1.0 - abs(float(td) - d)
        )
        if dance_pts > 0:
            reasons.append(f"danceability closeness (+{dance_pts:.2f})")
        score += dance_pts

    if "likes_acoustic" in user_prefs:
        ac = float(song.get("acousticness", 0))
        if user_prefs["likes_acoustic"]:
            acoustic_pts = WEIGHT_ACOUSTIC_ALIGN_MAX * ac
            if acoustic_pts > 0:
                reasons.append(f"favors acoustic sound (+{acoustic_pts:.2f})")
        else:
            acoustic_pts = WEIGHT_ACOUSTIC_ALIGN_MAX * (1.0 - ac)
            if acoustic_pts > 0:
                reasons.append(f"favors non-acoustic sound (+{acoustic_pts:.2f})")
        score += acoustic_pts

    if not reasons:
        reasons.append("weak or no alignment with this profile")

    return score, reasons


def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv-style catalog into a list of dicts with numeric fields coerced."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Song catalog not found: {csv_path}")

    songs: List[Dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": int(float(row["tempo_bpm"])),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )
    return songs


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, List[str]]]:
    """Rank all songs by score (tie-break: lower id) and return the top k with reasons."""
    # Judge every song with the scoring function
    scored = [
        (song, score, explanation)
        for song, (score, explanation) in (
            (song, score_song_with_explanation(song, user_prefs)) for song in songs
        )
    ]
    # Sort entire list by score (descending), breaking ties by lower id
    scored.sort(key=lambda x: (-x[1], x[0]["id"]))
    # Return the top k
    return scored[:k]


def _profile_to_prefs(user: UserProfile) -> Dict:
    """Flatten UserProfile into the dict keys score_song_with_explanation expects."""
    prefs: Dict = {
        "favorite_genre": user.favorite_genre,
        "favorite_mood": user.favorite_mood,
        "target_energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
    }
    if user.target_tempo_bpm is not None:
        prefs["target_tempo_bpm"] = user.target_tempo_bpm
    if user.target_valence is not None:
        prefs["target_valence"] = user.target_valence
    if user.target_danceability is not None:
        prefs["target_danceability"] = user.target_danceability
    return prefs


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        """Hold the song catalog used for recommendations."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return up to k highest-scoring Song objects for this user."""
        prefs = _profile_to_prefs(user)
        ranked: List[Tuple[Song, float]] = []
        for song in self.songs:
            s = score_song(_song_to_dict(song), prefs)
            ranked.append((song, s))
        ranked.sort(key=lambda x: (-x[1], x[0].id))
        return [s for s, _ in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return scoring reasons for one catalog song under this user as one line of text."""
        prefs = _profile_to_prefs(user)
        _, reasons = score_song_with_explanation(_song_to_dict(song), prefs)
        return "; ".join(reasons)
