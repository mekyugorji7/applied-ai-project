"""Spotify OAuth client — fetches listening history and derives a taste profile.

Credential resolution order:
  1. SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET environment variables
  2. .env file in the project root (loaded via python-dotenv)
  3. Streamlit st.secrets["SPOTIFY_CLIENT_ID"] / ["SPOTIFY_CLIENT_SECRET"]

If spotipy is not installed or credentials are missing, SpotifyClient.enabled = False
and the rest of the app degrades gracefully.
"""

from __future__ import annotations

import os
from pathlib import Path

from .logger import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).parent.parent
_ENV_FILE = _ROOT / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_FILE, override=False)
except ImportError:
    pass

# ── Genre mapping ─────────────────────────────────────────────────────────────
# Maps Spotify's granular genres → catalog genre labels
GENRE_MAP = {
    "lo-fi": "lofi", "lofi": "lofi", "chillhop": "lofi", "study music": "lofi",
    "hip hop": "hip-hop", "rap": "hip-hop", "trap": "hip-hop",
    "pop": "pop", "dance pop": "pop", "electropop": "pop", "synth-pop": "pop",
    "ambient": "ambient", "new age": "ambient", "drone": "ambient",
    "jazz": "jazz", "bebop": "jazz", "smooth jazz": "jazz", "swing": "jazz",
    "rock": "rock", "indie rock": "rock", "alternative rock": "rock",
    "folk": "folk", "indie folk": "folk", "singer-songwriter": "folk",
    "country": "country", "americana": "country",
    "r&b": "r&b", "soul": "r&b", "neo soul": "r&b",
    "electronic": "electronic", "edm": "electronic",
    "techno": "techno", "house": "techno",
    "latin": "latin", "reggaeton": "latin",
    "classical": "classical", "orchestra": "classical",
    "blues": "blues",
}

SCOPES = "user-top-read"
REDIRECT_URI = "http://localhost:8501"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _derive_mood(valence: float, energy: float) -> str:
    """Map valence + energy to a catalog mood label."""
    if valence >= 0.7 and energy >= 0.65:
        return "happy"
    if valence >= 0.6 and energy < 0.65:
        return "romantic"
    if valence < 0.35 and energy >= 0.65:
        return "intense"
    if valence < 0.35:
        return "melancholic"
    if energy < 0.40:
        return "chill"
    if energy < 0.55:
        return "relaxed"
    return "focused"


def _resolve_spotify_credentials() -> tuple[str, str]:
    """Return (client_id, client_secret) from env / .env / st.secrets."""
    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()

    if client_id and client_secret:
        return client_id, client_secret

    try:
        import streamlit as st
        cid = st.secrets.get("SPOTIFY_CLIENT_ID", "").strip()
        csec = st.secrets.get("SPOTIFY_CLIENT_SECRET", "").strip()
        if cid and csec:
            return cid, csec
    except Exception:
        pass

    return "", ""


# ── SpotifyClient ─────────────────────────────────────────────────────────────

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    from spotipy.cache_handler import MemoryCacheHandler
    _SPOTIPY_AVAILABLE = True
except ImportError:
    _SPOTIPY_AVAILABLE = False
    log.warning("spotipy not installed — Spotify integration disabled.")


class SpotifyClient:
    """OAuth-based Spotify client that derives a user_prefs-compatible taste profile."""

    def __init__(self, client_id: str, client_secret: str):
        self.enabled = bool(client_id and client_secret) and _SPOTIPY_AVAILABLE
        self._client_id = client_id
        self._client_secret = client_secret
        self._sp = None          # authenticated spotipy.Spotify instance
        self._auth_manager = None

    def _make_auth_manager(self) -> "SpotifyOAuth":
        return SpotifyOAuth(
            client_id=self._client_id,
            client_secret=self._client_secret,
            redirect_uri=REDIRECT_URI,
            scope=SCOPES,
            cache_handler=MemoryCacheHandler(),
            show_dialog=False,
        )

    def get_auth_url(self) -> str:
        """Return the Spotify OAuth URL the user should visit."""
        if not self.enabled:
            return ""
        self._auth_manager = self._make_auth_manager()
        return self._auth_manager.get_authorize_url()

    def handle_callback(self, code: str) -> bool:
        """Exchange auth code for token. Stores authenticated client in self._sp.
        Returns True on success, False on failure."""
        if not self.enabled:
            return False
        try:
            if self._auth_manager is None:
                self._auth_manager = self._make_auth_manager()
            self._auth_manager.get_access_token(code, as_dict=False)
            self._sp = spotipy.Spotify(auth_manager=self._auth_manager)
            log.info("Spotify authentication successful.")
            return True
        except Exception as exc:
            log.error("Spotify callback failed: %s", exc)
            return False

    def fetch_taste_profile(self, time_range: str = "medium_term") -> dict:
        """
        Fetch top tracks + audio features + top artists.

        Returns a dict with:
          - 'user_prefs'    : ready-to-use prefs dict
          - 'top_tracks'    : list of {title, artist, energy, valence, danceability, tempo}
          - 'top_genres'    : list of (genre, count) tuples sorted by count desc
          - 'track_count'   : int
          - 'display_name'  : Spotify username
        Raises RuntimeError if not authenticated.
        """
        if self._sp is None:
            raise RuntimeError("Spotify client is not authenticated.")

        try:
            # 1. Username
            user_info = self._sp.current_user()
            display_name = user_info.get("display_name") or user_info.get("id", "")

            # 2. Top tracks
            tracks_resp = self._sp.current_user_top_tracks(
                limit=20, time_range=time_range
            )
            tracks = tracks_resp.get("items", [])
            if not tracks:
                raise RuntimeError("No top tracks found in your Spotify history.")

            track_ids = [t["id"] for t in tracks]

            # 3. Audio features
            features_resp = self._sp.audio_features(track_ids)
            # Filter out None entries (tracks without audio features)
            valid_pairs = [
                (t, f)
                for t, f in zip(tracks, features_resp)
                if f is not None
            ]
            if not valid_pairs:
                raise RuntimeError("Could not retrieve audio features from Spotify.")

            valid_tracks, valid_features = zip(*valid_pairs)

            # 4. Top artists → genres
            artists_resp = self._sp.current_user_top_artists(
                limit=10, time_range=time_range
            )
            genre_counts: dict[str, int] = {}
            for artist in artists_resp.get("items", []):
                for g in artist.get("genres", []):
                    genre_counts[g] = genre_counts.get(g, 0) + 1

            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
            top_genre_names = [g for g, _ in top_genres]

            # 5. Build user_prefs
            user_prefs = self._derive_prefs(list(valid_features), top_genre_names)

            # 6. Build top_tracks display list
            top_tracks_display = [
                {
                    "title":        t["name"],
                    "artist":       ", ".join(a["name"] for a in t["artists"]),
                    "energy":       round(f["energy"], 2),
                    "valence":      round(f["valence"], 2),
                    "danceability": round(f["danceability"], 2),
                    "tempo":        round(f["tempo"], 0),
                }
                for t, f in zip(valid_tracks, valid_features)
            ]

            return {
                "user_prefs":   user_prefs,
                "top_tracks":   top_tracks_display,
                "top_genres":   top_genres,
                "track_count":  len(valid_tracks),
                "display_name": display_name,
            }

        except RuntimeError:
            raise
        except Exception as exc:
            log.error("fetch_taste_profile failed: %s", exc)
            raise RuntimeError(f"Spotify API error: {exc}") from exc

    def _derive_prefs(self, features: list[dict], top_genres: list[str]) -> dict:
        """Average audio features + map genre → user_prefs compatible dict."""
        import numpy as np

        energy   = float(np.mean([f["energy"]       for f in features]))
        valence  = float(np.mean([f["valence"]      for f in features]))
        dance    = float(np.mean([f["danceability"] for f in features]))
        acoustic = float(np.mean([f["acousticness"] for f in features]))
        tempo    = float(np.mean([f["tempo"]        for f in features]))

        # Genre: find first Spotify genre that maps to a catalog genre
        genre = ""
        for g in top_genres:
            g_lower = g.lower()
            for key, val in GENRE_MAP.items():
                if key in g_lower:
                    genre = val
                    break
            if genre:
                break

        return {
            "target_energy":       round(energy, 3),
            "target_valence":      round(valence, 3),
            "target_danceability": round(dance, 3),
            "target_tempo_bpm":    round(tempo, 1),
            "likes_acoustic":      acoustic > 0.5,
            "favorite_genre":      genre,
            "favorite_mood":       _derive_mood(valence, energy),
        }
