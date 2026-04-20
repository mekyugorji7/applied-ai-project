"""Profile-centric Streamlit frontend for the AI Music Recommender."""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from src.recommender import load_songs
from src.gemini_client import GeminiClient, _resolve_api_key
from src.rag import KnowledgeBase
from src.pipeline import RecommendationPipeline, validate_prefs
from src.logger import get_logger
from src.spotify_client import SpotifyClient, _resolve_spotify_credentials

log = get_logger("app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="My Vibe",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighten top padding */
.block-container { padding-top: 2rem; max-width: 860px; }

/* Preset buttons row */
div[data-testid="column"] > div > div > div > button {
    border-radius: 20px;
    font-size: 0.85rem;
}

/* Song card header */
.song-rank {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
}

/* Feature label */
.feat-label {
    font-size: 0.72rem;
    color: #aaa;
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_songs():
    return load_songs("data/songs.csv")

@st.cache_resource
def get_kb():
    return KnowledgeBase("data/knowledge_base")

@st.cache_resource
def get_gemini():
    return GeminiClient()

@st.cache_resource
def get_pipeline():
    return RecommendationPipeline(get_songs(), get_kb(), get_gemini())

@st.cache_resource
def get_chat_agent():
    from src.chat_agent import ChatMusicAgent
    return ChatMusicAgent(get_songs(), get_kb(), _resolve_api_key())

@st.cache_resource
def get_spotify_client():
    cid, csec = _resolve_spotify_credentials()
    return SpotifyClient(cid, csec)

songs   = get_songs()
gemini  = get_gemini()
pipeline = get_pipeline()

# ── Session state init ────────────────────────────────────────────────────────
if "profile_submitted"   not in st.session_state: st.session_state.profile_submitted   = False
if "user_prefs"          not in st.session_state: st.session_state.user_prefs          = {}
if "chat_messages"       not in st.session_state: st.session_state.chat_messages       = []
if "active_preset"       not in st.session_state: st.session_state.active_preset       = "Chill Lofi"
if "spotify_connected"   not in st.session_state: st.session_state.spotify_connected   = False
if "spotify_profile"     not in st.session_state: st.session_state.spotify_profile     = {}
if "spotify_tracks"      not in st.session_state: st.session_state.spotify_tracks      = []
if "spotify_display"     not in st.session_state: st.session_state.spotify_display     = ""
if "spotify_genres"      not in st.session_state: st.session_state.spotify_genres      = []

# ── Spotify OAuth callback (runs before any page render) ─────────────────────
_params = st.query_params
if "code" in _params and not st.session_state.spotify_connected:
    _spotify = get_spotify_client()
    if _spotify.enabled and _spotify.handle_callback(_params["code"]):
        with st.spinner("Analyzing your Spotify listening history…"):
            try:
                _data = _spotify.fetch_taste_profile()
                st.session_state.spotify_connected = True
                st.session_state.spotify_profile   = _data["user_prefs"]
                st.session_state.spotify_tracks    = _data["top_tracks"]
                st.session_state.spotify_display   = _data["display_name"]
                st.session_state.spotify_genres    = _data["top_genres"]
                # Auto-submit: go straight to recommendations
                _sp_prefs = dict(_data["user_prefs"])
                _sp_prefs["_k"] = 5
                st.session_state.user_prefs       = validate_prefs(_sp_prefs)
                st.session_state.profile_submitted = True
                st.session_state.chat_messages     = []
            except RuntimeError as _exc:
                st.error(f"Spotify import failed: {_exc}")
        st.query_params.clear()
        st.rerun()

PRESETS = {
    "Chill Lofi":   dict(genre="lofi",    mood="chill",       energy=0.42, bpm=78.0,  valence=0.58, dance=0.61, acoustic=True),
    "Upbeat Pop":   dict(genre="pop",     mood="happy",       energy=0.85, bpm=120.0, valence=0.82, dance=0.80, acoustic=False),
    "Soft Ambient": dict(genre="ambient", mood="chill",       energy=0.30, bpm=62.0,  valence=0.62, dance=0.42, acoustic=True),
    "Dark Focus":   dict(genre="",        mood="melancholic", energy=0.45, bpm=85.0,  valence=0.30, dance=0.50, acoustic=False),
    "Custom":       dict(genre="",        mood="",            energy=0.50, bpm=90.0,  valence=0.60, dance=0.60, acoustic=True),
}

all_genres = sorted({s["genre"] for s in songs})
all_moods  = sorted({s["mood"]  for s in songs})

# ═════════════════════════════════════════════════════════════════════════════
# PROFILE BUILDER PAGE
# ═════════════════════════════════════════════════════════════════════════════
def show_profile_builder():
    st.title("🎵 My Vibe")
    st.markdown("#### Discover songs that match your vibe.")
    st.markdown("---")

    # ── Preset quick-select ──────────────────────────────────────────────────
    st.markdown("**Start from a vibe**")
    preset_cols = st.columns(len(PRESETS))
    for col, name in zip(preset_cols, PRESETS):
        if col.button(name, use_container_width=True,
                      type="primary" if st.session_state.active_preset == name else "secondary"):
            st.session_state.active_preset = name
            st.rerun()

    p = PRESETS[st.session_state.active_preset]
    st.markdown("")

    # ── Upload profile JSON (optional) ───────────────────────────────────────
    with st.expander("Or upload a profile JSON"):
        uploaded = st.file_uploader("Profile JSON", type="json", label_visibility="collapsed")
        if uploaded:
            try:
                loaded = json.load(uploaded)
                st.session_state.user_prefs = validate_prefs(loaded)
                st.session_state.profile_submitted = True
                st.rerun()
            except Exception as exc:
                st.error(f"Could not parse profile: {exc}")

    # ── Spotify import + manual sliders ─────────────────────────────────────
    spotify = get_spotify_client()
    with st.expander("Spotify import (optional)"):
        if spotify.enabled:
            if st.session_state.spotify_connected:
                st.success(f"Connected as **{st.session_state.spotify_display}**")
                st.caption(
                    f"Analyzed {len(st.session_state.spotify_tracks)} top tracks\n\n"
                    f"Top genres: {', '.join(g for g, _ in st.session_state.spotify_genres[:3])}"
                )
                with st.expander("View analyzed tracks"):
                    st.dataframe(
                        pd.DataFrame(st.session_state.spotify_tracks),
                        use_container_width=True, hide_index=True,
                    )
                # Override p with Spotify-derived values so sliders start there
                sp = st.session_state.spotify_profile
                p = dict(
                    genre=sp.get("favorite_genre", ""),
                    mood=sp.get("favorite_mood", ""),
                    energy=sp.get("target_energy", 0.5),
                    bpm=sp.get("target_tempo_bpm", 90.0),
                    valence=sp.get("target_valence", 0.6),
                    dance=sp.get("target_danceability", 0.6),
                    acoustic=sp.get("likes_acoustic", True),
                )
                if st.button("Disconnect", key="spotify_disconnect"):
                    st.session_state.spotify_connected = False
                    st.session_state.spotify_profile   = {}
                    st.session_state.spotify_tracks    = []
                    st.session_state.spotify_genres    = []
                    st.rerun()
            else:
                auth_url = spotify.get_auth_url()
                st.link_button("Connect Spotify", auth_url, use_container_width=True)
                st.caption("Reads your top 20 tracks.\nNo data is stored.")
        else:
            st.info("Add `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` to `.env` to enable.")

    with st.form("profile_form"):
        st.markdown("**Refine your profile**")
        st.caption("Fine tune values to your exact taste.")

        genre_opts = ["(any)"] + all_genres
        mood_opts  = ["(any)"] + all_moods

        col_a, col_b = st.columns(2)
        fav_genre = col_a.selectbox(
            "Favorite genre",
            genre_opts,
            index=genre_opts.index(p["genre"]) if p["genre"] in genre_opts else 0,
        )
        fav_mood = col_b.selectbox(
            "Favorite mood",
            mood_opts,
            index=mood_opts.index(p["mood"]) if p["mood"] in mood_opts else 0,
        )

        target_energy  = st.slider("Energy",                   0.0, 1.0,   p["energy"],  0.01,
                                   help="How intense and active the music should feel")
        target_bpm     = st.slider("BPM (tempo)",              50.0, 200.0, p["bpm"],    1.0)

        col_c, col_d = st.columns(2)
        target_valence = col_c.slider("Valence — sad → happy", 0.0, 1.0,   p["valence"], 0.01)
        target_dance   = col_d.slider("Danceability",          0.0, 1.0,   p["dance"],   0.01)

        likes_acoustic = st.checkbox("Prefers acoustic sound", value=p["acoustic"])

        k_songs = st.slider("Number of recommendations", 3, 10, 5)

        submitted = st.form_submit_button("🎵 Find My Music", use_container_width=True, type="primary")

    if submitted:
        raw_prefs = {
            "target_energy":       target_energy,
            "target_tempo_bpm":    target_bpm,
            "target_valence":      target_valence,
            "target_danceability": target_dance,
            "likes_acoustic":      likes_acoustic,
            "_k":                  k_songs,
        }
        if fav_genre != "(any)":
            raw_prefs["favorite_genre"] = fav_genre
        if fav_mood != "(any)":
            raw_prefs["favorite_mood"] = fav_mood

        st.session_state.user_prefs = validate_prefs(raw_prefs)
        st.session_state.profile_submitted = True
        st.session_state.chat_messages = []
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS PAGE
# ═════════════════════════════════════════════════════════════════════════════
def _feat_bar(label: str, value: float, target: float | None = None):
    """Render a labeled progress bar with an optional target marker."""
    delta_str = ""
    if target is not None:
        delta = value - target
        delta_str = f"  (you: {target:.2f})"
    st.markdown(f"<p class='feat-label'>{label}{delta_str}</p>", unsafe_allow_html=True)
    st.progress(float(value))


def show_results():
    from src.chat_agent import find_similar_songs_util

    prefs   = dict(st.session_state.user_prefs)          # copy so we don't mutate state
    k_songs = int(prefs.pop("_k", 5))

    # ── Header row ───────────────────────────────────────────────────────────
    hcol_l, hcol_r = st.columns([5, 1])
    hcol_l.title("🎵 My Vibe")
    if hcol_r.button("Edit profile", use_container_width=True):
        st.session_state.profile_submitted = False
        st.session_state.user_prefs = {}
        st.rerun()

    # ── Spotify import banner ─────────────────────────────────────────────
    if st.session_state.spotify_connected:
        sp_genres_str = ", ".join(g for g, _ in st.session_state.spotify_genres[:3])
        st.info(
            f"Profile imported from Spotify — **{st.session_state.spotify_display}** · "
            f"{len(st.session_state.spotify_tracks)} tracks analyzed · "
            f"Top genres: {sp_genres_str}",
            icon="🎧",
        )

    # ── Profile summary strip ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Your profile**")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Genre",        prefs.get("favorite_genre", "any").title())
    m2.metric("Mood",         prefs.get("favorite_mood",  "any").title())
    m3.metric("Energy",       f"{prefs.get('target_energy', 0.5):.2f}")
    m4.metric("BPM",          f"{prefs.get('target_tempo_bpm', 90):.0f}")
    m5.metric("Sound",        "Acoustic" if prefs.get("likes_acoustic") else "Electronic")
    st.markdown("---")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    with st.spinner("Finding your music…"):
        result = pipeline.run(prefs, k=k_songs)

    if result.error:
        st.error(f"Something went wrong: {result.error}")
        return

    top_songs = result.top_songs
    top_ids   = [s["id"] for s, _, _ in top_songs]

    # ── Recommendations ───────────────────────────────────────────────────────
    st.subheader(f"Your top {len(top_songs)} picks")

    for i, (song, score, reasons) in enumerate(top_songs, 1):
        with st.expander(
            f"#{i}  {song['title']} — {song['artist']}",
            expanded=(i <= 2),
        ):
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Genre",  song["genre"].title())
            rc2.metric("Mood",   song["mood"].title())
            rc3.metric("Score",  f"{score:.2f} / 6.00")
            rc4.metric("BPM",    song["tempo_bpm"])

            st.markdown("")
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                _feat_bar("Energy",       song["energy"],       prefs.get("target_energy"))
            with fc2:
                _feat_bar("Valence",      song["valence"],      prefs.get("target_valence"))
            with fc3:
                _feat_bar("Danceability", song["danceability"], prefs.get("target_danceability"))
            with fc4:
                _feat_bar("Acousticness", song["acousticness"])

            st.caption("Scoring: " + " · ".join(reasons))

    # ── AI explanation ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Why these songs?")

    if result.explanation and not result.explanation.startswith("["):
        st.success(result.explanation)
        with st.expander("Knowledge base sources"):
            for doc in result.retrieved_docs:
                st.markdown(f"**{doc['filename']}** — relevance `{doc['score']:.2f}`")
                st.text(doc["content"][:280] + "…")
        st.caption(f"Pipeline: {result.elapsed_ms:.0f} ms  ·  Model: {gemini.model_name}")
    elif not gemini.enabled:
        st.info("Add `GEMINI_API_KEY` to `.env` to see AI explanations.")
    else:
        st.warning(result.explanation)

    # ── Discovery: similar to #1 ──────────────────────────────────────────────
    if top_songs:
        top_song = top_songs[0][0]
        st.markdown("---")
        st.markdown(f"#### You might also like — songs with similar audio DNA to _{top_song['title']}_")
        st.caption("Not in your top picks, but share similar energy, valence, and danceability")

        similar = find_similar_songs_util(top_song, songs, k=3, exclude_ids=top_ids)
        if similar:
            dc1, dc2, dc3 = st.columns(3)
            for col, item in zip([dc1, dc2, dc3], similar):
                s = item["song"]
                dist = item["distance"]
                col.markdown(f"**{s['title']}**")
                col.caption(f"{s['artist']}")
                col.markdown(f"`{s['genre']}` · `{s['mood']}`")
                col.progress(max(0.0, 1.0 - dist), text=f"Match: {max(0,(1-dist)*100):.0f}%")

    # ── Chat ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Chat with your music agent")
    st.caption("Ask for recommendations, explore moods, or discover new sounds.")

    chat_agent = get_chat_agent()

    if not chat_agent.enabled:
        st.info("Add `GEMINI_API_KEY` to `.env` to enable the chat agent.")
    else:
        # Render history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Clear button
        if st.session_state.chat_messages:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.chat_messages = []
                st.rerun()

        # Input
        prompt = st.chat_input("Ask for a recommendation…")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            history = st.session_state.chat_messages[:-1]
            with st.spinner("Agent thinking…"):
                response, steps = chat_agent.chat(prompt, history)

            if steps:
                with st.expander("🔧 Agent tool calls"):
                    for name, args in steps:
                        st.markdown(f"**`{name}`**")
                        st.json(args)

            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

    # ── Full catalog ──────────────────────────────────────────────────────────
    with st.expander("Browse full catalog"):
        st.dataframe(
            pd.DataFrame(songs).drop(columns=["id"]),
            use_container_width=True, hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state.profile_submitted:
    show_profile_builder()
else:
    show_results()
