"""LangChain-powered chat agent for music recommendation.

Provides 5 tools (search_songs, get_song_details, find_similar_songs,
score_and_recommend, retrieve_knowledge) and a ChatMusicAgent class that
wraps a LangChain AgentExecutor with tool-calling capability.

Degrades gracefully when LangChain or the API key is unavailable.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict

from .rag import KnowledgeBase
from .recommender import recommend_songs
from .logger import get_logger

log = get_logger(__name__)


def _extract_text(content) -> str:
    """Normalize LangChain message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(filter(None, parts))
    return str(content)


# ---------------------------------------------------------------------------
# Feature similarity utility (also exported for app.py discovery cards)
# ---------------------------------------------------------------------------

_SIMILARITY_FEATURES = ["energy", "valence", "danceability", "acousticness"]


def find_similar_songs_util(
    ref_song: Dict,
    songs: List[Dict],
    k: int = 3,
    exclude_ids: List[int] | None = None,
) -> List[Dict]:
    """Return k songs most similar to ref_song by cosine distance on audio features.

    Args:
        ref_song: The reference song dict (must have the 4 feature keys).
        songs: Full catalog list of song dicts.
        k: Number of similar songs to return.
        exclude_ids: Song IDs to skip (e.g. already-shown top picks).

    Returns:
        List of dicts each containing 'song' and 'distance' keys.
    """
    ref_id = ref_song.get("id")
    skip_ids = set(exclude_ids or [])
    if ref_id is not None:
        skip_ids.add(ref_id)

    ref_vec = np.array([float(ref_song.get(f, 0)) for f in _SIMILARITY_FEATURES])

    candidates = []
    for s in songs:
        if s.get("id") in skip_ids:
            continue
        vec = np.array([float(s.get(f, 0)) for f in _SIMILARITY_FEATURES])
        dist = float(np.linalg.norm(ref_vec - vec))
        candidates.append({"song": s, "distance": dist})

    candidates.sort(key=lambda x: x["distance"])
    return candidates[:k]


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def build_tools(songs: List[Dict], kb: KnowledgeBase):
    """Build and return the 5 LangChain tools, closing over songs and kb."""
    from langchain_core.tools import tool

    @tool
    def search_songs(query: str) -> str:
        """Search the music catalog by genre or mood keywords.

        Args:
            query: Keywords to match against genre and mood fields.
        """
        q = query.lower()
        matches = [
            s for s in songs
            if q in s.get("genre", "").lower() or q in s.get("mood", "").lower()
        ]
        if not matches:
            return f"No songs found matching '{query}'."
        lines = [
            f"- {s['title']} by {s['artist']} (genre={s['genre']}, mood={s['mood']}, "
            f"energy={s['energy']:.2f}, BPM={s['tempo_bpm']})"
            for s in matches[:10]
        ]
        return f"Found {len(matches)} song(s) matching '{query}':\n" + "\n".join(lines)

    @tool
    def get_song_details(title: str) -> str:
        """Get full audio feature profile for a song by title.

        Args:
            title: The song title to look up (case-insensitive).
        """
        t = title.lower()
        matches = [s for s in songs if t in s.get("title", "").lower()]
        if not matches:
            return f"No song found with title containing '{title}'."
        s = matches[0]
        return (
            f"Title: {s['title']}\n"
            f"Artist: {s['artist']}\n"
            f"Genre: {s['genre']} | Mood: {s['mood']}\n"
            f"Energy: {s['energy']:.2f} | BPM: {s['tempo_bpm']}\n"
            f"Valence: {s['valence']:.2f} | Danceability: {s['danceability']:.2f}\n"
            f"Acousticness: {s['acousticness']:.2f}"
        )

    @tool
    def find_similar_songs(title: str, k: int = 3) -> str:
        """Find songs with the most similar audio DNA to a given song.

        Uses cosine distance on energy, valence, danceability, and acousticness.

        Args:
            title: Reference song title (case-insensitive).
            k: Number of similar songs to return (default 3).
        """
        t = title.lower()
        matches = [s for s in songs if t in s.get("title", "").lower()]
        if not matches:
            return f"No song found with title containing '{title}'."
        ref = matches[0]
        similar = find_similar_songs_util(ref, songs, k=k)
        if not similar:
            return "No similar songs found."
        lines = [
            f"- {item['song']['title']} by {item['song']['artist']} "
            f"(genre={item['song']['genre']}, mood={item['song']['mood']}, "
            f"distance={item['distance']:.3f})"
            for item in similar
        ]
        return f"Songs most similar to '{ref['title']}':\n" + "\n".join(lines)

    @tool
    def score_and_recommend(genre: str, mood: str, energy_level: str) -> str:
        """Score the catalog and return top-5 recommendations for a profile.

        Args:
            genre: Preferred genre (e.g. 'lofi', 'pop', 'ambient').
            mood: Preferred mood (e.g. 'chill', 'happy', 'melancholic').
            energy_level: One of 'low', 'medium', or 'high'.
        """
        energy_map = {"low": 0.35, "medium": 0.60, "high": 0.85}
        energy = energy_map.get(energy_level.lower(), 0.55)
        prefs = {
            "favorite_genre": genre.lower().strip(),
            "favorite_mood": mood.lower().strip(),
            "target_energy": energy,
        }
        top = recommend_songs(prefs, songs, k=5)
        if not top:
            return "No recommendations found for this profile."
        lines = [
            f"{i}. {s['title']} by {s['artist']} "
            f"(genre={s['genre']}, mood={s['mood']}, score={sc:.2f}, "
            f"energy={s['energy']:.2f}, BPM={s['tempo_bpm']})"
            for i, (s, sc, _) in enumerate(top, 1)
        ]
        return "Top 5 recommendations:\n" + "\n".join(lines)

    @tool
    def retrieve_knowledge(query: str) -> str:
        """Retrieve relevant genre/mood knowledge from the music knowledge base.

        Args:
            query: Free-text query describing the genre, mood, or audio characteristics.
        """
        docs = kb.retrieve(query, top_k=2)
        if not docs:
            return "No relevant knowledge base documents found."
        parts = [f"[{d['filename']} relevance={d['score']:.2f}]\n{d['content']}" for d in docs]
        return "\n\n".join(parts)

    return [search_songs, get_song_details, find_similar_songs, score_and_recommend, retrieve_knowledge]


# ---------------------------------------------------------------------------
# ChatMusicAgent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a music recommendation assistant with access to a 20-song catalog and a music "
    "knowledge base. When the user asks for recommendations, always call at least one tool "
    "before answering. Use retrieve_knowledge to get genre/mood context, then score_and_recommend "
    "or find_similar_songs to ground your answer in real catalog data. Never invent song titles. "
    "Explain picks using concrete audio features (BPM, energy, acousticness)."
)


class ChatMusicAgent:
    """LangChain AgentExecutor-based conversational music recommender.

    Falls back gracefully if LangChain or the Gemini API key is unavailable.
    """

    def __init__(self, songs: List[Dict], kb: KnowledgeBase, api_key: str):
        self.enabled = False
        self._agent = None

        if not api_key:
            log.warning("ChatMusicAgent: no API key — agent disabled.")
            return

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langgraph.prebuilt import create_react_agent
            from langchain_core.messages import SystemMessage

            tools = build_tools(songs, kb)

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.7,
            )

            self._agent = create_react_agent(
                llm,
                tools,
                prompt=_SYSTEM_PROMPT,
            )
            self.enabled = True
            log.info("ChatMusicAgent ready with %d tools.", len(tools))

        except Exception as exc:
            log.error("ChatMusicAgent init failed: %s", exc, exc_info=True)
            self.enabled = False

    def chat(self, user_message: str, history: list) -> tuple[str, list]:
        """Run a chat turn and return (response_text, intermediate_steps).

        Args:
            user_message: The user's natural-language message.
            history: List of {"role": ..., "content": ...} dicts.

        Returns:
            Tuple of (response string, list of (action, observation) pairs).
        """
        if not self.enabled or self._agent is None:
            return (
                "The chat agent is unavailable — please add your GEMINI_API_KEY to .env.",
                [],
            )

        try:
            from langchain_core.messages import HumanMessage, AIMessage

            messages = []
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=user_message))

            result = self._agent.invoke({"messages": messages})

            # Extract final AI response and any tool calls from the message list
            response = ""
            steps = []
            for msg in result.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        steps.append((tc.get("name", "tool"), tc.get("args", {})))
                elif hasattr(msg, "content") and msg.content and type(msg).__name__ == "AIMessage":
                    response = _extract_text(msg.content)

            return response, steps

        except Exception as exc:
            log.error("ChatMusicAgent.chat failed: %s", exc, exc_info=True)
            return f"Sorry, something went wrong: {exc}", []
