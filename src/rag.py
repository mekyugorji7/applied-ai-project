"""RAG module: TF-IDF retrieval over a local knowledge base of genre/mood documents."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict

from .logger import get_logger

log = get_logger(__name__)


class KnowledgeBase:
    """TF-IDF retrieval over plain-text knowledge base documents."""

    def __init__(self, kb_dir: str = "data/knowledge_base"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        self._np = np
        self._docs: List[Dict] = []

        kb_path = Path(kb_dir)
        if not kb_path.is_dir():
            raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

        for txt_file in sorted(kb_path.glob("*.txt")):
            content = txt_file.read_text(encoding="utf-8").strip()
            self._docs.append({"filename": txt_file.name, "content": content})

        if not self._docs:
            raise ValueError(f"No .txt files found in {kb_dir}")

        self._vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1)
        corpus = [d["content"] for d in self._docs]
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        log.info("KnowledgeBase loaded: %d documents from %s", len(self._docs), kb_dir)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Return top_k most relevant documents for the query with cosine similarity scores."""
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "filename": self._docs[idx]["filename"],
                    "content": self._docs[idx]["content"],
                    "score": float(scores[idx]),
                }
            )
        log.debug(
            "retrieve(%r, top_k=%d) → %s",
            query[:60],
            top_k,
            [r["filename"] for r in results],
        )
        return results

    def build_context(self, user_prefs: dict) -> str:
        """Translate a numeric user profile into a text query and retrieve relevant docs."""
        query_parts: List[str] = []

        genre = user_prefs.get("favorite_genre")
        if genre:
            query_parts.append(genre)

        mood = user_prefs.get("favorite_mood")
        if mood:
            query_parts.append(mood)

        energy = user_prefs.get("target_energy")
        if energy is not None:
            try:
                e = float(energy)
                if e < 0.35:
                    query_parts.append("very low energy soft quiet")
                elif e < 0.55:
                    query_parts.append("low energy calm")
                elif e < 0.70:
                    query_parts.append("moderate energy")
                else:
                    query_parts.append("high energy driving")
            except (TypeError, ValueError):
                pass

        tempo = user_prefs.get("target_tempo_bpm")
        if tempo is not None:
            try:
                t = float(tempo)
                if t < 70:
                    query_parts.append("very slow tempo")
                elif t < 90:
                    query_parts.append("slow tempo relaxed BPM")
                elif t < 110:
                    query_parts.append("medium tempo")
                else:
                    query_parts.append("fast tempo upbeat")
            except (TypeError, ValueError):
                pass

        likes_acoustic = user_prefs.get("likes_acoustic")
        if likes_acoustic is True:
            query_parts.append("acoustic organic warm instruments")
        elif likes_acoustic is False:
            query_parts.append("electronic digital production")

        valence = user_prefs.get("target_valence")
        if valence is not None:
            try:
                v = float(valence)
                if v < 0.40:
                    query_parts.append("melancholic dark sad")
                elif v < 0.60:
                    query_parts.append("neutral bittersweet")
                else:
                    query_parts.append("positive uplifting happy")
            except (TypeError, ValueError):
                pass

        query = " ".join(query_parts) if query_parts else "music recommendation"
        docs = self.retrieve(query, top_k=3)

        context_parts = [
            f"[{doc['filename']} (relevance: {doc['score']:.2f})]\n{doc['content']}"
            for doc in docs
        ]
        return "\n\n".join(context_parts)
