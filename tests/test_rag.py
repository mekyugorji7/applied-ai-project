"""Unit tests for src/rag.py KnowledgeBase."""

import os
import pytest

# Ensure we can find data/knowledge_base relative to repo root
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
KB_DIR = os.path.join(REPO_ROOT, "data", "knowledge_base")


@pytest.fixture(scope="module")
def kb():
    from src.rag import KnowledgeBase
    return KnowledgeBase(KB_DIR)


def test_kb_loads_documents(kb):
    """Knowledge base should load at least 20 documents."""
    assert len(kb._docs) >= 20


def test_retrieve_returns_top_k(kb):
    """retrieve() should return exactly top_k results."""
    results = kb.retrieve("lofi chill acoustic", top_k=3)
    assert len(results) == 3


def test_retrieve_result_structure(kb):
    """Each retrieved doc should have filename, content, and score keys."""
    results = kb.retrieve("ambient energy low", top_k=2)
    for doc in results:
        assert "filename" in doc
        assert "content" in doc
        assert "score" in doc
        assert isinstance(doc["score"], float)


def test_retrieve_scores_descending(kb):
    """Results should be ordered by score descending."""
    results = kb.retrieve("pop happy upbeat dance", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_lofi_query_hits_lofi_doc(kb):
    """A lofi-specific query should surface genre_lofi.txt in the top results."""
    results = kb.retrieve("lofi chill acoustic vinyl studying", top_k=3)
    filenames = [r["filename"] for r in results]
    assert any("lofi" in f for f in filenames), (
        f"Expected genre_lofi.txt in top results, got: {filenames}"
    )


def test_retrieve_ambient_query_hits_ambient_doc(kb):
    """An ambient-specific query should surface genre_ambient.txt in top results."""
    results = kb.retrieve("ambient very low energy sparse slow sleep", top_k=3)
    filenames = [r["filename"] for r in results]
    assert any("ambient" in f for f in filenames), (
        f"Expected genre_ambient.txt in top results, got: {filenames}"
    )


def test_build_context_returns_string(kb):
    """build_context() should return a non-empty string for any profile."""
    prefs = {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.42,
        "target_tempo_bpm": 78.0,
        "likes_acoustic": True,
    }
    context = kb.build_context(prefs)
    assert isinstance(context, str)
    assert len(context) > 100


def test_build_context_sparse_profile(kb):
    """build_context() should not crash on a sparse profile."""
    context = kb.build_context({"target_energy": 0.42})
    assert isinstance(context, str)
    assert len(context) > 0


def test_kb_missing_dir_raises():
    """KnowledgeBase should raise FileNotFoundError for a missing directory."""
    from src.rag import KnowledgeBase
    with pytest.raises(FileNotFoundError):
        KnowledgeBase("/nonexistent/path/that/does/not/exist")
