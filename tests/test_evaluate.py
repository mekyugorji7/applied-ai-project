"""Unit tests for scripts/evaluate.py evaluation harness."""

import os
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(REPO_ROOT, "data", "songs.csv")


@pytest.fixture(scope="module")
def songs():
    from src.recommender import load_songs
    return load_songs(CSV_PATH)


@pytest.fixture(scope="module")
def primary_results(songs):
    from scripts.evaluate import _evaluate_profile, PRIMARY_PROFILES
    return [_evaluate_profile(name, prefs, songs) for name, prefs in PRIMARY_PROFILES]


@pytest.fixture(scope="module")
def edge_results(songs):
    from scripts.evaluate import _evaluate_profile, EDGE_PROFILES
    return [_evaluate_profile(name, prefs, songs) for name, prefs in EDGE_PROFILES]


# ---------------------------------------------------------------------------
# Primary profile tests
# ---------------------------------------------------------------------------

def test_all_primary_profiles_pass(primary_results):
    """All three primary profiles should PASS the evaluation criteria."""
    for r in primary_results:
        assert r["pass_fail"] == "PASS", (
            f"{r['name']} FAILED: genre={r['top_genre_match']}, "
            f"mood={r['top_mood_match']}, top1={r['top1_score']}"
        )


def test_primary_top1_score_reasonable(primary_results):
    """Top-1 score for primary profiles should be at least 3.0."""
    for r in primary_results:
        assert r["top1_score"] >= 3.0, (
            f"{r['name']} top1_score {r['top1_score']} < 3.0"
        )


def test_primary_genre_match(primary_results):
    """Primary profiles should have top genre match."""
    for r in primary_results:
        assert r["top_genre_match"], f"{r['name']} genre mismatch"


def test_primary_mood_match(primary_results):
    """Primary profiles should have top mood match."""
    for r in primary_results:
        assert r["top_mood_match"], f"{r['name']} mood mismatch"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_edge_no_exceptions(edge_results):
    """No edge profile should raise an unhandled exception."""
    for r in edge_results:
        assert r.get("error") is None, (
            f"{r['name']} raised an exception: {r['error']}"
        )


def test_edge_conflicting_affect_pass(edge_results):
    result = next(r for r in edge_results if r["name"] == "EDGE_CONFLICTING_AFFECT")
    assert result["pass_fail"] == "PASS", (
        f"EDGE_CONFLICTING_AFFECT failed: top1={result['top1_score']}"
    )


def test_edge_unknown_labels_pass(edge_results):
    result = next(r for r in edge_results if r["name"] == "EDGE_UNKNOWN_LABELS")
    # Unknown labels should not produce genre or mood matches
    assert result["pass_fail"] == "PASS", (
        f"EDGE_UNKNOWN_LABELS failed: genre_match={result['top_genre_match']}, "
        f"mood_match={result['top_mood_match']}, top1={result['top1_score']}"
    )


def test_edge_bool_string_trap_no_exception(edge_results):
    result = next(r for r in edge_results if r["name"] == "EDGE_BOOL_STRING_TRAP")
    assert result["error"] is None


def test_edge_none_acoustic_no_exception(edge_results):
    result = next(r for r in edge_results if r["name"] == "EDGE_NONE_ACOUSTIC")
    assert result["error"] is None


def test_edge_sparse_no_exception(edge_results):
    result = next(r for r in edge_results if r["name"] == "EDGE_SPARSE")
    assert result["error"] is None


# ---------------------------------------------------------------------------
# Integration: run_evaluation produces correct summary
# ---------------------------------------------------------------------------

def test_run_evaluation_returns_zero_failures(songs, capsys):
    from scripts.evaluate import run_evaluation
    failures = run_evaluation(songs)
    assert failures == 0, f"Expected 0 failures, got {failures}"
