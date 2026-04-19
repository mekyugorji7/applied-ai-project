#!/usr/bin/env python3
"""Evaluation harness: runs all profiles through the recommender and reports pass/fail metrics."""

from __future__ import annotations

import sys
import os
from datetime import date
from typing import Dict, List, Tuple

# Allow running as both `python scripts/evaluate.py` and `python -m src.main --evaluate`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.recommender import load_songs, recommend_songs
    from src.main import (
        USER_PREFS_CHILL_LOFI,
        USER_PREFS_UPBEAT_POP,
        USER_PREFS_SOFT_AMBIENT,
        EDGE_CONFLICTING_AFFECT,
        EDGE_UNKNOWN_LABELS,
        EDGE_BOOL_STRING_TRAP,
        EDGE_NONE_ACOUSTIC,
        EDGE_SPARSE,
    )
except ImportError:
    from recommender import load_songs, recommend_songs
    from main import (
        USER_PREFS_CHILL_LOFI,
        USER_PREFS_UPBEAT_POP,
        USER_PREFS_SOFT_AMBIENT,
        EDGE_CONFLICTING_AFFECT,
        EDGE_UNKNOWN_LABELS,
        EDGE_BOOL_STRING_TRAP,
        EDGE_NONE_ACOUSTIC,
        EDGE_SPARSE,
    )


PRIMARY_PROFILES: List[Tuple[str, Dict]] = [
    ("USER_PREFS_CHILL_LOFI", USER_PREFS_CHILL_LOFI),
    ("USER_PREFS_UPBEAT_POP", USER_PREFS_UPBEAT_POP),
    ("USER_PREFS_SOFT_AMBIENT", USER_PREFS_SOFT_AMBIENT),
]

EDGE_PROFILES: List[Tuple[str, Dict]] = [
    ("EDGE_CONFLICTING_AFFECT", EDGE_CONFLICTING_AFFECT),
    ("EDGE_UNKNOWN_LABELS", EDGE_UNKNOWN_LABELS),
    ("EDGE_BOOL_STRING_TRAP", EDGE_BOOL_STRING_TRAP),
    ("EDGE_NONE_ACOUSTIC", EDGE_NONE_ACOUSTIC),
    ("EDGE_SPARSE", EDGE_SPARSE),
]


def _evaluate_profile(
    name: str, prefs: Dict, songs: List[Dict], k: int = 5
) -> Dict:
    """Run one profile through the recommender and compute all metrics. Never raises."""
    result = {
        "name": name,
        "top_genre_match": False,
        "top_mood_match": False,
        "top1_score": 0.0,
        "genre_hit_rate": 0.0,
        "avg_top5_score": 0.0,
        "pass_fail": "FAIL",
        "error": None,
    }
    try:
        recs = recommend_songs(prefs, songs, k=k)
        if not recs:
            return result

        top_song, top_score, _ = recs[0]
        result["top1_score"] = round(top_score, 2)

        fav_genre = prefs.get("favorite_genre")
        fav_mood = prefs.get("favorite_mood")

        if fav_genre is not None:
            result["top_genre_match"] = top_song.get("genre") == fav_genre
        if fav_mood is not None:
            result["top_mood_match"] = top_song.get("mood") == fav_mood

        if fav_genre is not None:
            genre_matches = sum(
                1 for s, _, _ in recs if s.get("genre") == fav_genre
            )
            result["genre_hit_rate"] = round(genre_matches / len(recs) * 100, 1)

        result["avg_top5_score"] = round(
            sum(sc for _, sc, _ in recs) / len(recs), 2
        )

        result["pass_fail"] = _determine_pass_fail(name, result)

    except Exception as exc:
        result["error"] = str(exc)
        # Edge profiles that are tested for "no exception" automatically fail if one occurs
        if name in ("EDGE_BOOL_STRING_TRAP", "EDGE_NONE_ACOUSTIC", "EDGE_SPARSE"):
            result["pass_fail"] = "FAIL"
        else:
            result["pass_fail"] = "FAIL"

    return result


def _determine_pass_fail(name: str, metrics: Dict) -> str:
    if name in ("USER_PREFS_CHILL_LOFI", "USER_PREFS_UPBEAT_POP", "USER_PREFS_SOFT_AMBIENT"):
        if (
            metrics["top_genre_match"]
            and metrics["top_mood_match"]
            and metrics["top1_score"] >= 3.0
        ):
            return "PASS"
        return "FAIL"

    if name == "EDGE_CONFLICTING_AFFECT":
        return "PASS" if metrics["top1_score"] <= 3.5 else "FAIL"

    if name == "EDGE_UNKNOWN_LABELS":
        # Unknown genre/mood labels should produce no genre or mood match,
        # but numeric features (energy, tempo, valence, danceability) can still
        # contribute meaningful scores — so we only verify the labels were not matched.
        return (
            "PASS"
            if not metrics["top_genre_match"] and not metrics["top_mood_match"]
            else "FAIL"
        )

    if name in ("EDGE_BOOL_STRING_TRAP", "EDGE_NONE_ACOUSTIC", "EDGE_SPARSE"):
        # PASS = no exception was raised (error field stays None)
        return "PASS" if metrics.get("error") is None else "FAIL"

    return "FAIL"


def _bool_label(value: bool) -> str:
    return "YES" if value else "NO"


def run_evaluation(songs: List[Dict]) -> int:
    """Run full evaluation suite. Returns number of failures."""
    today = date.today().isoformat()

    print("=" * 60)
    print(f"MUSIC RECOMMENDER EVALUATION REPORT — {today}")
    print("=" * 60)

    all_results = []

    # Primary profiles
    print("\nPRIMARY PROFILES")
    header = f"{'Profile':<25} | {'Genre':>5} | {'Mood':>4} | {'Top1':>5} | {'Genre%':>6} | {'Pass':>4}"
    print(header)
    print("-" * len(header))

    primary_results = [
        _evaluate_profile(name, prefs, songs) for name, prefs in PRIMARY_PROFILES
    ]
    for r in primary_results:
        print(
            f"{r['name']:<25} | {_bool_label(r['top_genre_match']):>5} | "
            f"{_bool_label(r['top_mood_match']):>4} | {r['top1_score']:>5.2f} | "
            f"{r['genre_hit_rate']:>5.1f}% | {r['pass_fail']:>4}"
        )
    all_results.extend(primary_results)

    # Edge profiles
    print("\nEDGE-CASE PROFILES")
    print(f"{'Profile':<25} | {'Top1':>5} | {'Notes':<30} | {'Pass':>4}")
    print("-" * 75)

    edge_results = [
        _evaluate_profile(name, prefs, songs) for name, prefs in EDGE_PROFILES
    ]
    edge_notes = {
        "EDGE_CONFLICTING_AFFECT": "score <= 3.5",
        "EDGE_UNKNOWN_LABELS": "no genre/mood match",
        "EDGE_BOOL_STRING_TRAP": "no exception raised",
        "EDGE_NONE_ACOUSTIC": "no exception raised",
        "EDGE_SPARSE": "no exception raised",
    }
    for r in edge_results:
        note = edge_notes.get(r["name"], "—")
        err_info = f" [ERR: {r['error']}]" if r.get("error") else ""
        print(
            f"{r['name']:<25} | {r['top1_score']:>5.2f} | {note + err_info:<30} | {r['pass_fail']:>4}"
        )
    all_results.extend(edge_results)

    # Summary
    passed = sum(1 for r in all_results if r["pass_fail"] == "PASS")
    total = len(all_results)
    top1_scores = [r["top1_score"] for r in all_results if r["top1_score"] > 0]
    avg_top1 = sum(top1_scores) / len(top1_scores) if top1_scores else 0.0

    print()
    print("=" * 60)
    print(f"SUMMARY: {passed}/{total} PASSED | Avg top1: {avg_top1:.2f}/6.00")
    print("=" * 60)

    return total - passed


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)
    failures = run_evaluation(songs)
    sys.exit(0 if failures == 0 else 1)
