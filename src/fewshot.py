"""Few-shot specialization: compare zero-shot vs. few-shot Gemini prompting depth."""

from __future__ import annotations

import re
from typing import Dict, List

try:
    from .gemini_client import GeminiClient
except ImportError:
    from gemini_client import GeminiClient


# ---------------------------------------------------------------------------
# Few-shot examples used in the prompt
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "profile": "Chill Lofi listener: energy ~0.42, BPM ~78, acousticness ~0.75, mood: chill",
        "response": (
            "These recommendations reflect the psychoacoustic properties of lo-fi hip-hop. "
            "At 70-90 BPM with a swing-feel derived from triplet subdivision, these tracks "
            "inhabit the 'relaxed activity zone' on the arousal-valence circumplex — low arousal "
            "paired with moderately positive affect. Acousticness values above 0.70 are critical: "
            "the warm timbre of acoustic piano, muted guitar, and vinyl crackle reduces cognitive "
            "load compared to bright electronic timbres, making these tracks ideal for studying "
            "without attentional interruption. The energy range of 0.35-0.50 keeps the listener "
            "engaged without triggering the sympathetic nervous system activation that would break "
            "concentration. Valence around 0.55-0.65 provides positive hedonic tone while avoiding "
            "the emotional intensity that could redirect attention from the task at hand."
        ),
    },
    {
        "profile": "Upbeat Pop listener: energy ~0.85, BPM ~120, danceability ~0.80, mood: happy",
        "response": (
            "These selections align with the circumplex model of affect's high-arousal, "
            "high-valence quadrant — the emotional territory of joy and positive excitation. "
            "The 120 BPM tempo sits precisely at the motivational threshold where rhythmic "
            "entrainment naturally elevates heart rate, increases dopamine release, and promotes "
            "physical movement. The 4/4 time signature's predictable pulse reduces cognitive "
            "processing demands, freeing attentional resources for enjoyment rather than rhythm "
            "tracking. High danceability scores (0.78-0.88) result from the synergistic interaction "
            "between prominent four-on-the-floor kick patterns, syncopated hi-hat programming, and "
            "bass lines with strong beat-1 emphasis. Valence above 0.80 in these tracks reflects "
            "major-key harmony, ascending melodic phrases, and production choices — bright high "
            "frequencies, minimal compression release — that reinforce positive affect. The low "
            "acousticness (0.05-0.25) enables the clean, wide-frequency-range production that "
            "maximizes perceptual 'lift.'"
        ),
    },
    {
        "profile": "Soft Ambient listener: energy ~0.30, BPM ~62, acousticness ~0.88, mood: chill",
        "response": (
            "These tracks embody Brian Eno's foundational ambient principle: music that should be "
            "'ignorable but listenable' — present enough to enrich the acoustic environment but "
            "structured to make no cognitive demands. The 60-65 BPM pulse aligns with the resting "
            "heart rate range, facilitating physiological entrainment and activating the "
            "parasympathetic nervous system for genuine relaxation response. Energy values below "
            "0.35 eliminate the sharp transients and dynamic contrasts that would trigger the "
            "orienting response, allowing sustained background processing. High acousticness "
            "(0.85-0.95) reflects the use of extended-decay piano notes, slowly evolving string "
            "pads, and field recordings — timbres that research shows are processed as "
            "'non-threatening' and conducive to the default mode network activity associated with "
            "rest and gentle creativity. Sparse percussion maintains temporal structure without "
            "rhythmic imposition, preserving the listener's sense of temporal flow."
        ),
    },
]


def _format_examples_for_prompt() -> str:
    lines = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Profile: {ex['profile']}")
        lines.append(f"Response: {ex['response']}")
        lines.append("")
    return "\n".join(lines)


class FewShotRecommender:
    """Compare zero-shot vs. few-shot Gemini recommendation depth."""

    MUSIC_TERMS = [
        "bpm", "tempo", "energy", "acousticness", "danceability", "valence",
        "timbre", "harmonic", "melody", "rhythm", "beat", "frequency",
        "acoustic", "electronic", "arousal", "circumplex", "entrainment",
        "dopamine", "parasympathetic", "sympathetic", "swing", "syncopation",
        "pentatonic", "chord", "bass", "treble", "dynamic", "transient",
        "lo-fi", "lofi", "vinyl", "reverb", "decay", "sustain", "amplitude",
        "waveform", "oscillator", "synthesis", "percussion", "groove",
        "4/4", "backbeat", "kick", "snare", "hi-hat", "psychoacoustic",
        "hedonic", "affect", "valence", "cognitive", "cortisol", "neural",
        "chromatic", "diatonic", "polyrhythm", "counterpoint",
    ]

    def __init__(self, gemini: GeminiClient = None):
        self.gemini = gemini or GeminiClient()

    def _count_music_terms(self, text: str) -> int:
        text_lower = text.lower()
        return sum(1 for term in self.MUSIC_TERMS if term in text_lower)

    def _count_numeric_refs(self, text: str) -> int:
        return len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:bpm|hz|db|%|ms)?\b", text, re.IGNORECASE))

    def _count_sentences(self, text: str) -> int:
        sentences = re.split(r"[.!?]+", text.strip())
        return max(1, len([s for s in sentences if s.strip()]))

    def _depth_score(self, text: str) -> float:
        terms = self._count_music_terms(text)
        nums = self._count_numeric_refs(text)
        sentences = self._count_sentences(text)
        return round((terms + nums) / sentences * 3, 1)

    def recommend_baseline(self, user_prefs: dict, top_songs: list) -> str:
        song_list = "\n".join(
            f"- {s['title']} by {s['artist']} (score: {sc:.2f})"
            for s, sc, _ in top_songs
        )
        prompt = (
            "You are a music recommendation assistant. "
            "Explain why these songs fit the listener's preferences.\n\n"
            f"User preferences: {user_prefs}\n\n"
            f"Top recommended songs:\n{song_list}"
        )
        return self.gemini.generate(prompt, max_tokens=256)

    def recommend_specialized(self, user_prefs: dict, top_songs: list) -> str:
        song_list = "\n".join(
            f"- {s['title']} by {s['artist']} (score: {sc:.2f})"
            for s, sc, _ in top_songs
        )
        examples = _format_examples_for_prompt()
        prompt = (
            "You are an expert musicologist and audio engineer with deep knowledge of "
            "psychoacoustics, music theory, and listener psychology. "
            "When explaining music recommendations, always reference specific audio features "
            "(BPM, energy, acousticness, valence, danceability), music theory concepts, "
            "and listener psychology research. Be precise and technical.\n\n"
            f"{examples}"
            "Now explain why the following songs fit this listener's profile, using the same "
            "level of technical music-theory depth as the examples above.\n\n"
            f"Profile: {user_prefs}\n\n"
            f"Top recommended songs:\n{song_list}"
        )
        return self.gemini.generate(prompt, max_tokens=512)

    def compare(self, user_prefs: dict, top_songs: list) -> dict:
        """Run both prompts, compute metrics, return results dict."""
        baseline_text = self.recommend_baseline(user_prefs, top_songs)
        specialized_text = self.recommend_specialized(user_prefs, top_songs)

        baseline_terms = self._count_music_terms(baseline_text)
        specialized_terms = self._count_music_terms(specialized_text)
        baseline_nums = self._count_numeric_refs(baseline_text)
        specialized_nums = self._count_numeric_refs(specialized_text)
        baseline_depth = self._depth_score(baseline_text)
        specialized_depth = self._depth_score(specialized_text)

        return {
            "baseline_text": baseline_text,
            "specialized_text": specialized_text,
            "baseline_terms": baseline_terms,
            "specialized_terms": specialized_terms,
            "baseline_nums": baseline_nums,
            "specialized_nums": specialized_nums,
            "baseline_depth": baseline_depth,
            "specialized_depth": specialized_depth,
        }


def run_fewshot(songs: list) -> None:
    """Entry point called from main.py --fewshot flag."""
    print("\n" + "=" * 60)
    print("FEW-SHOT vs. ZERO-SHOT COMPARISON")
    print("=" * 60)

    try:
        from .recommender import recommend_songs
        from .gemini_client import GeminiClient
        from .main import TASTE_PROFILE
    except ImportError:
        from recommender import recommend_songs
        from gemini_client import GeminiClient
        from main import TASTE_PROFILE

    gemini = GeminiClient()
    top_songs = recommend_songs(TASTE_PROFILE, songs, k=5)
    recommender = FewShotRecommender(gemini)
    results = recommender.compare(TASTE_PROFILE, top_songs)

    print("\n[BASELINE — Zero-Shot]\n")
    print(results["baseline_text"])
    print("\n[FEW-SHOT — Specialized]\n")
    print(results["specialized_text"])

    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)
    print(f"{'Metric':<25} | {'Baseline':>8} | {'Few-Shot':>8} | {'Delta':>8}")
    print("-" * 60)

    terms_delta = results["specialized_terms"] - results["baseline_terms"]
    nums_delta = results["specialized_nums"] - results["baseline_nums"]
    depth_delta = results["specialized_depth"] - results["baseline_depth"]

    print(f"{'Music domain terms':<25} | {results['baseline_terms']:>8} | {results['specialized_terms']:>8} | {terms_delta:>+8}")
    print(f"{'Numeric references':<25} | {results['baseline_nums']:>8} | {results['specialized_nums']:>8} | {nums_delta:>+8}")
    print(f"{'Technical depth score':<25} | {results['baseline_depth']:>8.1f} | {results['specialized_depth']:>8.1f} | {depth_delta:>+8.1f}")

    direction = "improvement" if depth_delta >= 0 else "regression"
    print(f"\nDelta: {depth_delta:+.1f} (measurable specialization {direction} demonstrated)")
    print("=" * 60)
