# 🎵 Music Recommender Simulation

## Project Summary

This project builds a small music recommender called My Vibe v1.0. It represents songs and user taste as structured features, then scores each song by how closely it matches the profile. The system ranks songs, shows top recommendations, and is tested with normal and edge-case profiles. I also evaluate strengths and limits, including bias from small data and simple scoring rules.

---

### Screenshot for Terminal App

<img width="779" height="800" alt="MyVIbe Screenshot" src="https://github.com/user-attachments/assets/068cbf2d-2c6f-4d04-b1b1-f409eb10229b" />


### Screenshot for Terminal Output Edge Cases

=============================================

----- Top Recommendations: USER_PREFS_CHILL_LOFI -----

=============================================

1. Midnight Coding                | Score: 5.84

---

1. Library Rain                   | Score: 5.80

---

1. Focus Flow                     | Score: 4.85

---

1. Spacewalk Thoughts             | Score: 3.57

---

1. Coffee Shop Stories            | Score: 2.72

---

=============================================

----- Top Recommendations: USER_PREFS_UPBEAT_POP -----

=============================================

1. Sunrise City                   | Score: 5.85

---

1. Gym Hero                       | Score: 4.75

---

1. Rooftop Lights                 | Score: 3.70

---

1. Voltage Verse                  | Score: 2.69

---

1. Calle Sin Fin                  | Score: 2.69

---

=============================================

----- Top Recommendations: USER_PREFS_SOFT_AMBIENT -----

=============================================

1. Spacewalk Thoughts             | Score: 5.91

---

1. Library Rain                   | Score: 3.73

---

1. Midnight Coding                | Score: 3.50

---

1. Willow & Ember                 | Score: 2.69

---

1. Winter Adagio                  | Score: 2.65

---

=============================================

----- Top Recommendations: EDGE_CONFLICTING_AFFECT -----

=============================================

1. Winter Adagio                  | Score: 3.37

---

1. Storm Runner                   | Score: 2.19

---

1. Boneforge March                | Score: 2.07

---

1. Gym Hero                       | Score: 2.05

---

1. Glass Cathedrals               | Score: 1.97

---

=============================================

----- Top Recommendations: EDGE_OUT_OF_RANGE -----

=============================================

1. Boneforge March                | Score: 3.79

---

1. Gym Hero                       | Score: 0.60

---

1. Storm Runner                   | Score: 0.57

---

1. Voltage Verse                  | Score: 0.54

---

1. Glass Cathedrals               | Score: 0.52

---

=============================================

----- Top Recommendations: EDGE_UNKNOWN_LABELS -----

=============================================

1. Coffee Shop Stories            | Score: 2.63

---

1. Willow & Ember                 | Score: 2.61

---

1. Dust Road Echoes               | Score: 2.58

---

1. Focus Flow                     | Score: 2.57

---

1. Midnight Coding                | Score: 2.55

---

=============================================

----- Top Recommendations: EDGE_BOOL_STRING_TRAP -----

=============================================

1. Library Rain                   | Score: 5.82

---

1. Midnight Coding                | Score: 5.79

---

1. Focus Flow                     | Score: 4.88

---

1. Spacewalk Thoughts             | Score: 3.59

---

1. Coffee Shop Stories            | Score: 2.77

---

=============================================

----- Top Recommendations: EDGE_NONE_ACOUSTIC -----

=============================================

1. Coffee Shop Stories            | Score: 5.51

---

1. Dust Road Echoes               | Score: 2.47

---

1. Willow & Ember                 | Score: 2.44

---

1. Focus Flow                     | Score: 2.39

---

1. Dawn Letters                   | Score: 2.38

---

=============================================

----- Top Recommendations: EDGE_SPARSE -----

=============================================

1. Midnight Coding                | Score: 1.00

---

1. Focus Flow                     | Score: 0.98

---

1. Coffee Shop Stories            | Score: 0.95

---

1. Library Rain                   | Score: 0.93

---

1. Low Tide Blues                 | Score: 0.91

---

## How The System Works

In production, recommenders usually blend many signals, what you played, skipped, or replayed; what similar users like; fresh or promoted titles; diversity and fairness rules; and often ML models trained on huge logs. My simulation is much simpler as it is content-based only. It never uses crowd behavior or listening history. It utilizes set weights like genre and mood matches, closeness on numeric vibe fields (especially energy, and optionally tempo, valence, danceability, plus an acoustic preference), then ranks by total score to produce top‑k suggestions. The higher the score, the better the match

### Song features:

id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness

### UserProfile features:

favorite_genre, favorite_mood, target_energy, likes_acoustic

### What features does each Song use?

Each Song is a small bundle of metadata + vibe numbers (from your CSV). The dataclass fields are:

Identity / display: id, title, artist
Style / vibe labels: genre, mood
Numeric audio-style features (0–1 or BPM): energy, tempo_bpm, valence, danceability, acousticness

Scoring uses genre, mood, and the numeric fields when the user prefs include matching keys (see below). Titles and artists are not part of the score; they’re for showing results.

### What does UserProfile store?

`UserProfile` is the object-oriented user model (what your tests use). It holds:

`favorite_genre`
`favorite_mood`
`target_energy`
`likes_acoustic`
`target_tempo_bpm`
`target_valence`
`target_danceability`

### How does Recommender compute a score for each song?

All scoring goes through score_song_with_explanation (or score_song, which calls it). Roughly:

Genre match: if the song’s genre equals favorite_genre, add 2.0 points.
Mood match: if the song’s mood equals favorite_mood, add 1.0 point.
Energy closeness: up to 1.0 extra points, using 1 − |song_energy − target_energy| (capped at zero).

Optional extras (only if those keys exist on the prefs dict): tempo vs target_tempo_bpm, valence vs target_valence, danceability vs target_danceability, and an acoustic alignment term from likes_acoustic and the song’s acousticness.

The high-level weights are declared at the top of the file (genre vs mood vs energy max, etc.).

```

WEIGHT_GENRE_MATCH = 2.0
WEIGHT_MOOD_MATCH = 1.0
...
WEIGHT_ENERGY_SIMILARITY_MAX = 1.0
...

```

The Recommender class turns each Song into a dict and runs the same scoring:

recommender.py
Lines 207-214

```
  recommender.py
  Lines 207-214
      def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        prefs = _profile_to_prefs(user)
        ranked: List[Tuple[Song, float]] = []
        for song in self.songs:
            s = score_song(_song_to_dict(song), prefs)
            ranked.append((song, s))
        ranked.sort(key=lambda x: (-x[1], x[0].id))
        return [s for s, _ in ranked[:k]]
```

### How do you choose which songs to recommend?

1. Score every song in the catalog (one pass).
2. Sort by score highest first. If two songs tie, lower id wins (stable, predictable ordering).
3. Take the first k (e.g. top 5).

### Algorithm Recipe

1. Load all tracks from data/songs.csv into a list.
2. Set the user’s targets (genre, mood, energy, acoustic preference, and optionally tempo / valence / danceability if you include them).
3. For each song, add points:
  - +2 if its genre matches your favorite genre
  - +1 if its mood matches your favorite mood
  - Up to +1 for how close its energy is to your target (closer = more points)
  - Up to +0.5 each for tempo, valence, and danceability
  - Up to +0.5 for acoustic fit (rewards higher acousticness if you like acoustic, lower if you don’t)
4. Sort songs by total score (highest first). If there’s a tie, lower id wins.

Return the top k songs (and a one-line explanation of what added points).

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):
  ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
  ```
2. Install dependencies

```bash
pip install -r requirements.txt
```

1. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

I tested the system with several user profiles.
These included chill, upbeat, and soft-listening profiles.
I also tried edge cases, like unknown labels and missing values.

For each test, I checked the top recommendations and compared whether they felt reasonable.
I compared outputs before and after weight changes.
The rankings changed in meaningful ways, which showed the model responds to scoring choices.

---

## Limitations and Risks

One pattern is that high-energy songs can appear often because energy has strong influence.
Another pattern is that exact genre and mood matches get a big boost.

This creates bias toward labels that appear in the small dataset.
Users with niche or mixed tastes may get weaker recommendations.
Related labels (like similar mood words) may not match well because matching is strict.
The top results can also feel repetitive because there is no diversity step.

---

## Reflection

Read and complete `model_card.md`:

**[Model Card](model_card.md)**

I learned that even a simple recommender can feel useful when user preferences are clear. Through this project, I learned that weight choices matter a lot, because small changes can reshuffle rankings greatly. One thing that surprised me was how confident the model can look even with weak or messy inputs. That made me see how easy it is for a system to hide bias behind clean-looking results. This changed how I think about music apps. Now, I'll pay more attention to data quality, diversity, and explanation, not just accuracy. If I were to extend this project, I would definetly add more songs and more adaptable personalization from user feedback when they like and skip songs.

---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

```

