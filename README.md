# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

image.png

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

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

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

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


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

