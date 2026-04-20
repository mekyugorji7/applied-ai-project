# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

MyVibe v1.0

---

## 2. Intended Use

Goal / Task: This recommender suggests songs a user might like based on their stated preferences.

It assumes users can describe their taste with a few traits, like genre, mood, and energy.

**Intended use:** quick music suggestions for experiments and demos.

**Not intended use:** medical, mental health, hiring, legal, or other high-stakes decisions. It is also not meant to fully replace real streaming recommendation systems.

---

## 3. How the Model Works

The model compares each song to a user profile. It uses genre, mood, energy, tempo, valence, danceability, and acousticness.

If genre or mood matches, the song gets extra points. For numeric traits, songs get more points when they are close to the user preference. All points are added into one final score, then songs are ranked from highest to lowest.

I adjusted the scoring weights so energy has more effect and genre has a bit less effect than before.

---

## 4. Data

The catalog has 20 songs.
It includes genres like pop, rock, jazz, hip-hop, lofi, ambient, EDM, and acoustic.
Moods include calm, happy, sad, energetic, and mellow.

The dataset is small, so many tastes are missing.
It has limited artist variety and does not include lyrics, release year, language, or listening history.
I did not add new songs in this version.

---

## 5. Strengths

The model works best for users with clear and simple preferences. It performs well when genre and mood are specific. It also does a good job matching overall energy level.

In tests, chill profiles often got lofi or ambient songs near the top. Upbeat profiles often got pop or dance tracks near the top. These results matched my expectations.

---

## 6. Limitations and Bias

One pattern is that high-energy songs can appear often because energy has strong influence.
Another pattern is that exact genre and mood matches get a big boost.

This creates bias toward labels that appear in the small dataset.
Users with niche or mixed tastes may get weaker recommendations.
Related labels (like similar mood words) may not match well because matching is strict.
The top results can also feel repetitive because there is no diversity step.

## 7. Evaluation

I tested the system with several user profiles.
These included chill, upbeat, and soft-listening profiles.
I also tried edge cases, like unknown labels and missing values.

For each test, I checked the top recommendations and compared whether they felt reasonable.
I compared outputs before and after weight changes.
The rankings changed in meaningful ways, which showed the model responds to scoring choices.

---

### 8. Design Decisions

Why weighted arithmetic instead of a neural recommender?

The catalog has 20 songs — far too small for any learned model to generalize. Weighted arithmetic scoring is fully deterministic, every recommendation comes with an exact reason ("genre match +1.0, energy closeness +1.84"), and the weights are easy to tune and explain. A neural approach would add significant complexity, require far more data, and produce a black box for a problem that simply doesn't require one.

Why TF-IDF for RAG instead of embeddings?

The knowledge base is 23 short plain-text documents with dense genre/mood vocabulary. TF-IDF excels at keyword-rich retrieval on small corpora, requires no API calls, runs entirely offline, and is deterministic, meaning RAG retrieval tests are reproducible without mocking. The trade-off is that TF-IDF misses semantic synonyms (like "melancholy" vs "sad"). For a 23-doc set, that cost is acceptable.

Why LangChain + LangGraph for the chat agent instead of a raw Gemini call?

A single Gemini call cannot look up catalog data, retrieve KB documents, or compute similarity scores — it would have to hallucinate all of them. LangGraph's create_react_agent gives the model genuine tool-calling capability: it reasons about what it needs, calls the appropriate tools, and synthesizes a grounded response. Tool calls are captured and rendered in the UI, making reasoning visible rather than opaque.

### Trade-offs accepted

| Decision              | Upside                                | Cost                                              |
| --------------------- | ------------------------------------- | ------------------------------------------------- |
| TF-IDF RAG            | No API, deterministic, fully testable | Misses semantic synonyms                          |
| 20-song catalog       | Fast, controllable, fits in memory    | Limited genre and mood coverage                   |
| Gemini 2.5 Flash      | Free tier, capable, fast              | Rate-limited; degrades gracefully without a key   |
| LangGraph agent       | Observable multi-step reasoning       | ~3–6 s per chat turn vs. ~1 s for a direct prompt |
| No persistent storage | Simple, zero privacy surface          | Profile resets on every page reload               |

### Stretch scope

*Multi-source in this repo means combined use of the song catalog and the knowledge base, rankings are computed from `songs.csv`, TF‑IDF retrieval uses the KB documents and the chat agent can access both via tools. The Few-shot vs zero-shot comparison runs from the CLI (`python -m src.main --fewshot`) and not the Streamlit app. The evaluation harness (`scripts/evaluate.py`) runs `recommend_songs` on predefined profiles and prints pass/fail summaries, it doesn't call Gemini or the chat agent.


## 9. Future Work

Ideas for how I would improve the model next:

- Add more songs and broader genres to reduce bias from small data.
- Add a diversity rule so the top results are less repetitive.
- Show simple explanations for each recommendation, like "matched your mood and energy."

---

## 10. AI Reflection

### How I used AI during development

I used an AI coding assistant across design, implementation, and debugging. Early on I used it to help compare architectural options (for example, RAG versus a single unprompted model call, and where input validation should sit in the pipeline). During coding it helped scaffold repetitive pieces (Streamlit layout patterns, pytest cases, LangGraph tool wrappers) and walk through stack traces when libraries had moved APIs between versions. For documentation, I used it to tighten structure and wording in the README and this model card, then checked each technical claim against the actual code and test output so the docs stayed accurate.


### Describe your collaboration with AI during this project. Identify one helpful and one flawed suggestion.

#### A helpful suggestion

The most useful recurring advice was to keep automated tests on deterministic behavior only, scoring, ranking, RAG retrieval shape, and the evaluation harness. This helped me avoid verifying exact Gemini text in `pytest`. Using stable tests that still protect the core recommender and retrieval helped a lot in testing, while qualitative checks on explanations stayed in the human review process.

#### A flawed suggestion

The assistant often surfaced LangChain examples from older tutorials (`AgentExecutor`, older tool-calling agent patterns) that did not match LangChain/LangGraph in this repo. Following those snippets blindly would have led to import errors or dead code paths until I cross-checked the installed versions and official migration notes. I ended up standardizing on LangGraph `create_react_agent` after reading current docs and not the outdated snippets. I have to make sure to treat AI-generated code as a draft, and then verify before I integrate.

### How this connects to the system

AI assistance sped up boilerplate and exploration, but it didn't remove the need to manually judge whether rankings and chat answers felt grounded for real profiles. System limitations (small catalog, strict labels, TF-IDF synonym gaps) are still rea. Where the assistant helped most was making those limitations easier to see: guardrails, evaluation profiles, and observable agent tool calls so weak inputs show up as low confidence or visible traces instead of silent failure.

### What are the limitations or biases in your system?

The biggest limitation is dataset size: a 20-song catalog cannot represent the full range of real listening tastes. The scoring logic also favors strong energy matches and exact genre/mood labels, which can bias results toward common labels and under-serve niche or mixed preferences. TF-IDF retrieval is deterministic and useful on a small corpus, but it can miss semantic synonyms (for example, "melancholy" vs "sad"). Because there is no diversity re-ranking step, top recommendations can also feel repetitive.

### Could your AI be misused, and how would you prevent that?

Yes, a user could treat the recommendations as authoritative for emotional or mental-health guidance, or assume the chat assistant is always correct because it sounds confident. To reduce misuse, this project applies scope guardrails in documentation and UX. It is explicitly framed as a demo recommender. Reliability is improved by grounding responses in retrieval + scoring tools rather than free-form generation, showing tool traces where possible, and keeping deterministic tests for core logic so failures are visible. 

### What surprised you while testing your AI's reliability?

The most surprising result was how quickly output quality improved when the model had grounded context. The same Gemini model that gave generic answers with a raw prompt became much more specific when given retrieved KB facts and scored recommendations. Another surprise was that the deterministic components (scoring + retrieval shape tests) were much easier to validate than generated text, which reinforced the decision to separate deterministic automated tests from qualitative AI-output review.

---

## 11. Personal Reflection

I learned that even a simple recommender can feel useful when user preferences are clear. Through this project, I learned that weight choices matter a lot, because small changes can reshuffle rankings greatly. One thing that surprised me was how confident the model can look even with weak or messy inputs. That made me see how easy it is for a system to hide bias behind clean-looking results. This changed how I think about music apps. Now, I'll pay more attention to data quality, diversity, and explanation, not just accuracy. If I were to extend this project, I would definetly add more songs and more adaptable personalization from user feedback when they like and skip songs.


---

## 12. Testing Summary

**Automated tests:** 22/22 passed

```text
pytest tests/  →  22 passed in 0.86s
```

### Test coverage

| Test file | What it covers |
| --- | --- |
| `test_recommender.py` | Scoring correctness, sort order, explanation strings |
| `test_rag.py` | KB loading, retrieval count and structure, score ordering, genre document surfacing |
| `test_evaluate.py` | All 8 profiles pass/fail, top-1 score thresholds, zero unhandled exceptions |

**Evaluation harness:** 8/8 profiles passed

| Profile | Top-1 score | Result |
| --- | --- | --- |
| Chill Lofi (genre + mood + energy) | 5.84 / 6.00 | PASS |
| Upbeat Pop (genre + mood + energy) | 5.82 / 6.00 | PASS |
| Soft Ambient (genre + mood + energy) | 5.89 / 6.00 | PASS |
| Edge: Conflicting affect (happy mood, low energy) | 3.15 / 6.00 | PASS |
| Edge: Unknown genre + mood labels | 3.53 / 6.00 | PASS |
| Edge: `likes_acoustic` passed as string `"True"` | 5.77 / 6.00 | PASS |
| Edge: `likes_acoustic` = `None` | 5.49 / 6.00 | PASS |
| Edge: Sparse profile (energy only) | 2.00 / 6.00 | PASS |

**Average top-1 score** across all 8 profiles: **4.69 / 6.00**

### What worked

- **Scoring engine:** Produced correct rankings on all three primary profiles before any tuning. Predictable enough that the first implementation was essentially correct
- **RAG retrieval:** 80% precision (4/5 test queries returned the correct genre document at rank 1). Fully deterministic and reproducible across runs.
- **Chat agent:** Successfully calls multiple tools in sequence before answering, and never invented a song title in any test interaction.

### What didn't work / was harder than expected

- **LangChain version fragmentation:** `AgentExecutor` and `create_tool_calling_agent` — the standard imports in every tutorial — were removed in LangChain 1.x. The API moved to LangGraph's `create_react_agent`. This cost several hours and was poorly documented. The lesson: LLM framework APIs move faster than their documentation.
- **Gemini content format:** The LangChain-wrapped Gemini model returns message content as a list of typed dicts rather than a plain string. The chat panel displayed raw Python objects until a `_extract_text()` normalizer was added.
- **RAG lofi miss:** On the query "lofi chill acoustic", TF-IDF ranked `mood_chill.txt` above `genre_lofi.txt` because the mood term had higher IDF weight in this corpus. Weighting genre tokens more heavily would fix it, but wasn't worth the added complexity for 23 documents.

### Confidence scoring

Confidence is `top_score / 6.0` (proportion of the theoretical maximum):

- **Full profile** (genre + mood + energy): avg **0.73** — strong, well-separated picks
- **Sparse profile** (energy only): **0.33** — correctly low; the system signals it is guessing
- **Top-1 vs top-2 score gap** on named profiles: **0.14–1.12** pts — larger gaps mean higher certainty in the top pick

### Reflection

**Grounding matters more than model capability:** When Gemini received a raw song list with no context, it produced generic, vague explanations. When the same prompt included retrieved knowledge base documents — concrete facts about BPM ranges, acousticness, and psychoacoustic effects — the explanations became specific and accurate. The model's capability didn't change. What changed was the quality of the information it was given to work with. RAG is less about making a smarter model and more about making sure the model has what it needs before it speaks.

**Observability is not a nice-to-have:** The shift from a single Gemini call to a LangGraph agent with visible tool calls felt like the difference between trusting a result and understanding it. When the agent calls `retrieve_knowledge()` then `score_and_recommend()` before answering, you can see exactly why it said what it said. That traceability is what makes an AI system debuggable — and debuggability is what makes it trustworthy enough to actually use.

**Testing AI systems requires a clean separation.** All 22 automated tests target deterministic behavior, they never call the Gemini API. This makes the suite fast, stable, and reproducible. Testing the AI layer is a different problem as it requires defining qualitative criteria ("does the explanation cite BPM?") and checking them through human review. Trying to write automated assertions for non-deterministic text generation produces tests that break whenever the model changes.

### What I'd do differently with more time

- Add semantic embeddings alongside TF-IDF to catch synonym mismatches in RAG retrieval
- Make the Spotify integration work so coverage across songs, genres, moods is more meaningful
- Add session persistence so profiles survive page reloads


### Honest takeaway

This project demonstrates that a useful, explainable, and testable recommendation system doesn't require a large dataset, a GPU, or a complex ML pipeline. The AI is one tool surrounded by deterministic logic, input validation, retrieval infrastructure, and automated tests. That surrounding structure is what makes the AI component reliable.

