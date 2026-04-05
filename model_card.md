# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

MyVibe v1.0

---

## 2. Intended Use  

Goal / Task: This recommender suggests songs a user might like based on their stated preferences.

It assumes users can describe their taste with a few traits, like genre, mood, and energy.

Intended use: quick music suggestions for experiments and demos.
Not intended use: medical, mental health, hiring, legal, or other high-stakes decisions.
It is also not meant to fully replace real streaming recommendation systems.

---

## 3. How the Model Works  

The model compares each song to a user profile.
It uses genre, mood, energy, tempo, valence, danceability, and acousticness.

If genre or mood matches, the song gets extra points.
For numeric traits, songs get more points when they are close to the user preference.
All points are added into one final score, then songs are ranked from highest to lowest.

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

The model works best for users with clear and simple preferences.
It performs well when genre and mood are specific.
It also does a good job matching overall energy level.

In tests, chill profiles often got lofi or ambient songs near the top.
Upbeat profiles often got pop or dance tracks near the top.
These results matched my expectations.

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


## 8. Future Work  

Ideas for how I would improve the model next:

- Add more songs and broader genres to reduce bias from small data.
- Add a diversity rule so the top results are less repetitive.
- Show simple explanations for each recommendation, like "matched your mood and energy."

---

## 9. Personal Reflection  

I learned that even a simple recommender can feel useful when user preferences are clear. Through this project, I learned that weight choices matter a lot, because small changes can reshuffle rankings greatly. One thing that surprised me was how confident the model can look even with weak or messy inputs. That made me see how easy it is for a system to hide bias behind clean-looking results. This changed how I think about music apps. Now, I'll pay more attention to data quality, diversity, and explanation, not just accuracy. If I were to extend this project, I would definetly add more songs and more adaptable personalization from user feedback when they like and skip songs.
