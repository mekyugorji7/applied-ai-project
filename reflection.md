# Pairwise Profile Output Reflection

Below are pair-by-pair comparisons of profile outputs from the current model. Each comment explains what changed in the top results and why that behavior is expected from the preference settings.

## Core Profile Pairs

- **`USER_PREFS_CHILL_LOFI` vs `USER_PREFS_UPBEAT_POP`**: The chill profile surfaces `Midnight Coding` and `Library Rain`, while the upbeat profile shifts to `Sunrise City`, `Gym Hero`, and `Rooftop Lights`. This makes sense because the upbeat profile asks for much higher energy, higher danceability, and lower acousticness.

- **`USER_PREFS_CHILL_LOFI` vs `USER_PREFS_SOFT_AMBIENT`**: Both include calmer songs, but ambient pushes `Spacewalk Thoughts` to the top and includes `Willow & Ember` / `Low Tide Blues`, while chill-lofi keeps stronger lofi matches (`Midnight Coding`, `Focus Flow`) higher. This fits because ambient targets even lower energy and slower tempo than chill-lofi.

- **`USER_PREFS_UPBEAT_POP` vs `USER_PREFS_SOFT_AMBIENT`**: The upbeat profile favors high-energy pop-adjacent tracks (`Sunrise City`, `Gym Hero`), while soft ambient favors low-energy, high-acoustic songs (`Spacewalk Thoughts`, `Willow & Ember`). The split is expected because their targets are opposite on energy, tempo, and acoustic preference.

## Edge Profile Pair Comments

- **`EDGE_OUT_OF_RANGE` vs `EDGE_CONFLICTING_AFFECT`**: Out-of-range keeps `Boneforge March` first mainly from exact genre/mood plus non-acoustic alignment, while conflicting-affect elevates high-energy songs like `Storm Runner`. This makes sense because invalid numeric targets weaken continuous matching and leave categorical matches to dominate.

- **`EDGE_UNKNOWN_LABELS` vs `EDGE_SPARSE`**: Unknown labels still produce varied top songs (`Dust Road Echoes`, `Willow & Ember`) due to numeric similarity terms, while sparse mostly follows energy closeness (`Midnight Coding`, `Focus Flow`). This is expected because unknown categories remove exact-match bonuses and leave continuous features as the deciding signal.

- **`EDGE_BOOL_STRING_TRAP` vs `USER_PREFS_CHILL_LOFI`**: These outputs are almost identical, which is informative. The string value `"False"` is truthy in Python, so the model treats it similarly to `likes_acoustic=True`, preserving a chill/acoustic-leaning list.

- **`EDGE_NONE_ACOUSTIC` vs `USER_PREFS_CHILL_LOFI`**: `EDGE_NONE_ACOUSTIC` shifts `Coffee Shop Stories` and `Willow & Ember` upward and changes the order of lofi tracks. This is reasonable under current logic because `likes_acoustic=None` falls into the non-acoustic branch, changing acoustic scoring behavior in an unintuitive way.
