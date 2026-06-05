# 🏏 ICC Women's T20 World Cup 2026 — ML Predictions

A machine learning model that predicts match outcomes for the ICC Women's T20 World Cup 2026 (England, June–July 2026).

**Model accuracy: 93.5% on held-out 2022 + 2024 World Cups (43/46 correct)**

🌐 **[View the live prediction website →](https://tarak-ram-1908.github.io/wc2026-predictions)**

---

## 🎯 Results

| Tournament | Accuracy | Correct |
|---|---|---|
| 2022 WC (South Africa) | 91.3% | 21/23 |
| 2024 WC (UAE) | 95.7% | 22/23 |
| **Combined** | **93.5%** | **43/46** |

**Predicted champion: 🏆 Australia** (79.4% win probability in the final vs England)

---

## 📁 Repository Structure

```
wc2026-predictions/
├── index.html                    # Live prediction website
├── train_model.py                # Full training pipeline
├── predict_wc2026.py             # Generate predictions
├── cricket_data_loader.py        # Data loading & feature extraction
│
├── data/
│   ├── match_index.csv           # 690 match history
│   ├── elo_ratings.csv           # ELO ratings per team
│   ├── team_ratings.csv          # Phase-by-phase batting/bowling ratings
│   ├── team_stats_raw.csv        # Raw team statistics
│   ├── venue_ratings.csv         # 104 venue profiles
│   ├── venue_familiarity_matrix.csv
│   ├── h2h_win_rate.csv          # Head-to-head win rates (all time)
│   ├── h2h_win_rate_recent.csv   # H2H (recent matches)
│   ├── h2h_win_rate_decay.csv    # H2H with recency decay
│   ├── h2h_match_count.csv
│   ├── bowling_batting_matchups.csv
│   ├── team_bowling_composition.csv
│   ├── team_eco_adjusted.csv
│   └── toss_stats.csv
│
├── model/
│   └── wc2026_model.pkl          # Trained LightGBM model
│
└── output/
    └── wc2026_predictions.csv    # All 30 match predictions
```

---

## 🧠 Model Architecture

### Algorithm
**LightGBM** (Gradient Boosted Decision Trees) with:
- `num_leaves = 15` (shallow trees to prevent overfitting on 644 training rows)
- Early stopping (patience = 30 rounds)
- L1 + L2 regularisation
- Feature fraction = 0.8 (bagging for diversity)

### Feature Groups (63 total)

| Group | Features | Description |
|---|---|---|
| G1 — Strength | 10 | ELO ratings, H2H records, recent form, win rate |
| G2 — Venue | 12 | Venue familiarity, chase win rate, pitch type, toss importance |
| G3 — Phases | 34 | Powerplay/middle/death batting & bowling, spin/pace fit |
| G5 — Matchup/Toss | 7 | Bowling matchup history, toss advantage |

> G4 (Domain: fielding, captaincy, pressure) was **dropped** — it increased log-loss.

### Top Features by Importance
1. `h2h_all_time_a` — historical head-to-head win rate
2. `venue_chase_wr` — venue's historical chase success rate
3. `chase_venue_fit_b` — how well team B suits chasing at this venue
4. `familiarity_diff` — difference in venue familiarity
5. `venue_toss_importance` — how much the toss matters at this venue

### Training / Validation Split
- **Train**: 644 non-WC matches
- **Val 2022**: 23 matches (2022 Women's T20 WC, South Africa) — held out entirely
- **Val 2024**: 23 matches (2024 Women's T20 WC, UAE) — held out entirely

No validation data was ever used during training or hyperparameter tuning.

---

## 🚀 Running the Model

### Requirements
```bash
pip install lightgbm scikit-learn pandas numpy
```

### Train from scratch
```bash
python train_model.py
```

### Generate predictions
```bash
python predict_wc2026.py
```

### Load the saved model
```python
import pickle
obj = pickle.load(open('model/wc2026_model.pkl', 'rb'))
model = obj['model']
features = obj['features']
```

---

## 📊 2026 Predictions Summary

### Group A (Predicted qualifiers: Australia, India)
| Team | Points |
|---|---|
| 🇦🇺 Australia | 10 |
| 🇮🇳 India | 6 |
| 🇿🇦 South Africa | 6 |
| 🇵🇰 Pakistan | 4 |
| 🇧🇩 Bangladesh | 4 |
| 🇳🇱 Netherlands | 0 |

### Group B (Predicted qualifiers: England, New Zealand)
| Team | Points |
|---|---|
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 10 |
| 🇳🇿 New Zealand | 8 |
| 🏳 West Indies | 6 |
| 🇱🇰 Sri Lanka | 4 |
| 🇮🇪 Ireland | 0 |
| 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland | 0 |

### Knockouts
- **SF1 (Jun 30, The Oval)**: Australia 67.8% vs New Zealand 32.2%
- **SF2 (Jul 2, The Oval)**: England 87.5% vs India 12.5%
- **Final (Jul 5, Lord's)**: Australia **79.4%** vs England 20.6%

---

## 🌐 Deploying the Website

The `index.html` is a self-contained single-page website — no build step needed.

### GitHub Pages (recommended)
1. Push this repository to GitHub
2. Go to **Settings → Pages**
3. Set source to **Deploy from a branch → main → / (root)**
4. Your site will be live at `https://tarak-ram-1908.github.io/wc2026-predictions`

### Updating the GitHub link in the website
Edit `index.html` and update line with `id="ghLink"`:
```html
<a href="https://github.com/Tarak-Ram-1908/wc2026-predictions" class="gh-link" id="ghLink">
```

---

## 📝 Notes

- Probabilities are model outputs — real matches involve weather, injuries, form dips, and randomness
- The model does not account for squad changes announced after the training data cutoff
- All predictions are from the perspective of team A (listed first) winning

---

## 📄 License

MIT — use freely, attribute appreciated.