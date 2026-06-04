"""
ICC Women's T20 World Cup 2026 — Match Predictions
====================================================
Generates win probabilities for all 30 group stage matches.
Run: python predict_wc2026.py
"""

import pickle
import pandas as pd
import sys
sys.path.insert(0, '.')
from cricket_data_loader import CricketData

# ── Load model ──────────────────────────────────────────────
with open('wc2026_model.pkl', 'rb') as f:
    obj = pickle.load(f)
model    = obj['model']
features = obj['features']

cd = CricketData()

# ── Venue name mapping ───────────────────────────────────────
# Maps schedule venue names → closest match in venue_ratings.csv
# Venues with no exact match fall back to England regional average
VENUE_MAP = {
    'Edgbaston, Birmingham':    'Edgbaston, Birmingham',
    'Old Trafford, Manchester': 'County Ground',        # England regional fallback
    'Rose Bowl, Southampton':   'County Ground',        # England regional fallback
    'Headingley, Leeds':        'County Ground',        # England regional fallback
    'County Ground, Bristol':   'County Ground, Bristol',
    'The Oval, London':         'Kennington Oval, London',
    "Lord's, London":           'County Ground',        # England regional fallback
}

# ── Full 2026 WC Group Stage Schedule ───────────────────────
# Format: (match_no, date, team_a, team_b, venue, group)
SCHEDULE = [
    # Group B
    ( 1, 'Jun 12', 'England',      'Sri Lanka',    'Edgbaston, Birmingham',    'B'),
    # Group A
    ( 2, 'Jun 13', 'Scotland',     'Ireland',      'Old Trafford, Manchester', 'A'),
    ( 3, 'Jun 13', 'Australia',    'South Africa', 'Old Trafford, Manchester', 'A'),
    # Group B
    ( 4, 'Jun 13', 'West Indies',  'New Zealand',  'Rose Bowl, Southampton',   'B'),
    # Group A
    ( 5, 'Jun 14', 'Bangladesh',   'Netherlands',  'Edgbaston, Birmingham',    'A'),
    ( 6, 'Jun 14', 'India',        'Pakistan',     'Edgbaston, Birmingham',    'A'),
    # Group B
    ( 7, 'Jun 16', 'New Zealand',  'Sri Lanka',    'Rose Bowl, Southampton',   'B'),
    ( 8, 'Jun 16', 'England',      'Ireland',      'Rose Bowl, Southampton',   'B'),
    # Group A
    ( 9, 'Jun 17', 'Australia',    'Bangladesh',   'Headingley, Leeds',        'A'),
    (10, 'Jun 17', 'India',        'Netherlands',  'Headingley, Leeds',        'A'),
    (11, 'Jun 17', 'South Africa', 'Pakistan',     'Edgbaston, Birmingham',    'A'),
    # Group B
    (12, 'Jun 18', 'West Indies',  'Scotland',     'Headingley, Leeds',        'B'),
    (13, 'Jun 19', 'New Zealand',  'Ireland',      'Rose Bowl, Southampton',   'B'),
    # Group A
    (14, 'Jun 20', 'Australia',    'Netherlands',  'Rose Bowl, Southampton',   'A'),
    (15, 'Jun 20', 'Pakistan',     'Bangladesh',   'Rose Bowl, Southampton',   'A'),
    # Group B
    (16, 'Jun 20', 'England',      'Scotland',     'Headingley, Leeds',        'B'),
    (17, 'Jun 21', 'West Indies',  'Sri Lanka',    'County Ground, Bristol',   'B'),
    # Group A
    (18, 'Jun 21', 'South Africa', 'India',        'Old Trafford, Manchester', 'A'),
    # Group B
    (19, 'Jun 23', 'New Zealand',  'Scotland',     'County Ground, Bristol',   'B'),
    (20, 'Jun 23', 'Sri Lanka',    'Ireland',      'County Ground, Bristol',   'B'),
    # Group A
    (21, 'Jun 23', 'Australia',    'Pakistan',     'Headingley, Leeds',        'A'),
    # Group B
    (22, 'Jun 24', 'England',      'West Indies',  "Lord's, London",           'B'),
    # Group A
    (23, 'Jun 25', 'India',        'Bangladesh',   'Old Trafford, Manchester', 'A'),
    (24, 'Jun 25', 'South Africa', 'Netherlands',  'County Ground, Bristol',   'A'),
    # Group B
    (25, 'Jun 26', 'Sri Lanka',    'Scotland',     'Old Trafford, Manchester', 'B'),
    # Group A
    (26, 'Jun 27', 'Pakistan',     'Netherlands',  'County Ground, Bristol',   'A'),
    # Group B
    (27, 'Jun 27', 'West Indies',  'Ireland',      'County Ground, Bristol',   'B'),
    (28, 'Jun 27', 'England',      'New Zealand',  'The Oval, London',         'B'),
    # Group A
    (29, 'Jun 28', 'South Africa', 'Bangladesh',   "Lord's, London",           'A'),
    (30, 'Jun 28', 'Australia',    'India',        "Lord's, London",           'A'),
]

# ── Predict all matches ──────────────────────────────────────
def predict_match(team_a, team_b, venue_name):
    mapped_venue = VENUE_MAP.get(venue_name, venue_name)
    feats = cd.get_numeric_features(team_a, team_b, mapped_venue)
    feat_vals = [feats.get(f, 0.5) for f in features]
    prob_a = model.predict_proba([feat_vals])[0][1]
    prob_b = 1 - prob_a
    winner = team_a if prob_a >= 0.5 else team_b
    win_prob = max(prob_a, prob_b)
    return winner, prob_a, prob_b, win_prob

# ── Run predictions and display ──────────────────────────────
print("=" * 72)
print("ICC WOMEN'S T20 WORLD CUP 2026 — MATCH PREDICTIONS")
print("=" * 72)

results = []
group_a_points = {t: 0 for t in ['Australia','India','South Africa','Pakistan','Bangladesh','Netherlands']}
group_b_points = {t: 0 for t in ['England','New Zealand','West Indies','Sri Lanka','Ireland','Scotland']}

current_date = ''
for match_no, date, team_a, team_b, venue, group in SCHEDULE:
    if date != current_date:
        print(f"\n{'─'*72}")
        print(f"  {date}")
        print(f"{'─'*72}")
        current_date = date

    winner, prob_a, prob_b, win_prob = predict_match(team_a, team_b, venue)
    loser = team_b if winner == team_a else team_a

    conf = "HIGH" if win_prob >= 0.75 else "MED" if win_prob >= 0.60 else "LOW"
    conf_colors = {"HIGH": "★★★", "MED": "★★ ", "LOW": "★  "}

    print(f"  M{match_no:02d} | Grp {group} | {team_a:<14} vs {team_b:<14} | "
          f"{venue.split(',')[0]:<22}")
    print(f"       → {winner:<14} wins  |  "
          f"{team_a}: {prob_a:.1%}  {team_b}: {prob_b:.1%}  "
          f"| Conf: {conf_colors[conf]} {conf}")

    # Track points
    points = group_a_points if group == 'A' else group_b_points
    if winner in points:
        points[winner] += 2

    results.append({
        'match': match_no, 'date': date, 'group': group,
        'team_a': team_a, 'team_b': team_b, 'venue': venue,
        'predicted_winner': winner, 'loser': loser,
        'prob_a': round(prob_a, 3), 'prob_b': round(prob_b, 3),
        'win_prob': round(win_prob, 3), 'confidence': conf,
    })

# ── Group standings ──────────────────────────────────────────
print("\n" + "=" * 72)
print("PREDICTED GROUP STANDINGS")
print("=" * 72)

for grp_name, points_dict in [('GROUP A', group_a_points), ('GROUP B', group_b_points)]:
    print(f"\n{grp_name}:")
    print(f"  {'Team':<20} {'Points':>8} {'Predicted'}")
    print(f"  {'─'*45}")
    sorted_teams = sorted(points_dict.items(), key=lambda x: -x[1])
    for i, (team, pts) in enumerate(sorted_teams):
        qualifier = " ✅ QUALIFIES" if i < 2 else ""
        print(f"  {team:<20} {pts:>6} pts{qualifier}")

# ── Predicted knockouts ──────────────────────────────────────
print("\n" + "=" * 72)
print("PREDICTED KNOCKOUT STAGE")
print("=" * 72)

grp_a_sorted = sorted(group_a_points.items(), key=lambda x: -x[1])
grp_b_sorted = sorted(group_b_points.items(), key=lambda x: -x[1])
sf1_a, sf1_b = grp_a_sorted[0][0], grp_b_sorted[1][0]
sf2_a, sf2_b = grp_b_sorted[0][0], grp_a_sorted[1][0]

sf1_winner, sf1_pa, sf1_pb, _ = predict_match(sf1_a, sf1_b, 'The Oval, London')
sf2_winner, sf2_pa, sf2_pb, _ = predict_match(sf2_a, sf2_b, 'The Oval, London')
fin_winner, fin_pa, fin_pb, _ = predict_match(sf1_winner, sf2_winner, "Lord's, London")

print(f"\n  Semi-Final 1 (The Oval, Jun 30):")
print(f"    {sf1_a} ({sf1_pa:.1%}) vs {sf1_b} ({sf1_pb:.1%})")
print(f"    → Predicted winner: {sf1_winner}")

print(f"\n  Semi-Final 2 (The Oval, Jul 2):")
print(f"    {sf2_a} ({sf2_pa:.1%}) vs {sf2_b} ({sf2_pb:.1%})")
print(f"    → Predicted winner: {sf2_winner}")

print(f"\n  FINAL (Lord's, Jul 5):")
print(f"    {sf1_winner} ({fin_pa:.1%}) vs {sf2_winner} ({fin_pb:.1%})")
print(f"    → 🏆 PREDICTED CHAMPION: {fin_winner}")

# ── Save predictions ─────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv('wc2026_predictions.csv', index=False)
print(f"\n✅ All predictions saved to wc2026_predictions.csv")
print(f"   Total matches predicted: {len(results)}")
print(f"   High confidence: {len([r for r in results if r['confidence']=='HIGH'])}")
print(f"   Medium confidence: {len([r for r in results if r['confidence']=='MED'])}")
print(f"   Low confidence: {len([r for r in results if r['confidence']=='LOW'])}")
