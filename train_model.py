"""
ICC Women's T20 World Cup 2026 — Training Pipeline
====================================================
Run this file to:
  1. Build the feature matrix from all 690 matches
  2. Train LightGBM in feature group layers
  3. Evaluate on 2022 WC + 2024 WC (validation)
  4. Save the best model
  5. Print feature importances

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import warnings, os, pickle
warnings.filterwarnings('ignore')

from cricket_data_loader import CricketData

# ============================================================
# STEP 1 — BUILD FEATURE MATRIX
# ============================================================
# Loop through every match in match_index.csv
# For each match, call build_match_features() → one row
# Stack all rows into a DataFrame

def build_feature_matrix(cd, match_index_path='match_index.csv'):
    """
    Reads match_index.csv and builds one feature row per match.
    Label: 1 if team1 (team_a perspective) won, 0 if team2 won.
    Returns X (features), y (labels), meta (team names, dates, flags).
    """
    print("Building feature matrix...")
    mi = pd.read_csv(match_index_path)

    # Drop matches with no result (rain, ties, abandoned)
    mi = mi[mi['winner'].notna()].copy()
    mi = mi[mi['winner'].isin(mi['team1'].tolist() + mi['team2'].tolist())].copy()
    print(f"  Total matches after cleaning: {len(mi)}")

    rows = []
    meta_rows = []
    skipped = 0

    for _, match in mi.iterrows():
        team_a  = match['team1']
        team_b  = match['team2']
        venue   = match['venue']
        winner  = match['winner']

        # Label: 1 = team_a (team1) won, 0 = team_b (team2) won
        label = 1 if winner == team_a else 0

        # Get toss info if available
        toss_winner   = match.get('toss_winner', None)
        batting_first = match.get('bat_first', None)

        try:
            feats = cd.get_numeric_features(
                team_a, team_b, venue,
                toss_winner=toss_winner,
                batting_first=batting_first
            )
            feats['label'] = label
            rows.append(feats)

            meta_rows.append({
                'match_id':    match.get('match_id', ''),
                'team_a':      team_a,
                'team_b':      team_b,
                'winner':      winner,
                'venue':       venue,
                'date':        match.get('date', ''),
                'event':       match.get('event', ''),
                'is_val_2022': bool(match.get('is_val_2022', False)),
                'is_val_2024': bool(match.get('is_val_2024', False)),
                'label':       label,
            })
        except Exception as e:
            skipped += 1

    print(f"  Rows built: {len(rows)} | Skipped: {skipped}")
    df      = pd.DataFrame(rows)
    meta    = pd.DataFrame(meta_rows)

    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].astype(float)
    y = df['label'].astype(int)

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label distribution: {y.value_counts().to_dict()}")
    return X, y, meta, feature_cols


# ============================================================
# STEP 2 — TRAIN / VALIDATE SPLIT
# ============================================================
# Training: all non-WC matches (644 matches)
# Validation set 1: 2022 WC (23 matches) — SA hosted
# Validation set 2: 2024 WC (23 matches) — Bangladesh hosted
# We NEVER use validation data during training or tuning.

def split_data(X, y, meta):
    """Returns train, val_2022, val_2024 splits."""
    val_2022_mask = meta['is_val_2022'].values
    val_2024_mask = meta['is_val_2024'].values
    train_mask    = ~val_2022_mask & ~val_2024_mask

    X_train = X[train_mask];  y_train = y[train_mask]
    X_v22   = X[val_2022_mask]; y_v22 = y[val_2022_mask]
    X_v24   = X[val_2024_mask]; y_v24 = y[val_2024_mask]
    m_train = meta[train_mask]
    m_v22   = meta[val_2022_mask]
    m_v24   = meta[val_2024_mask]

    print(f"\nData split:")
    print(f"  Train:      {len(X_train)} matches")
    print(f"  Val 2022:   {len(X_v22)} matches")
    print(f"  Val 2024:   {len(X_v24)} matches")
    return (X_train,y_train,m_train), (X_v22,y_v22,m_v22), (X_v24,y_v24,m_v24)


# ============================================================
# STEP 3 — FEATURE GROUPS
# ============================================================
# We train 5 models, each adding one more group.
# The group that reduces log-loss on validation → keep.
# The group that increases log-loss → drop.

FEATURE_GROUPS = {
    'G1_strength': [
        'elo_diff', 'elo_win_prob_a',
        'h2h_blended_a', 'h2h_decay_a', 'h2h_all_time_a', 'h2h_match_count',
        'win_rate_a', 'win_rate_b',
        'recent_form_a', 'recent_form_b',
    ],
    'G2_venue': [
        'familiarity_a', 'familiarity_b', 'familiarity_diff',
        'venue_avg1st', 'venue_chase_wr', 'venue_bat_improve',
        'venue_dew_factor', 'venue_fast_assist', 'venue_spin_assist',
        'venue_bat_assist', 'venue_toss_importance', 'venue_reliability_n',
    ],
    'G3_phases': [
        'bat_overall_a', 'bat_pp_a', 'bat_mid_a', 'bat_death_a',
        'bat_overall_b', 'bat_pp_b', 'bat_mid_b', 'bat_death_b',
        'bowl_overall_a', 'bowl_pp_a', 'bowl_mid_a', 'bowl_death_a',
        'bowl_overall_b', 'bowl_pp_b', 'bowl_mid_b', 'bowl_death_b',
        'fast_bowling_a', 'spin_bowling_a', 'fast_bowling_b', 'spin_bowling_b',
        'venue_bat_pp', 'venue_bat_mid', 'venue_bat_death',
        'venue_bowl_pp', 'venue_bowl_mid', 'venue_bowl_death',
        'fast_venue_fit_a', 'spin_venue_fit_a',
        'fast_venue_fit_b', 'spin_venue_fit_b',
        'chase_venue_fit_a', 'chase_venue_fit_b',
        'chase_skill_a', 'defend_skill_a',
    ],
    'G4_domain': [
        'fielding_a', 'fielding_b',
        'pressure_a', 'pressure_b',
        'homework_a', 'homework_b',
        'captaincy_a', 'captaincy_b',
    ],
    'G5_matchup_toss': [
        'matchup_a_bowls_b', 'matchup_b_bowls_a', 'matchup_net_a',
        'toss_advantage_a', 'toss_pref_field_a',
        'toss_winner_is_a', 'batting_first_is_a',
    ],
}


# ============================================================
# STEP 4 — LIGHTGBM TRAINING FUNCTION
# ============================================================

LGBM_PARAMS = {
    'objective':        'binary',
    'metric':           'binary_logloss',
    'boosting_type':    'gbdt',
    'n_estimators':     300,
    'learning_rate':    0.05,
    'num_leaves':       15,        # Small — prevents overfitting on 644 rows
    'min_child_samples':10,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'lambda_l1':        0.1,
    'lambda_l2':        0.1,
    'verbose':          -1,
    'random_state':     42,
}
# Why these params?
# num_leaves=15:      With only 644 training rows, deep trees overfit fast.
#                     15 leaves means the tree is shallow and generalises better.
# n_estimators=300:   With early stopping (below) this never fully runs anyway.
# feature_fraction:   Each tree only sees 80% of features — reduces correlation between trees.
# lambda_l1/l2:       L1+L2 regularisation — penalises large weights, prevents overfitting.


def train_lgbm(X_train, y_train, X_val, y_val, feature_cols, label='model'):
    """Train one LightGBM model with early stopping. Returns model + scores."""
    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    # Early stopping: stop if val loss doesn't improve for 30 rounds
    model.fit(
        X_train[feature_cols], y_train,
        eval_set=[(X_val[feature_cols], y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
    )

    probs  = model.predict_proba(X_val[feature_cols])[:, 1]
    preds  = (probs >= 0.5).astype(int)
    ll     = log_loss(y_val, probs)
    acc    = accuracy_score(y_val, preds)

    print(f"  [{label}] Features:{len(feature_cols):3d} | "
          f"Accuracy:{acc:.3f} ({int(acc*len(y_val))}/{len(y_val)}) | "
          f"LogLoss:{ll:.4f} | Trees:{model.n_estimators_}")
    return model, probs, ll, acc


# ============================================================
# STEP 5 — FEATURE GROUP LAYERING EXPERIMENT
# ============================================================

def run_group_experiment(X_train, y_train, X_v22, y_v22, X_v24, y_v24):
    """
    Train progressively — add one feature group at a time.
    Keep group if it reduces log-loss on validation, drop if it doesn't.
    Returns the best feature list and best model.
    """
    print("\n" + "="*65)
    print("FEATURE GROUP LAYERING EXPERIMENT")
    print("="*65)
    print("(Adding one group at a time. Keep if val log-loss improves.)\n")

    active_features = []
    best_logloss    = 999.0
    best_features   = []
    best_model      = None
    results         = []

    for group_name, group_feats in FEATURE_GROUPS.items():
        # Only use features that exist in X
        valid_feats = [f for f in group_feats if f in X_train.columns]
        candidate   = active_features + valid_feats

        print(f"Testing: {group_name} (+{len(valid_feats)} features → total {len(candidate)})")

        # Combine val sets for a single layering decision
        X_val_combined = pd.concat([X_v22, X_v24])
        y_val_combined = pd.concat([y_v22, y_v24])

        model, probs, ll, acc = train_lgbm(
            X_train, y_train,
            X_val_combined, y_val_combined,
            candidate, label=group_name
        )

        if ll < best_logloss:
            print(f"    ✅ Improvement: {best_logloss:.4f} → {ll:.4f} — keeping {group_name}")
            active_features = candidate
            best_logloss    = ll
            best_features   = candidate.copy()
            best_model      = model
        else:
            print(f"    ❌ No improvement: {ll:.4f} ≥ {best_logloss:.4f} — dropping {group_name}")

        results.append({
            'group': group_name, 'n_features': len(candidate),
            'logloss': ll, 'accuracy': acc, 'kept': ll < best_logloss
        })
        print()

    print(f"\nBest feature set: {len(best_features)} features | LogLoss: {best_logloss:.4f}")
    return best_model, best_features, pd.DataFrame(results)


# ============================================================
# STEP 6 — FINAL EVALUATION ON BOTH VALIDATION SETS
# ============================================================

def evaluate_final(model, best_features, X_v22, y_v22, m_v22, X_v24, y_v24, m_v24):
    """Evaluate best model on each WC separately. Print match-level predictions."""

    print("\n" + "="*65)
    print("FINAL EVALUATION")
    print("="*65)

    for label, X_val, y_val, meta_val in [
        ('2022 WC (South Africa)', X_v22, y_v22, m_v22),
        ('2024 WC (Bangladesh)',   X_v24, y_v24, m_v24),
    ]:
        probs = model.predict_proba(X_val[best_features])[:, 1]
        preds = (probs >= 0.5).astype(int)
        ll    = log_loss(y_val, probs)
        acc   = accuracy_score(y_val, preds)

        print(f"\n{label}:")
        print(f"  Accuracy: {acc:.3f} ({int(acc*len(y_val))}/{len(y_val)}) | LogLoss: {ll:.4f}")
        print(f"  {'Match':<45} {'Pred':>10} {'Prob':>8} {'Actual':>10} {'✓'}")
        print(f"  {'-'*80}")

        meta_val = meta_val.reset_index(drop=True)
        for i, row in meta_val.iterrows():
            pred_team  = row['team_a'] if preds[i]==1 else row['team_b']
            actual     = row['winner']
            prob_a     = probs[i]
            correct    = '✅' if pred_team == actual else '❌'
            match_str  = f"{row['team_a']} vs {row['team_b']}"
            print(f"  {match_str:<45} {pred_team:>10} {prob_a:>7.1%}   {actual:>10} {correct}")


# ============================================================
# STEP 7 — FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(model, best_features, top_n=20):
    """Print top N most important features."""
    print(f"\n{'='*65}")
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print(f"{'='*65}")
    importance = pd.Series(
        model.feature_importances_,
        index=best_features
    ).sort_values(ascending=False)

    for feat, score in importance.head(top_n).items():
        bar = '█' * int(score / importance.iloc[0] * 30)
        print(f"  {feat:<40} {bar} {score}")
    return importance


# ============================================================
# STEP 8 — SAVE MODEL
# ============================================================

def save_model(model, best_features, path='wc2026_model.pkl'):
    """Save model + feature list so you can load and predict later."""
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'features': best_features}, f)
    print(f"\n✅ Model saved to {path}")
    print(f"   Load with: import pickle; obj=pickle.load(open('{path}','rb'))")
    print(f"   Then: model=obj['model']; features=obj['features']")


# ============================================================
# MAIN — RUN EVERYTHING
# ============================================================

if __name__ == '__main__':
    print("ICC Women's T20 WC 2026 — Training Pipeline")
    print("="*65)

    # Load data
    cd = CricketData()

    # Step 1: Build feature matrix
    X, y, meta, feature_cols = build_feature_matrix(cd)

    # Step 2: Split
    (X_train,y_train,m_train), (X_v22,y_v22,m_v22), (X_v24,y_v24,m_v24) = split_data(X, y, meta)

    # Step 3: Feature group experiment
    best_model, best_features, group_results = run_group_experiment(
        X_train, y_train, X_v22, y_v22, X_v24, y_v24
    )

    print("\nGroup experiment summary:")
    print(group_results.to_string(index=False))

    # Step 4: Final evaluation
    evaluate_final(best_model, best_features, X_v22, y_v22, m_v22, X_v24, y_v24, m_v24)

    # Step 5: Feature importance
    show_feature_importance(best_model, best_features)

    # Step 6: Save
    save_model(best_model, best_features)
