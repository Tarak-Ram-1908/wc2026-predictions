"""
ICC Women's T20 World Cup 2026 — Data Loader v2.1
==================================================
All issues from review fixed. Drop all files in same folder, then:
    from cricket_data_loader import CricketData
    cd = CricketData()
    feats = cd.build_match_features('Australia', 'India', 'Edgbaston, Birmingham')
"""
import pandas as pd, numpy as np, os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

WC_TEAMS = ['Australia','India','England','New Zealand','South Africa','West Indies',
            'Pakistan','Sri Lanka','Bangladesh','Ireland','Scotland','Netherlands']

REGIONS = ['England','Ireland','Scotland','Netherlands','India','Pakistan','Sri Lanka',
           'Bangladesh','Australia','New Zealand','South Africa','West Indies','UAE','Zimbabwe','Other']

class CricketData:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self._load()

    def _p(self, f): return os.path.join(self.data_dir, f)

    def _load(self):
        self.venue_ratings = pd.read_csv(self._p('venue_ratings.csv'))
        self.venue_ratings['venue'] = self.venue_ratings['venue'].str.strip('"').str.strip("'")
        self.venue_ratings.set_index('venue', inplace=True)

        self.familiarity   = pd.read_csv(self._p('venue_familiarity_matrix.csv'), index_col='team')
        self.team_ratings  = pd.read_csv(self._p('team_ratings.csv'), index_col='team')
        self.elo           = pd.read_csv(self._p('elo_ratings.csv'), index_col='team')['elo_rating']
        self.toss_stats    = pd.read_csv(self._p('toss_stats.csv'), index_col='team')
        self.bowling_comp  = pd.read_csv(self._p('team_bowling_composition.csv'), index_col='team')
        self.h2h_all       = pd.read_csv(self._p('h2h_win_rate.csv'), index_col='team')
        self.h2h_recent    = pd.read_csv(self._p('h2h_win_rate_recent.csv'), index_col='team')
        self.h2h_decay     = pd.read_csv(self._p('h2h_win_rate_decay.csv'), index_col='team')
        self.h2h_count     = pd.read_csv(self._p('h2h_match_count.csv'), index_col='team')

        raw = pd.read_csv(self._p('bowling_batting_matchups.csv'))
        self.matchup_table = {}
        for _, r in raw.iterrows():
            key = tuple(r['matchup'].strip("()").replace("'","").split(", "))
            self.matchup_table[key] = r['advantage']

        print(f"✅ CricketData v2.1 loaded | {len(self.venue_ratings)} venues | {len(self.team_ratings)} teams")

    # ── VENUE ────────────────────────────────────────────────
    def _clean(self, name): return str(name).strip('"').strip("'").strip()

    def _infer_region(self, venue):
        vl = venue.lower()
        kw = {
            'England':     ['edgbaston','lords','the oval','headingley','trent bridge','old trafford',
                            'rose bowl','county ground','durham','hove','chelmsford','southampton'],
            'Australia':   ['mcg','scg','waca','gabba','manuka','junction oval','bellerive','adelaide oval'],
            'India':       ['wankhede','brabourne','chinnaswamy','chidambaram','kotla','feroz','eden garden'],
            'New Zealand': ['hagley','basin reserve','eden park','bay oval','saxton','seddon'],
            'South Africa':['newlands','wanderers','kingsmead','buffalo park','boland','senwes','willowmoore'],
            'Bangladesh':  ['sylhet','shere bangla','zahur ahmed','mirpur'],
            'Sri Lanka':   ['premadasa','galle','dambulla','rangiri'],
            'UAE':         ['sharjah','dubai','abu dhabi','icc academy'],
            'Pakistan':    ['karachi','lahore','gaddafi','rawalpindi','multan'],
            'West Indies': ['kensington','providence','sabina','daren sammy','warner park',
                            'sir vivian','queens park','beausejour','arnos vale'],
            'Ireland':     ['ymca','stormont','castle avenue','malahide','pembroke'],
            'Scotland':    ['grange','raeburn','forthill'],
            'Netherlands': ['vra','schootsveld'],
            'Malaysia':    ['kinrara'],
        }
        for region, keywords in kw.items():
            if any(k in vl for k in keywords): return region
        return 'Other'

    def get_venue_features(self, venue_name):
        """Returns venue feature dict. Falls back: exact → region avg → global avg."""
        v = self._clean(venue_name)
        if v in self.venue_ratings.index:
            return {**self.venue_ratings.loc[v].to_dict(), '_fallback': 'exact'}
        region = self._infer_region(v)
        rd = self.venue_ratings[self.venue_ratings['region']==region]
        if len(rd) >= 3:
            fb = rd.mean(numeric_only=True).to_dict()
            fb.update({'venue':v,'region':region,'n':0,'reliability':'low','_fallback':f'region:{region}'})
            return fb
        fb = self.venue_ratings.mean(numeric_only=True).to_dict()
        fb.update({'venue':v,'region':'Other','n':0,'reliability':'low','_fallback':'global_avg'})
        return fb

    def get_venue_region(self, venue_name):
        v = self._clean(venue_name)
        if v in self.venue_ratings.index: return self.venue_ratings.loc[v,'region']
        return self._infer_region(v)

    def get_familiarity(self, team, venue_name):
        region = self.get_venue_region(venue_name)
        if team in self.familiarity.index and region in self.familiarity.columns:
            return float(self.familiarity.loc[team, region])
        return 0.5

    # ── H2H ─────────────────────────────────────────────────
    def get_h2h(self, team_a, team_b, mode='blended'):
        """mode: 'all' | 'recent' | 'decay' | 'blended' (default)"""
        if team_a not in self.h2h_all.index or team_b not in self.h2h_all.columns: return 0.5
        all_t  = float(self.h2h_all.loc[team_a, team_b])
        recent = float(self.h2h_recent.loc[team_a, team_b])
        decay  = float(self.h2h_decay.loc[team_a, team_b])
        count  = float(self.h2h_count.loc[team_a, team_b])
        if mode=='all':    return all_t
        if mode=='recent': return recent
        if mode=='decay':  return decay
        conf = min(1.0, count/15.0)
        return round(conf*decay + (1-conf)*all_t, 4)

    # ── ELO ─────────────────────────────────────────────────
    def get_elo_win_prob(self, team_a, team_b):
        if team_a not in self.elo.index or team_b not in self.elo.index: return 0.5
        diff = float(self.elo[team_a]) - float(self.elo[team_b])
        return round(1/(1+10**(-diff/400)), 4)

    def get_elo_diff(self, team_a, team_b):
        if team_a not in self.elo.index or team_b not in self.elo.index: return 0.0
        return round(float(self.elo[team_a]) - float(self.elo[team_b]), 1)

    # ── MATCHUP ─────────────────────────────────────────────
    def get_matchup_score(self, bowling_team, batting_team):
        """Net bowling advantage from delivery-type vs RHB/LHB matchups. Positive = bowling team advantage."""
        if bowling_team not in self.bowling_comp.index or batting_team not in self.bowling_comp.index:
            return 0.0
        bc = self.bowling_comp.loc[bowling_team]
        rhb = float(self.bowling_comp.loc[batting_team, 'rhb_pct'])
        lhb = 1 - rhb
        # RA pace is the baseline (all teams have it, ~40% of overs)
        score = (self.matchup_table.get(('RA_pace','RHB'),0.05)*rhb +
                 self.matchup_table.get(('RA_pace','LHB'),-0.03)*lhb) * 0.40
        for col, style in [('has_LA_orthodox','LA_orthodox'),('has_RA_offbreak','RA_offbreak'),
                            ('has_LA_pace','LA_pace'),('has_RA_legbreak','RA_legbreak')]:
            if bc.get(col, False):
                net = (self.matchup_table.get((style,'RHB'),0.0)*rhb +
                       self.matchup_table.get((style,'LHB'),0.0)*lhb)
                score += net * 0.15
        return round(score, 4)

    # ── TEAM FEATURE ────────────────────────────────────────
    def tf(self, team, feat):
        if team in self.team_ratings.index and feat in self.team_ratings.columns:
            return float(self.team_ratings.loc[team, feat])
        return 0.5

    # ── MAIN FEATURE BUILDER ────────────────────────────────
    def build_match_features(self, team_a, team_b, venue_name,
                              batting_first=None, toss_winner=None):
        """
        Returns dict of ~60 numeric features for one match.
        Keys starting with '_' are metadata — strip before passing to model.
        """
        vf     = self.get_venue_features(venue_name)
        region = self.get_venue_region(venue_name)

        fam_a = self.get_familiarity(team_a, venue_name)
        fam_b = self.get_familiarity(team_b, venue_name)

        f_assist = float(vf.get('fast_assist', 0.55))
        s_assist = float(vf.get('spin_assist', 0.55))
        c_wr     = float(vf.get('chase_wr', 0.5))
        t_imp    = float(vf.get('toss_importance', 0.50))

        toss_adv_a = float(self.toss_stats.loc[team_a,'toss_advantage']) \
                     if team_a in self.toss_stats.index else 0.0
        toss_pref_a = int(self.toss_stats.loc[team_a,'preferred_choice_enc']) \
                      if team_a in self.toss_stats.index else 1

        return {
            # H2H
            'h2h_blended_a':    self.get_h2h(team_a, team_b, 'blended'),
            'h2h_decay_a':      self.get_h2h(team_a, team_b, 'decay'),
            'h2h_all_time_a':   self.get_h2h(team_a, team_b, 'all'),
            'h2h_match_count':  float(self.h2h_count.loc[team_a,team_b])
                                if team_a in self.h2h_count.index else 0.0,
            # ELO
            'elo_diff':         self.get_elo_diff(team_a, team_b),
            'elo_win_prob_a':   self.get_elo_win_prob(team_a, team_b),
            # Familiarity
            'familiarity_a':    fam_a,
            'familiarity_b':    fam_b,
            'familiarity_diff': round(fam_a - fam_b, 3),
            # Venue conditions
            'venue_avg1st':         float(vf.get('avg1st', 135)),
            'venue_chase_wr':       c_wr,
            'venue_bat_improve':    float(vf.get('bat_improve', 0.48)),
            'venue_dew_factor':     float(vf.get('dew_factor', 0.40)),
            'venue_fast_assist':    f_assist,
            'venue_spin_assist':    s_assist,
            'venue_bat_assist':     float(vf.get('bat_assist', 0.55)),
            'venue_toss_importance':t_imp,
            'venue_reliability_n':  float(vf.get('n', 0)),
            # Venue phase importance
            'venue_bat_pp':    float(vf.get('bat_pp', 0.55)),
            'venue_bat_mid':   float(vf.get('bat_mid', 0.55)),
            'venue_bat_death': float(vf.get('bat_death', 0.55)),
            'venue_bowl_pp':   float(vf.get('bowl_pp', 0.55)),
            'venue_bowl_mid':  float(vf.get('bowl_mid', 0.55)),
            'venue_bowl_death':float(vf.get('bowl_death', 0.55)),
            # Batting
            'bat_overall_a': self.tf(team_a,'bat_overall'),
            'bat_pp_a':      self.tf(team_a,'bat_pp'),
            'bat_mid_a':     self.tf(team_a,'bat_mid'),
            'bat_death_a':   self.tf(team_a,'bat_death'),
            'bat_overall_b': self.tf(team_b,'bat_overall'),
            'bat_pp_b':      self.tf(team_b,'bat_pp'),
            'bat_mid_b':     self.tf(team_b,'bat_mid'),
            'bat_death_b':   self.tf(team_b,'bat_death'),
            # Bowling
            'bowl_overall_a': self.tf(team_a,'bowl_overall'),
            'bowl_pp_a':      self.tf(team_a,'bowl_pp'),
            'bowl_mid_a':     self.tf(team_a,'bowl_mid'),
            'bowl_death_a':   self.tf(team_a,'bowl_death'),
            'fast_bowling_a': self.tf(team_a,'fast_bowling'),
            'spin_bowling_a': self.tf(team_a,'spin_bowling'),
            'bowl_overall_b': self.tf(team_b,'bowl_overall'),
            'bowl_pp_b':      self.tf(team_b,'bowl_pp'),
            'bowl_mid_b':     self.tf(team_b,'bowl_mid'),
            'bowl_death_b':   self.tf(team_b,'bowl_death'),
            'fast_bowling_b': self.tf(team_b,'fast_bowling'),
            'spin_bowling_b': self.tf(team_b,'spin_bowling'),
            # Style × venue interaction
            'fast_venue_fit_a':  self.tf(team_a,'fast_bowling') * f_assist,
            'spin_venue_fit_a':  self.tf(team_a,'spin_bowling') * s_assist,
            'fast_venue_fit_b':  self.tf(team_b,'fast_bowling') * f_assist,
            'spin_venue_fit_b':  self.tf(team_b,'spin_bowling') * s_assist,
            'chase_venue_fit_a': self.tf(team_a,'chase_skill') * c_wr,
            'chase_venue_fit_b': self.tf(team_b,'chase_skill') * c_wr,
            # Matchup
            'matchup_a_bowls_b': self.get_matchup_score(team_a, team_b),
            'matchup_b_bowls_a': self.get_matchup_score(team_b, team_a),
            'matchup_net_a':     round(self.get_matchup_score(team_a,team_b) - self.get_matchup_score(team_b,team_a), 4),
            # Team quality
            'win_rate_a':    self.tf(team_a,'win_rate'),
            'win_rate_b':    self.tf(team_b,'win_rate'),
            'recent_form_a': self.tf(team_a,'recent_form'),
            'recent_form_b': self.tf(team_b,'recent_form'),
            'chase_skill_a': self.tf(team_a,'chase_skill'),
            'defend_skill_a':self.tf(team_a,'defend_skill'),
            # Domain knowledge
            'fielding_a':  self.tf(team_a,'fielding'),
            'fielding_b':  self.tf(team_b,'fielding'),
            'pressure_a':  self.tf(team_a,'pressure_handling'),
            'pressure_b':  self.tf(team_b,'pressure_handling'),
            'homework_a':  self.tf(team_a,'homework'),
            'homework_b':  self.tf(team_b,'homework'),
            'captaincy_a': self.tf(team_a,'captaincy'),
            'captaincy_b': self.tf(team_b,'captaincy'),
            # Toss
            'toss_advantage_a':     toss_adv_a,
            'toss_pref_field_a':    toss_pref_a,
            'venue_toss_importance':t_imp,
            'toss_winner_is_a':     int(toss_winner==team_a) if toss_winner else 0,
            'batting_first_is_a':   int(batting_first==team_a) if batting_first else 0,
            # Metadata
            '_team_a':  team_a, '_team_b': team_b,
            '_venue':   venue_name, '_region': region,
            '_reliable':vf.get('reliability','low'), '_fallback':vf.get('_fallback','exact'),
        }

    def get_numeric_features(self, team_a, team_b, venue_name, **kw):
        """Same as build_match_features but strips metadata. Ready for model.predict()."""
        return {k:v for k,v in self.build_match_features(team_a,team_b,venue_name,**kw).items()
                if not k.startswith('_')}

    def summary(self):
        n_feats = len(self.get_numeric_features('Australia','India','Edgbaston, Birmingham'))
        print(f"Venues:{len(self.venue_ratings)}  Teams:{len(self.team_ratings)}  "
              f"Features per match:{n_feats}")
        print("ELO top 5:", self.elo.nlargest(5).to_dict())


if __name__ == '__main__':
    cd = CricketData()
    cd.summary()
    print("\nSample: Australia vs India at Edgbaston")
    for k,v in sorted(cd.get_numeric_features('Australia','India','Edgbaston, Birmingham').items()):
        print(f"  {k:35s}: {v}")
