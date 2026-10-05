"""FPL engine - the only engine code in this project.

Both notebooks import this module: the weekly assistant runs it against the live
FPL API, the backtest runs it against the vaastav archive. There is no second
copy. If you change how points are projected, transfers are planned or chips are
timed, you change it here and both sides move together.

Settings come in two halves:

    DEFAULTS  the model, the planner and the chip rules - what the backtest tunes.
              A backtest winner can be pasted straight in.
    USER      your team id, free transfers, injury overrides, locks and excludes
              - what you edit each week.

`resolve_cfg()` merges them. Nothing in this module keeps mutable state at module
level: `build_projections` returns a `Projections` object carrying its own
per-player breakdown, so two sets of projections can be alive at once (the
backtest holds all 38 gameweeks simultaneously).
"""
import math
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

try:
    import pulp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pulp"])
    import pulp

API = "https://fantasy.premierleague.com/api/"
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POS_KEY = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
DC_THRESHOLD = {2: 10, 3: 12, 4: 12}
CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit",
              "bboost": "Bench Boost", "3xc": "Triple Captain"}
COMPONENTS = ["Playing time", "Goals", "Assists", "Clean sheet", "Goals conceded",
              "Saves", "Def. contributions", "Bonus", "Yellow cards"]
STAT_KEYS = ["expected_goals", "goals_scored", "expected_assists", "assists",
             "bonus", "saves", "defensive_contribution", "yellow_cards"]
SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}          # 15-man squad by position
XI_LIMITS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}   # legal starting XI
MIN_IN_XI = {1: 1, 2: 3, 3: 2, 4: 1}            # autosubs must respect these


# ---------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------
# What the backtest tunes. These are the live assistant's values - when the
# engines were merged the backtest's copy had drifted (PLAN_WEEKS 6, LONG_VIEW 18,
# DECAY 0.75, MAX_TRANSFERS 3, XG_WEIGHT 0.9, PRIOR_MINUTES 270, MINUTES_GAMES 10),
# so it was tuning a different program than the one being run. The assistant's
# values won.
DEFAULTS = {
    # ---- minutes model ----
    # Weight each player's recent games, newest first. Each game back counts this
    # much of the one after it (0.6 = last game 1, the one before 0.6, then 0.36...).
    # None = old method (season starts / team games, which punishes late signings).
    "MINUTES_DECAY": 0.6,           # backtest-tuned (0.5-0.6 best)
    "MINUTES_GAMES": 5,             # how many recent games to look at
    # Ignore games before a player's first appearance this season. Backtest: slightly
    # worse overall (over-trusts players with 1-2 appearances), so off. Use
    # START_OVERRIDES instead.
    "SKIP_BEFORE_DEBUT": False,

    # ---- planning ----
    # Three different horizons, previously conflated into one:
    #   PLAN_WEEKS        weeks of transfers solved together
    #   LONG_VIEW         weeks used for a squad's long-term value (the tail)
    #   PROJECTION_WEEKS  how far points are projected at all. Chips need more than
    #                     LONG_VIEW: a wildcard in week g is valued over the LONG_VIEW
    #                     weeks from g, so every week of its window needs projections
    #                     LONG_VIEW past it. At 24 a GW17 wildcard seen from GW2 was
    #                     valued over 9 weeks and a GW3 one over 19, so later weeks
    #                     lost by construction and the rule always played early.
    #                     38 = to the end of the season; the view then only shortens
    #                     when the season itself is running out.
    "PROJECTION_WEEKS": 38,
    "PLAN_WEEKS": 5,                # weeks of transfers planned together
    "LONG_VIEW": 19,                # weeks of projections used for long-term value
    "DECAY": 0.8,                   # later weeks count a bit less (uncertainty)
    "TAIL_WEIGHT": 0.5,             # how much weeks beyond the plan count
    "MAX_TRANSFERS": 11,            # most transfers in any one week
    "HIT_COST": 6,
    "FT_END_VALUE": 1.5,            # value of each free transfer left at plan end
    "BENCH_WEIGHT": 0.1,

    # ---- model ----
    "XG_WEIGHT": 0.85,              # 85% xG / 15% actual output (backtest-tuned)
    # Every player's per-90 stats are steadied with the position average, counted as
    # this many minutes (stops tiny samples running wild):
    "PRIOR_MINUTES": 90,
    # Last season, split into its two separate jobs:
    "LAST_SEASON_STATS": True,      # blend last season's per-90 stats (xG, xA, bonus...)
    "LAST_SEASON_MINUTES": 540,     #   ...counted as this many minutes
    # Use last season's starts to predict who plays. Only reached when a player has
    # no games yet this season (the recency-weighted model needs at least one), so
    # in practice it is GW1 - where without it every player projects 0 minutes.
    "LAST_SEASON_STARTS": True,
    "LAST_SEASON_STARTS_GAMES": 3,  #   ...counted as this many games (early season only)
    # Per stat (last season, position average) weights in minutes, overriding the
    # two above. Fitted by signal_noise.py: for each stat, the weights that best
    # predict the rest of the season, scored on held-out seasons. xG and xA keep
    # the defaults (already near-optimal); the noisy stats lean much harder on the
    # past - a run of goals or bonus over a few weeks is mostly luck. A key with
    # "|GK" or "|OUT" applies to keepers or outfield players only.
    "STAT_PRIOR_MINUTES": {
        "goals_scored": [2700, 900], "assists": [2700, 1350],
        "bonus|OUT": [1800, 900], "bonus|GK": [5400, 6750],
        "saves": [900, 90], "yellow_cards": [2700, 1350],
    },
    # Goalkeepers get their own steadying (saves & bonus are very noisy).
    # None = use PRIOR_MINUTES like everyone else.
    "GK_PRIOR_MINUTES": 360,
    "GK_PRIOR_SCALE": 0.6,          # keeper position average used at this share
    # Team attack/defence blend FPL's pre-season strength rating, counted as this
    # many games:
    "TEAM_PRIOR_GAMES": 8,
    # Team attack/defence: share from xG (rest from actual goals). None = XG_WEIGHT.
    "TEAM_XG_WEIGHT": None,
    # A keeper who makes lots of saves is facing lots of shots: treat his save rate as
    # extra evidence his defence is leaky (0 = off). Adjusts clean-sheet odds.
    "SAVES_LEAK": 0.5,              # backtest-tuned with "shots" mode
    # "shots" = saves + goals conceded (shots on target faced - correct),
    # "saves" = saves only (original version: rewards keepers who concede).
    "SAVES_LEAK_MODE": "shots",
    # False = saves/3 (backtest: predicts keepers better overall).
    # True = FPL's exact "per completed 3 saves" rule.
    "EXACT_SAVES": False,
    "PRICE_CHANGES": True,          # value buying risers / selling fallers
    "POINTS_PER_TENTH": 0.5,        # value of £0.1m team value, in points

    # ---- chips ----
    # Play a chip when this week ranks in the top share of the weeks left in its
    # window. The bar tightens on its own as the window runs out and reaches
    # certainty in the last week, so no absolute point minimums are needed.
    # Backtest-tuned per chip.
    "CHIP_TOP_PCT": {"3xc": 0.15, "bboost": 0.40, "freehit": 0.05, "wildcard": 0.15},

    # ---- solver ----
    "SOLVER_SECONDS": 60,           # time limit per optimisation
    "PATH_SOLVER_SECONDS": 5,       # ...per week of the squad forecast (many solves)
    "FORECAST_POOL_PER_POS": 20,    # smaller player pool keeps the forecast quick
}

# What you edit each week.
USER = {
    # Your FPL team ID (the number in the URL of your "Points" page, e.g.
    # fantasy.premierleague.com/entry/1234567/event/4 -> 1234567). Squad, bank,
    # selling prices and remaining chips load automatically.
    "TEAM_ID": 7399216,

    "FREE_TRANSFERS": 1,       # the one thing FPL doesn't publish - update weekly

    # Only needed if TEAM_ID is None. Ambiguous names: "Gakpo|LIV".
    "MY_SQUAD": None,
    "BANK": None,              # ignored if TEAM_ID set
    # Chips left: "wildcard", "freehit", "bboost", "3xc".
    # None = detect automatically from TEAM_ID.
    "CHIPS_AVAILABLE": None,
    # Selling prices are calculated automatically from TEAM_ID. Only fill in to
    # override, or when using MY_SQUAD.
    "SELLING_PRICES": {},

    # Injury news the model can't read: {player: {gameweek: chance 0-1}}
    "AVAILABILITY_OVERRIDES": {},
    # How often you expect a player to start (0-1), e.g. a new signing:
    "START_OVERRIDES": {},
    "LOCK": [],                # never sell
    "EXCLUDE": [],             # never buy
}


def resolve_cfg(*overrides, **kw):
    """DEFAULTS <- USER <- each override dict, in order <- keyword arguments.

    The backtest passes a complete config as an override so that no setting is
    left falling through to a default; the assistant just calls resolve_cfg().
    """
    cfg = dict(DEFAULTS)
    cfg.update(USER)
    for o in overrides:
        if o:
            cfg.update(o)
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------
def fetch(path):
    r = requests.get(API + path, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    return r.json()


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().replace("'", "").replace(".", "").strip()


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def pois_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def pois_at_least(t, mu):
    return 0.0 if mu <= 0 else max(0.0, 1 - sum(pois_pmf(k, mu) for k in range(t)))


def exp_per_three(mu):
    """Expected FPL save points: 1 point per COMPLETED set of 3 saves, E[floor(X/3)]."""
    if mu <= 0:
        return 0.0
    return sum(pois_at_least(3 * j, mu) for j in range(1, 12))


def exp_conceded_penalty(lam):
    return sum((k // 2) * pois_pmf(k, lam) for k in range(16))


# ---------------------------------------------------------------------
# 3. TEAM STRENGTH
# ---------------------------------------------------------------------
def team_strength(bs, fixtures, cfg):
    teams = {t["id"]: t for t in bs["teams"]}
    played = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for fx in fixtures:
        if fx.get("finished") and fx.get("team_h_score") is not None:
            h, a = fx["team_h"], fx["team_a"]
            played[h] += 1; played[a] += 1
            gf[h] += fx["team_h_score"]; ga[h] += fx["team_a_score"]
            gf[a] += fx["team_a_score"]; ga[a] += fx["team_h_score"]
    xgf = {t: 0.0 for t in teams}
    best_min = {t: (0, 0.0) for t in teams}
    for e in bs["elements"]:
        t = e["team"]
        xgf[t] += f(e["expected_goals"])
        if e["minutes"] > best_min[t][0]:
            best_min[t] = (e["minutes"], f(e["expected_goals_conceded"]))
    w = cfg.get("TEAM_XG_WEIGHT")
    w = cfg["XG_WEIGHT"] if w is None else w
    att_obs, def_obs = {}, {}
    for t in teams:
        n = played[t]
        if not n:
            continue
        att_obs[t] = w * xgf[t] / n + (1 - w) * gf[t] / n
        mins, xgc = best_min[t]
        def_obs[t] = w * (xgc / mins * 90 if mins else ga[t] / n) + (1 - w) * ga[t] / n
    avg = max(sum(att_obs.values()) / len(att_obs), 0.5) if att_obs else 1.35
    att, dfn, detail = {}, {}, {}
    for t, info in teams.items():
        s = (info.get("strength_overall_home", 3) + info.get("strength_overall_away", 3)) / 2 or 3
        pa, pd_ = avg * (1 + 0.15 * (s - 3)), avg * (1 - 0.15 * (s - 3))
        k = played[t] / (played[t] + cfg.get("TEAM_PRIOR_GAMES", 6))
        att[t] = k * att_obs.get(t, pa) + (1 - k) * pa
        dfn[t] = k * def_obs.get(t, pd_) + (1 - k) * pd_
        n = played[t]
        mins, xgc = best_min[t]
        detail[t] = {"played": n, "ga_pg": ga[t] / n if n else None,
                     "xga_pg": (xgc / mins * 90) if mins else None,
                     "prior": pd_ / avg, "data_share": k, "leak": 1.0, "keeper_busy": None}
    leak = cfg.get("SAVES_LEAK", 0.0)
    if leak:
        gk = {}
        use_shots = cfg.get("SAVES_LEAK_MODE", "shots") == "shots"
        for e in bs["elements"]:                    # each team's most-used keeper
            if e["element_type"] == 1 and e["minutes"] >= 180 and e["minutes"] > gk.get(e["team"], (0, 0))[0]:
                busy = f(e.get("saves", 0)) + (f(e.get("goals_conceded", 0)) if use_shots else 0)
                gk[e["team"]] = (e["minutes"], busy / e["minutes"] * 90)
        if gk:
            mean_sv = sum(v[1] for v in gk.values()) / len(gk)
            for t in gk:
                if mean_sv > 0:
                    mult = (gk[t][1] / mean_sv) ** leak
                    dfn[t] *= mult
                    detail[t]["leak"], detail[t]["keeper_busy"] = mult, gk[t][1]
    return {"avg": avg, "att": {t: att[t] / avg for t in teams},
            "def": {t: dfn[t] / avg for t in teams}, "played": played, "detail": detail,
            "short": {t: teams[t]["short_name"] for t in teams}}


# ---------------------------------------------------------------------
# 4. LAST-SEASON DATA  (fetched only for relevant players, in parallel)
# ---------------------------------------------------------------------
def fetch_player_pages(ids):
    """One request per player gives last season's totals AND this season's game-by-game record."""
    def one(pid):
        try:
            d = fetch(f"element-summary/{pid}/")
            past = d.get("history_past", [])
            games = sorted(d.get("history", []), key=lambda g: (g.get("round", 0), g.get("kickoff_time", "")))
            recent = [{"minutes": g.get("minutes", 0), "starts": g.get("starts", 0)} for g in games]
            return pid, (past[-1] if past else None), recent
        except Exception:
            return pid, None, None
    with ThreadPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(one, ids))
    past = {pid: s for pid, s, _ in res if s and s.get("minutes", 0) >= 600}
    recent = {pid: r for pid, _, r in res if r}
    return past, recent


# ---------------------------------------------------------------------
# 5. PLAYER PROJECTIONS
# ---------------------------------------------------------------------
@dataclass
class Projections:
    """Everything one `build_projections` call produced.

    `explain` holds the per-player breakdown that used to live in a module-level
    EXPLAIN dict. That global was rewritten by every call, so the backtest - which
    builds all 38 gameweeks up front - left it holding GW38 while deciding GW10.
    Keeping it here means several sets of projections can be alive at once.
    """
    df: pd.DataFrame
    gws: list
    ts: dict
    fx_map: dict
    unscheduled: list
    explain: dict = field(default_factory=dict)

    def __iter__(self):
        """Unpack as (df, gws, ts, fx_map, unscheduled), as the old return did."""
        return iter((self.df, self.gws, self.ts, self.fx_map, self.unscheduled))

    def play_chance(self, i, gw):
        """Chance the player gets ANY minutes (only 0 minutes triggers an autosub)."""
        e = self.explain.get(i)
        if not e or gw not in e["gws"] or not e["gws"][gw]["fixtures"]:
            return 0.0
        return max(0.0, min(1.0, e["gws"][gw]["availability"] * e.get("p_any", 1.0)))

    def fixture_str(self, tid, gw):
        fl = self.fx_map.get((tid, gw), [])
        return " + ".join(f"{self.ts['short'][o]} ({'H' if h else 'A'})"
                          for o, h in fl) or "BLANK"


def build_projections(bs, fixtures, cfg, next_gw, past=None):
    past = past or {}
    ts = team_strength(bs, fixtures, cfg)
    sc = bs.get("game_config", {}).get("scoring", {})
    goal_pts = sc.get("goals_scored", {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4})
    cs_pts = sc.get("clean_sheets", {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})
    dc_pts = sc.get("defensive_contribution", {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2})
    ast_pts = sc.get("assists", 3)
    w = cfg["XG_WEIGHT"]
    # Project as far as PROJECTION_WEEKS; the planner is handed a LONG_VIEW slice
    # of this and chips read the whole thing. Defaults to LONG_VIEW so a config
    # predating the split behaves exactly as it used to.
    horizon = cfg.get("PROJECTION_WEEKS") or cfg["LONG_VIEW"]
    gws = list(range(next_gw, min(next_gw + horizon, 39)))

    fx_map = {}
    for fx in fixtures:
        if fx.get("event") in gws:
            fx_map.setdefault((fx["team_h"], fx["event"]), []).append((fx["team_a"], True))
            fx_map.setdefault((fx["team_a"], fx["event"]), []).append((fx["team_h"], False))

    def per90(src, key, mins):
        return f(src.get(key, 0)) / mins * 90 if mins else 0.0

    priors = {}
    for p in POS:
        grp = [e for e in bs["elements"] if e["element_type"] == p and e["minutes"] >= 270]
        scale = cfg.get("GK_PRIOR_SCALE", 0.6) if p == 1 else 0.6
        priors[p] = {k: scale * (sum(per90(e, k, e["minutes"]) for e in grp) / len(grp) if grp else 0)
                     for k in STAT_KEYS}

    overrides = {norm(k): v for k, v in cfg["AVAILABILITY_OVERRIDES"].items()}
    start_ov = {norm(k): v for k, v in cfg.get("START_OVERRIDES", {}).items()}
    stats_on = bool(cfg.get("LAST_SEASON_STATS", True))
    starts_on = bool(cfg.get("LAST_SEASON_STARTS", False))
    K_LAST = cfg.get("LAST_SEASON_MINUTES", 540) if stats_on else 0
    M0_ALL = cfg.get("PRIOR_MINUTES", 270)
    M0_GK = cfg.get("GK_PRIOR_MINUTES") or M0_ALL
    G_LAST = cfg.get("LAST_SEASON_STARTS_GAMES", 3)
    # Per-stat (last season, position average) weights in minutes, overriding the
    # two above. Keys are a stat, or "<stat>|GK" for keepers only.
    stat_w = {}
    for k, v in sorted((cfg.get("STAT_PRIOR_MINUTES") or {}).items(), key=lambda kv: "|" in kv[0]):
        stat, _, grp = k.partition("|")
        for gk in ((True,) if grp == "GK" else (False,) if grp == "OUT" else (True, False)):
            stat_w[(stat, gk)] = (v[0] if stats_on else 0, v[1])
    exact_saves = cfg.get("EXACT_SAVES", False)
    m_decay = cfg.get("MINUTES_DECAY")
    rows, breakdown = [], {}
    for e in bs["elements"]:
        pt, t, mins = e["element_type"], e["team"], e["minutes"]
        M0 = M0_GK if pt == 1 else M0_ALL
        n = max(ts["played"].get(t, 0), 1)
        last = past.get(e["id"]) if (stats_on or starts_on) else None
        last_mins = last.get("minutes", 0) if last else 0

        rate_info = {}

        def rate(key):
            # this season + position average (always) + last season (if on, added on top)
            k_last, m0 = stat_w.get((key, pt == 1), (K_LAST, M0))
            obs = per90(e, key, mins)
            num, den = mins * obs + m0 * priors[pt][key], mins + m0
            info = {"this": obs, "this_w": mins, "prior": priors[pt][key], "prior_w": m0,
                    "last": None, "last_w": 0}
            if last and k_last and key in last:
                info["last"], info["last_w"] = per90(last, key, last_mins), k_last
                num += k_last * info["last"]
                den += k_last
            info["blended"] = num / den if den else 0.0
            rate_info[key] = info
            return info["blended"]

        g90 = w * rate("expected_goals") + (1 - w) * rate("goals_scored")
        a90 = w * rate("expected_assists") + (1 - w) * rate("assists")
        b90, s90 = rate("bonus"), rate("saves")
        dc90, y90 = rate("defensive_contribution"), rate("yellow_cards")

        recent = e.get("recent") or []
        if cfg.get("SKIP_BEFORE_DEBUT", False):
            first = next((j for j, g in enumerate(recent) if g["minutes"] > 0), len(recent))
            recent = recent[first:]
        recent = recent[-cfg.get("MINUTES_GAMES", 10):]
        if m_decay and recent:
            # recency-weighted: newest game weight 1, each older game x MINUTES_DECAY
            ws = [m_decay ** k for k in range(len(recent))][::-1]
            W = sum(ws)
            start_rate = sum(w * g["starts"] for w, g in zip(ws, recent)) / W
            mpg = sum(w * min(90, g["minutes"]) for w, g in zip(ws, recent)) / W
            p60 = sum(w * (g["minutes"] >= 60) for w, g in zip(ws, recent)) / W
            p_any = sum(w * (g["minutes"] > 0) for w, g in zip(ws, recent)) / W
            minutes_method = "recent"
        else:
            start_rate = min(1.0, e.get("starts", 0) / n)
            mpg = min(90.0, mins / n)
            if starts_on and last and n < 8 and "starts" in last:   # early season: last year's role
                ls = min(1.0, last["starts"] / 38)
                start_rate = (n * start_rate + G_LAST * ls) / (n + G_LAST)
                mpg = (n * mpg + G_LAST * min(90, last_mins / 38)) / (n + G_LAST)
            sub_mins = max(0.0, mpg - start_rate * 85)
            p_any = start_rate + min(max(0.0, sub_mins / 20), 1 - start_rate)
            p60 = start_rate * 0.92
            minutes_method = "season"
        if norm(e["web_name"]) in start_ov:
            start_rate = start_ov[norm(e["web_name"])]
            mpg = start_rate * 85
            p60 = start_rate * 0.92
            p_any = max(p_any, start_rate)
            minutes_method = "override"

        status = e.get("status", "a")
        cop = e.get("chance_of_playing_next_round")
        base_av = 0.0 if status == "u" else (1.0 if cop is None else cop / 100)
        ov = overrides.get(norm(e["web_name"]), {})
        pk = POS_KEY[pt]
        xp, gw_detail = {}, {}
        for i, gw in enumerate(gws):
            av = 0.0 if status == "u" else base_av + (1 - base_av) * min(1.0, i / 3)
            av = ov.get(gw, ov.get(str(gw), av))
            comp = dict.fromkeys(COMPONENTS, 0.0)
            fixtures_info = []
            for opp, home in fx_map.get((t, gw), []):
                att_mult = ts["def"][opp] * (1.1 if home else 0.9)
                lam_ag = ts["avg"] * ts["att"][opp] * ts["def"][t] * (0.9 if home else 1.1)
                mf = mpg / 90
                cs_prob = math.exp(-lam_ag)
                saves_mu = s90 * mf * ts["att"][opp] * (0.9 if home else 1.1) if pt == 1 else 0.0
                c = {"Playing time": p60 * 2 + max(0.0, p_any - p60),
                     "Goals": g90 * mf * att_mult * goal_pts.get(pk, 4),
                     "Assists": a90 * mf * att_mult * ast_pts,
                     "Clean sheet": p60 * cs_prob * cs_pts.get(pk, 0),
                     "Goals conceded": -p60 * exp_conceded_penalty(lam_ag) if pt in (1, 2) else 0.0,
                     "Saves": exp_per_three(saves_mu) if exact_saves else saves_mu / 3,
                     "Def. contributions": (dc_pts.get(pk, 2) * pois_at_least(DC_THRESHOLD[pt], dc90 * mf)
                                            if pt in DC_THRESHOLD else 0.0),
                     "Bonus": b90 * mf * math.sqrt(att_mult),
                     "Yellow cards": -y90 * mf}
                for k in COMPONENTS:
                    comp[k] += c[k]
                fixtures_info.append({"opp": ts["short"][opp], "home": home, "cs_prob": cs_prob,
                                      "exp_conceded": lam_ag, "exp_saves": saves_mu,
                                      "own_def": ts["def"][t], "own_att": ts["att"][t],
                                      "opp_att": ts["att"][opp], "opp_def": ts["def"][opp]})
            xp[gw] = round(av * sum(comp.values()), 2)
            gw_detail[gw] = {"availability": av, "components": {k: av * v for k, v in comp.items()},
                             "fixtures": fixtures_info}
        breakdown[e["id"]] = {"pos": pt, "start_rate": start_rate, "starts": e.get("starts", 0),
                              "p60": p60, "p_any": p_any, "minutes_method": minutes_method,
                              "recent": [g["minutes"] for g in recent][::-1],
                              "team_games": n, "mpg": mpg, "rates": rate_info,
                              "last_season": bool(last and K_LAST), "gws": gw_detail,
                              "news": e.get("news", "")}

        # price change outlook (FPL's own progress-to-change data, if present)
        rise = fall = 0.0
        proj = e.get("price_change_projections") or []
        if proj:
            pct = f(proj[min(1, len(proj) - 1)].get("projected_percent", 0))
            rise, fall = min(1.0, max(0.0, pct / 100)), min(1.0, max(0.0, -pct / 100))
        rows.append({"id": e["id"], "name": e["web_name"], "team": ts["short"][t], "team_id": t,
                     "pos": pt, "price": e["now_cost"] / 10, "status": status,
                     "news": e.get("news", ""), "can_select": e.get("can_select", True),
                     "minutes": mins, "has_last": bool(last and K_LAST), "rise": rise, "fall": fall,
                     **{f"gw{g}": xp[g] for g in gws}})
    df = pd.DataFrame(rows).set_index("id")
    df["xp_next"] = df[f"gw{gws[0]}"]
    # Long-term value is a planning quantity, so it spans LONG_VIEW, not the whole
    # projection horizon.
    df["xp_long"] = sum(df[f"gw{g}"] * cfg["DECAY"] ** i
                        for i, g in enumerate(gws[:cfg["LONG_VIEW"]]))
    unscheduled = [fx for fx in fixtures if fx.get("event") is None]
    return Projections(df, gws, ts, fx_map, unscheduled, breakdown)


# ---------------------------------------------------------------------
# 6. LOADING YOUR TEAM
# ---------------------------------------------------------------------
def find_player(df, name):
    team = None
    if "|" in name:
        name, team = [x.strip() for x in name.split("|", 1)]
    if str(name).isdigit():
        return int(name)
    key = norm(name)
    cand = df[df["name"].map(norm) == key]
    if team:
        cand = cand[cand["team"].str.upper() == team.upper()]
    if len(cand) == 0:
        cand = df[df["name"].map(norm).str.contains(key, regex=False)]
        if team:
            cand = cand[cand["team"].str.upper() == team.upper()]
    if len(cand) == 1:
        return cand.index[0]
    if len(cand) == 0:
        raise ValueError(f"Couldn't find player '{name}'. Check the spelling in the FPL app.")
    raise ValueError(f"'{name}' is ambiguous. Use one of: " +
                     ", ".join(f"{r['name']}|{r['team']}" for _, r in cand.iterrows()))


def selling_price(purchase, now):
    p, n = round(purchase * 10), round(now * 10)
    return (p + (n - p) // 2) / 10 if n > p else n / 10


def auto_prices(tid, squad, bs, used_chips):
    fh_weeks = {u["event"] for u in used_chips if u["name"] == "freehit"}
    bought = {}
    for t in sorted(fetch(f"entry/{tid}/transfers/"), key=lambda t: t["time"]):
        if t["event"] not in fh_weeks:
            bought[t["element_in"]] = t["element_in_cost"] / 10
    els = {e["id"]: e for e in bs["elements"]}
    out = {}
    for pid in squad:
        now = els[pid]["now_cost"] / 10
        start = (els[pid]["now_cost"] - els[pid].get("cost_change_start", 0)) / 10
        purchase = bought.get(pid, start)
        out[pid] = (purchase, selling_price(purchase, now))
    return out


def load_team(bs, cfg, next_gw, df):
    auto = {}
    if cfg["TEAM_ID"]:
        tid, gw = cfg["TEAM_ID"], next_gw - 1
        picks = fetch(f"entry/{tid}/event/{gw}/picks/")
        bank = picks["entry_history"]["bank"] / 10
        if picks.get("active_chip") == "freehit" and gw > 1:
            picks = fetch(f"entry/{tid}/event/{gw - 1}/picks/")
        squad = [p["element"] for p in picks["picks"]]
        used = fetch(f"entry/{tid}/history/").get("chips", [])
        auto = auto_prices(tid, squad, bs, used)
        chips = cfg["CHIPS_AVAILABLE"]
        if chips is None:
            chips = []
            for c in CHIP_NAMES:
                win = [ch for ch in bs.get("chips", []) if ch["name"] == c
                       and ch["start_event"] <= next_gw <= ch["stop_event"]]
                if win and not any(u["name"] == c and win[0]["start_event"] <= u["event"]
                                   <= win[0]["stop_event"] for u in used):
                    chips.append(c)
    else:
        squad = [find_player(df, n) for n in cfg["MY_SQUAD"]]
        bank, chips = cfg.get("BANK") or 0.0, cfg["CHIPS_AVAILABLE"] or []
    if len(set(squad)) != 15:
        raise ValueError("Squad must have 15 different players.")
    sell = {pid: (auto[pid][1] if pid in auto else None) for pid in squad}
    for n, v in cfg["SELLING_PRICES"].items():
        pid = find_player(df, n)
        if pid in sell:
            sell[pid] = v
    return squad, bank, sell, chips


# ---------------------------------------------------------------------
# 6b. FPL RULES
# ---------------------------------------------------------------------
# One home for the rules of the game. The planner expresses them as LP
# constraints and the simulator has to apply them to real state; keeping the
# statements here is what stops the two drifting apart, which is how the
# backtest ended up testing something other than what runs on a Saturday.
MAX_SAVED_FT = 5


def chip_window(bs, chip, gw):
    """Last gameweek of the window `chip` is in at `gw`, or None if it isn't in one."""
    for c in bs.get("chips", []):
        if c["name"] == chip and c["start_event"] <= gw <= c["stop_event"]:
            return c["stop_event"]
    return None if bs.get("chips") else 38


def chips_left(bs, gw, used):
    """Chips still available in the window containing `gw`.

    `used` is a set of (chip, stop_event) pairs - a chip spent in the first half
    comes back for the second, because FPL gives a fresh set per window.
    """
    out = []
    for c in CHIP_NAMES:
        stop = chip_window(bs, c, gw)
        if stop is not None and (c, stop) not in used:
            out.append(c)
    return out


def free_transfers_after(ft, n_transfers, hits, chip=None):
    """Free transfers carried into next week.

    You bank one a week up to MAX_SAVED_FT and always have at least one. Hits are
    transfers bought with points, so they don't consume a free one. Wildcard and
    Free Hit spend no free transfers at all, so the saved ones carry over intact.
    """
    if chip in ("wildcard", "freehit"):
        return min(MAX_SAVED_FT, ft)
    return min(MAX_SAVED_FT, max(1, ft - (n_transfers - hits) + 1))


# ---------------------------------------------------------------------
# 7. MULTI-WEEK TRANSFER PLANNER
# ---------------------------------------------------------------------
def solver_pool(df, squad, cfg, per_pos=30):
    excl = {find_player(df, n) for n in cfg["EXCLUDE"]}
    pool = set(squad)
    for p in POS:
        c = df[(df["pos"] == p) & df["can_select"] & ~df.index.isin(excl)]
        pool |= set(c["xp_long"].sort_values(ascending=False).head(per_pos).index)
    return sorted(pool), excl


def plan_transfers(df, gws, squad, bank, sell, cfg, n_first=None, wc_week=None, pool=None):
    """Plan transfers over PLAN_WEEKS weeks. Returns the week-by-week plan."""
    H = min(cfg["PLAN_WEEKS"], len(gws))
    T, tail_gws = gws[:H], gws[H:]
    cur = set(squad)
    if pool is None:
        pool, excl = solver_pool(df, squad, cfg)
    else:
        excl = {find_player(df, n) for n in cfg["EXCLUDE"]}
    lock = {find_player(df, n) for n in cfg["LOCK"]}
    wt = {g: cfg["DECAY"] ** i for i, g in enumerate(gws)}
    tail = {i: cfg["TAIL_WEIGHT"] * sum(wt[g] * df.loc[i, f"gw{g}"] for g in tail_gws) for i in pool}
    sp = {i: (sell[i] if i in cur else df.loc[i, "price"]) for i in pool}   # sell price
    bp = {i: df.loc[i, "price"] for i in pool}                               # buy price
    pos = {i: df.loc[i, "pos"] for i in pool}
    team = {i: df.loc[i, "team_id"] for i in pool}

    m = pulp.LpProblem("plan", pulp.LpMaximize)
    x = {(i, g): pulp.LpVariable(f"x_{i}_{g}", cat="Binary") for i in pool for g in T}
    bi = {(i, g): pulp.LpVariable(f"in_{i}_{g}", cat="Binary") for i in pool for g in T}
    so = {(i, g): pulp.LpVariable(f"out_{i}_{g}", cat="Binary") for i in pool for g in T}
    y = {(i, g): pulp.LpVariable(f"y_{i}_{g}", cat="Binary") for i in pool for g in T}
    c = {(i, g): pulp.LpVariable(f"c_{i}_{g}", cat="Binary") for i in pool for g in T}
    money = {g: pulp.LpVariable(f"bank_{g}", lowBound=0) for g in T}
    paid = {g: pulp.LpVariable(f"hits_{g}", lowBound=0, cat="Integer") for g in T}
    ft = {k: pulp.LpVariable(f"ft_{k}", lowBound=0, upBound=5, cat="Integer") for k in range(H + 1)}
    m += ft[0] == min(5, cfg["FREE_TRANSFERS"])

    obj = []
    for k, g in enumerate(T):
        prev = (lambda i: 1 if i in cur else 0) if k == 0 else (lambda i, pg=T[k - 1]: x[i, pg])
        for i in pool:
            m += x[i, g] == prev(i) + bi[i, g] - so[i, g]
            m += bi[i, g] + so[i, g] <= 1
            m += y[i, g] <= x[i, g]
            m += c[i, g] <= y[i, g]
        for p, need in SQUAD_SHAPE.items():
            m += pulp.lpSum(x[i, g] for i in pool if pos[i] == p) == need
        for tm in set(team.values()):
            m += pulp.lpSum(x[i, g] for i in pool if team[i] == tm) <= 3
        m += pulp.lpSum(y[i, g] for i in pool) == 11
        m += pulp.lpSum(c[i, g] for i in pool) == 1
        for p, (lo, hi) in XI_LIMITS.items():
            s = pulp.lpSum(y[i, g] for i in pool if pos[i] == p)
            m += s >= lo
            m += s <= hi
        prev_money = bank if k == 0 else money[T[k - 1]]
        m += money[g] == prev_money + pulp.lpSum(sp[i] * so[i, g] - bp[i] * bi[i, g] for i in pool)

        n = pulp.lpSum(bi[i, g] for i in pool)
        if wc_week is not None and k == wc_week:
            m += paid[g] == 0                     # wildcard: unlimited free transfers
            m += ft[k + 1] <= ft[k]               # saved transfers are kept
        else:
            # free transfers used = n - paid. Hits are only allowed once ALL
            # free transfers are used (stops "paying" hits to bank transfers).
            used_all = pulp.LpVariable(f"allft_{g}", cat="Binary")
            free_used = n - paid[g]
            m += free_used <= ft[k]
            m += free_used >= ft[k] - 5 * (1 - used_all)
            m += paid[g] <= 15 * used_all
            m += n <= cfg["MAX_TRANSFERS"]
            m += ft[k + 1] <= ft[k] - free_used + 1
        m += ft[k + 1] >= 1

        col = f"gw{g}"
        obj.append(wt[g] * pulp.lpSum(df.loc[i, col] * (y[i, g] + c[i, g]) for i in pool))
        obj.append(wt[g] * cfg["BENCH_WEIGHT"] *
                   pulp.lpSum(df.loc[i, col] * (x[i, g] - y[i, g]) for i in pool))
        obj.append(-cfg["HIT_COST"] * paid[g])

    for i in lock:
        if i in pool:
            m += x[i, T[0]] == 1
    for i in excl:
        if i in pool and i not in cur:
            for g in T:
                m += x[i, g] == 0
    if n_first is not None:
        m += pulp.lpSum(bi[i, T[0]] for i in pool) == n_first

    # long-term value of the final squad, leftover transfers, price changes
    obj.append(pulp.lpSum(tail[i] * x[i, T[-1]] for i in pool))
    obj.append(cfg["FT_END_VALUE"] * ft[H])
    if cfg["PRICE_CHANGES"]:
        v = cfg["POINTS_PER_TENTH"]
        for i in pool:
            r, fl = df.loc[i, "rise"], df.loc[i, "fall"]
            if i in cur:     # keeping a riser gains a little; keeping a faller loses
                obj.append(v * (0.5 * r - fl) * x[i, T[0]])
            else:            # buying before a rise locks in value; before a fall loses it
                obj.append(v * (r - fl) * bi[i, T[0]])
    m += pulp.lpSum(obj)
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=cfg["SOLVER_SECONDS"], gapRel=0.005))
    if m.status != 1:
        return None

    weeks = []
    for k, g in enumerate(T):
        ins = [i for i in pool if bi[i, g].value() > 0.5]
        outs = [i for i in pool if so[i, g].value() > 0.5]
        weeks.append({"gw": g, "in": ins, "out": outs,
                      "squad": [i for i in pool if x[i, g].value() > 0.5],
                      "hits": round(paid[g].value() or 0), "ft": round(ft[k].value()),
                      "bank": round(money[g].value(), 1),
                      "wildcard": wc_week == k})
    return {"weeks": weeks, "objective": pulp.value(m.objective)}


# ---------------------------------------------------------------------
# 8. LINEUPS & FREE HIT
# ---------------------------------------------------------------------
def pair_moves(df, outs, ins):
    pairs = []
    for p in POS:
        pairs += list(zip([i for i in outs if df.loc[i, "pos"] == p],
                          [i for i in ins if df.loc[i, "pos"] == p]))
    return pairs


def best_xi(df, squad, gw):
    col = f"gw{gw}"
    by = {p: sorted([i for i in squad if df.loc[i, "pos"] == p], key=lambda i: -df.loc[i, col])
          for p in POS}
    best = None
    for d in range(3, 6):
        for mm in range(2, 6):
            fw = 10 - d - mm
            if 1 <= fw <= 3 and len(by[2]) >= d and len(by[3]) >= mm and len(by[4]) >= fw:
                xi = by[1][:1] + by[2][:d] + by[3][:mm] + by[4][:fw]
                pts = sum(df.loc[i, col] for i in xi)
                if best is None or pts > best[0]:
                    best = (pts, xi)
    pts, xi = best
    order = sorted(xi, key=lambda i: -df.loc[i, col])
    bench = ([i for i in squad if i not in xi and df.loc[i, "pos"] == 1] +
             sorted([i for i in squad if i not in xi and df.loc[i, "pos"] != 1],
                    key=lambda i: -df.loc[i, col]))
    return {"xi": xi, "cap": order[0], "vice": order[1], "pts": pts,
            "total": pts + df.loc[order[0], col], "cap_xp": df.loc[order[0], col],
            "bench": bench, "bench_pts": sum(df.loc[i, col] for i in bench)}


def pick_lineup(proj, squad, gw, sims=4000, seed=1):
    """Choose XI, bench order and captain to maximise expected points INCLUDING autosubs.
    Each player plays (any minutes) with his play chance; if he plays he scores his
    points-if-he-plays. Non-players are replaced from the bench in order, keeping a legal
    formation; the vice gets the armband if the captain doesn't play."""
    df = proj.df
    col = f"gw{gw}"
    squad = list(squad)
    pos = {i: df.loc[i, "pos"] for i in squad}
    p = {i: proj.play_chance(i, gw) for i in squad}
    v = {i: (df.loc[i, col] / p[i] if p[i] > 0.02 else 0.0) for i in squad}   # points if he plays
    rng = np.random.default_rng(seed)
    idx = {i: k for k, i in enumerate(squad)}
    plays = rng.random((sims, len(squad))) < np.array([p[i] for i in squad])
    vals = np.array([v[i] for i in squad])

    def score(xi, bench, cap, vice):
        """Expected points for this XI, bench order and armband, over all draws at once.

        Every quantity below is a vector over the simulation axis, so the autosub
        rules are applied to all `sims` draws in a handful of array operations
        rather than a Python loop per draw.
        """
        xi_k = np.fromiter((idx[i] for i in xi), int, len(xi))
        bench_k = [idx[i] for i in bench]
        xi_pos = np.fromiter((pos[squad[k]] for k in xi_k), int, len(xi_k))
        pl = plays[:, xi_k]                              # (sims, 11) who turned out

        total = (pl * vals[xi_k]).sum(1)
        count = {q: (pl & (xi_pos == q)).sum(1) for q in POS}
        missing_gk = (~pl & (xi_pos == 1)).any(1)
        n_missing_out = (~pl & (xi_pos != 1)).sum(1)

        gk_b = bench_k[0]                                # goalkeeper autosub
        total = total + (missing_gk & plays[:, gk_b]) * vals[gk_b]

        for b in bench_k[1:]:                            # outfield, in bench order
            q = pos[squad[b]]
            need_q = np.maximum(0, MIN_IN_XI[q] - count[q])
            need_all = sum(np.maximum(0, MIN_IN_XI[r] - count[r]) for r in (2, 3, 4))
            # Come on if you fill a shortfall, or if there is room spare once every
            # formation minimum is covered.
            take = (plays[:, b] & (n_missing_out > 0)
                    & ((need_q > 0) | (n_missing_out - need_all > 0)))
            total = total + take * vals[b]
            count[q] = count[q] + take
            n_missing_out = n_missing_out - take

        cap_k, vice_k = idx[cap], idx[vice]               # vice takes over if needed
        total = total + np.where(plays[:, cap_k], vals[cap_k],
                                 np.where(plays[:, vice_k], vals[vice_k], 0.0))
        return total.mean()

    def bench_for(xi):
        rest = [i for i in squad if i not in xi]
        return ([i for i in rest if pos[i] == 1] +
                sorted([i for i in rest if pos[i] != 1], key=lambda i: -v[i] * p[i] - 0.01 * v[i]))

    # candidate XIs: for every legal formation, best by expected points and by points-if-he-plays
    by = lambda key, q: sorted([i for i in squad if pos[i] == q], key=key)
    cands = set()
    for d in range(3, 6):
        for mm in range(2, 6):
            fw = 10 - d - mm
            if not 1 <= fw <= 3:
                continue
            for key in (lambda i: -df.loc[i, col], lambda i: -v[i]):
                g, de, mi, fo = by(key, 1), by(key, 2), by(key, 3), by(key, 4)
                if len(de) >= d and len(mi) >= mm and len(fo) >= fw:
                    cands.add(tuple(sorted(g[:1] + de[:d] + mi[:mm] + fo[:fw])))
    # quick screen with a smaller sample, then full evaluation of the best few
    best = None
    for xi in cands:
        xi = list(xi)
        order = sorted(xi, key=lambda i: -v[i] * (0.5 + 0.5 * p[i]))
        cap, vice = order[0], order[1]
        sc = score(xi, bench_for(xi), cap, vice)
        if best is None or sc > best[0]:
            best = (sc, xi)
    xi = best[1]
    bench = bench_for(xi)
    top = sorted(xi, key=lambda i: -v[i])[:4]
    best_cv = max(((score(xi, bench, c, vc), c, vc) for c in top for vc in top if vc != c),
                  key=lambda t: t[0])
    exp_total, cap, vice = best_cv
    # Expected points from one armband: the captain if he plays, else the vice.
    # This is exactly what a Triple Captain adds on top.
    arm_xp = p[cap] * v[cap] + (1 - p[cap]) * p[vice] * v[vice]
    return {"xi": xi, "cap": cap, "vice": vice, "bench": bench, "total": exp_total,
            "if_plays": v, "play_chance": p, "arm_xp": arm_xp,
            "pts": sum(df.loc[i, col] for i in xi), "cap_xp": df.loc[cap, col],
            "bench_pts": sum(df.loc[i, col] for i in bench)}


def free_hit_squad(df, gw, budget, cfg):
    col = f"gw{gw}"
    excl = {find_player(df, n) for n in cfg["EXCLUDE"]}
    pool = set()
    for p in POS:
        cands = df[(df["pos"] == p) & df["can_select"] & ~df.index.isin(excl)]
        pool |= set(cands[col].sort_values(ascending=False).head(25).index)
    pool = sorted(pool)
    m = pulp.LpProblem("fh", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", pool, cat="Binary")
    y = pulp.LpVariable.dicts("y", pool, cat="Binary")
    c = pulp.LpVariable.dicts("c", pool, cat="Binary")
    for p, need in SQUAD_SHAPE.items():
        m += pulp.lpSum(x[i] for i in pool if df.loc[i, "pos"] == p) == need
    for t in set(df.loc[pool, "team_id"]):
        m += pulp.lpSum(x[i] for i in pool if df.loc[i, "team_id"] == t) <= 3
    m += pulp.lpSum(df.loc[i, "price"] * x[i] for i in pool) <= budget
    m += pulp.lpSum(y.values()) == 11
    m += pulp.lpSum(c.values()) == 1
    for p, (lo, hi) in XI_LIMITS.items():
        s = pulp.lpSum(y[i] for i in pool if df.loc[i, "pos"] == p)
        m += s >= lo
        m += s <= hi
    for i in pool:
        m += y[i] <= x[i]
        m += c[i] <= y[i]
    m += pulp.lpSum(df.loc[i, col] * (y[i] + c[i]) for i in pool)
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=20))
    squad = [i for i in pool if x[i].value() > 0.5]
    return squad, pulp.value(m.objective)


# ---------------------------------------------------------------------
# 8b. SQUAD FORECAST
# ---------------------------------------------------------------------
def forecast_squads(proj, squad, bank, sell, cfg, until, plan_gws=None, pool_per_pos=None):
    """The squad you would plausibly own in each week from now to `until`.

    Steps forward one week at a time: run the ordinary planner from the forecast
    state, keep only its first week's moves, then move on. That is what you would
    actually do on each of those deadlines.

    Chips need this because a chip is worth what it adds to the team you will have
    when you play it. The old code valued every future week against
    `plan["weeks"][-1]["squad"]` - the squad frozen at the end of the 5-week plan -
    so a GW30 Bench Boost was priced against a GW6 bench.

    Returns {gw: {"squad", "bank", "sell", "budget"}} for the state after that
    week's moves. Assumes no chips are played later; if a Wildcard is played now,
    rebuild this from the Wildcard squad.
    """
    df, gws = proj.df, proj.gws
    horizon = [g for g in gws if g <= until]
    # How many weeks ahead the planner looks at each future deadline. Chips see
    # further than the planner does, so this is not simply len(proj.gws).
    view_len = len(plan_gws) if plan_gws else len(gws)
    cfg = dict(cfg)
    cfg["SOLVER_SECONDS"] = cfg.get("PATH_SOLVER_SECONDS", 5)
    per_pos = pool_per_pos or cfg.get("FORECAST_POOL_PER_POS", 20)

    out, cur, cur_bank, cur_sell = {}, list(squad), bank, dict(sell)
    ft = cfg.get("FREE_TRANSFERS", 1)
    for k, g in enumerate(horizon):
        remaining = [w for w in gws if w >= g][:view_len]   # what week g would see
        # State as it stands on week g's deadline, before that week's moves. The
        # wildcard is valued from here, because playing it replaces those moves.
        before = {"squad": list(cur), "bank": cur_bank, "sell": dict(cur_sell),
                  "ft": ft, "remaining": remaining}
        objective = None
        if len(remaining) >= 2:
            cfg["FREE_TRANSFERS"] = ft
            pool, _ = solver_pool(df, cur, cfg, per_pos=per_pos)
            p = plan_transfers(df, remaining, cur, cur_bank, cur_sell, cfg, pool=pool)
            if p:                              # infeasible: hold what we have
                objective = p["objective"]
                wk = p["weeks"][0]
                for i in wk["in"]:
                    # No price movement is modelled ahead, so a player bought
                    # later sells for what he cost.
                    cur_sell[i] = df.loc[i, "price"]
                for o in wk["out"]:
                    cur_sell.pop(o, None)
                cur, cur_bank = wk["squad"], wk["bank"]
                ft = free_transfers_after(ft, len(wk["in"]), wk["hits"])
        out[g] = {"squad": list(cur), "bank": cur_bank, "sell": dict(cur_sell),
                  "budget": cur_bank + sum(cur_sell.get(i, df.loc[i, "price"]) for i in cur),
                  "before": before, "objective": objective}
        if len(remaining) < 2:
            break
    return out


def forecast_horizon(bs, gws, chips):
    """Last week worth forecasting to: the latest expiry among unused chips."""
    stops = [chip_window(bs, c, gws[0]) for c in chips]
    stops = [s for s in stops if s]
    return min(max(stops), gws[-1]) if stops else gws[0]


# ---------------------------------------------------------------------
# 9. CHIP TIMING
# ---------------------------------------------------------------------
# A chip is worth nothing unused, so the question is never "is this week good?"
# but "is this week good enough, given how many chances are left?". Each chip is
# valued the same way in every remaining week of its window, this week is ranked
# among them, and it is played if it lands in the top x% - a bar that tightens
# automatically as the window runs out, and reaches "play it" in the last week.
#
# This replaces CHIP_RATIO, CHIP_WAIT_DECAY, CHIP_IGNORE_MIN_IF_VISIBLE and four
# absolute minimums (WILDCARD_THRESHOLD, FREEHIT_THRESHOLD, BBOOST_MIN, TC_MIN),
# which between them tried to express the same idea with fixed point thresholds
# that could not know how much season was left.




def chip_value(chip, g, proj, state, cfg, fh_cache=None, window_stop=None):
    """What `chip` would be worth if played in gameweek `g`.

    `state` is one week of `forecast_squads`, so every week is measured against
    the squad you would actually own then. Each chip is in its own units; they are
    never compared with each other, only with the same chip in other weeks.
    """
    df, gws = proj.df, proj.gws
    squad, budget = state["squad"], state["budget"]

    if chip == "3xc":
        # The extra armband, allowing for the captain not playing and the vice
        # taking over.
        return pick_lineup(proj, squad, g)["arm_xp"]

    if chip == "bboost":
        return pick_lineup(proj, squad, g)["bench_pts"]

    if chip == "freehit":
        # Best one-week team money can buy, against the team you would have had.
        key = (g, round(budget, 1))
        if fh_cache is not None and key in fh_cache:
            fh_pts = fh_cache[key][1]
        else:
            fh_squad, fh_pts = free_hit_squad(df, g, budget, cfg)
            if fh_cache is not None:
                fh_cache[key] = (fh_squad, fh_pts)
        return fh_pts - best_xi(df, squad, g)["total"]

    if chip == "wildcard":
        # What a free rebuild is worth from here on: the planner's multi-week
        # objective with unlimited transfers this week, minus the same objective
        # without. Both are measured from the same state on week g's deadline and
        # over the same weeks, so weeks can be ranked against each other.
        #
        # Not "best possible 15 minus the squad you'd have" - that gap shrinks every
        # week, because the forecast assumes you keep transferring towards the same
        # optimum, so it would rank the first week of the window best every time and
        # the chip would always be played immediately.
        #
        # A first-half wildcard is only credited up to the end of its window: from
        # the next window's first week a fresh wildcard can rebuild the squad anyway,
        # so benefit after that is not this chip's. Without the cap a GW17 wildcard
        # was credited out to GW35. WC_CAP_AT_NEXT_WINDOW=False restores it.
        before, baseline = state["before"], state["objective"]
        if baseline is None:
            return 0.0
        wcfg = dict(cfg, FREE_TRANSFERS=before["ft"],
                    SOLVER_SECONDS=cfg.get("PATH_SOLVER_SECONDS", 5))
        pool, _ = solver_pool(df, before["squad"], wcfg,
                              per_pos=cfg.get("FORECAST_POOL_PER_POS", 20))
        weeks = before["remaining"]
        if window_stop and window_stop < 38 and cfg.get("WC_CAP_AT_NEXT_WINDOW", True):
            capped = [w for w in weeks if w <= window_stop]
            if len(capped) < len(weeks):
                # the baseline must cover the same weeks, so re-solve it too
                weeks = capped
                b = plan_transfers(df, weeks, before["squad"], before["bank"],
                                   before["sell"], wcfg, pool=pool)
                if not b:
                    return 0.0
                baseline = b["objective"]
        p = plan_transfers(df, weeks, before["squad"], before["bank"],
                           before["sell"], wcfg, wc_week=0, pool=pool)
        return (p["objective"] - baseline) if p else 0.0

    raise ValueError(f"unknown chip {chip!r}")


def rank_verdict(values, g0, stop, x):
    """Rank this week among the remaining ones and decide.

    Play if this week is in the top `x` share of what is left, always play in the
    window's last week, and never play a chip worth nothing.
    """
    weeks = sorted(values)
    n = len(weeks)
    rank = 1 + sum(1 for g in weeks if values[g] > values[g0])
    cutoff = max(1, math.ceil(x * n))
    last_chance = (g0 == stop) or n == 1
    play = bool(values[g0] > 0) and (rank <= cutoff or last_chance)
    later = [values[g] for g in weeks if g > g0]
    # What playing now costs you if a better week was coming. Infinite when there
    # is no later week, so a last-chance chip always wins a clash.
    regret = (values[g0] - max(later)) if later else float("inf")
    return {"play": play, "rank": rank, "n": n, "cutoff": cutoff, "now": values[g0],
            "best_gw": max(weeks, key=lambda g: values[g]), "values": values,
            "regret": regret, "last_chance": last_chance, "stop": stop}


def chip_advice(proj, bs, chips, squad, bank, sell, cfg, forecast=None, plan_gws=None,
                fh_cache=None):
    """Value every unused chip in every remaining week of its window, then decide.

    Returns {chip: verdict}. At most one chip comes back with play=True: when two
    qualify in the same week, the one that loses most by waiting is played and the
    other is reconsidered next week. Comparing the chips by how much each would
    lose keeps the comparison inside each chip's own units, rather than weighing
    captain points against weighted wildcard points as the old tie-break did.
    """
    gws = proj.gws
    g0 = gws[0]
    if not chips:
        return {}
    if forecast is None:
        until = forecast_horizon(bs, gws, chips)
        forecast = forecast_squads(proj, squad, bank, sell, cfg, until, plan_gws)

    top_pct = cfg.get("CHIP_TOP_PCT", {})
    fh_cache = {} if fh_cache is None else fh_cache
    out = {}
    for chip in chips:
        stop = chip_window(bs, chip, g0)
        weeks = [g for g in gws if stop and g <= stop and g in forecast]
        if not weeks:
            continue
        gap_bar = (cfg.get("CHIP_BAR") or {}).get(chip)
        if chip == "wildcard" and cfg.get("WILDCARD_GAP") is not None:
            gap_bar = cfg["WILDCARD_GAP"]
        if gap_bar is not None:
            # Bar rule: play when the chip is worth at least `gap_bar` now, rather
            # than ranking this week against a forecast that assumes no news (which
            # always makes later weeks look better, so the chip is held too long).
            now = chip_value(chip, g0, proj, forecast[g0], cfg, fh_cache, window_stop=stop)
            last_chance = g0 == stop or len(weeks) == 1
            v = {"play": bool(now > 0 and (now >= gap_bar or last_chance)), "rank": 1, "n": 1,
                 "cutoff": 1, "now": now, "best_gw": g0, "values": {g0: now},
                 "regret": float("inf") if last_chance else now - gap_bar,
                 "last_chance": last_chance, "stop": stop, "gap_bar": gap_bar}
        else:
            values = {g: chip_value(chip, g, proj, forecast[g], cfg, fh_cache, window_stop=stop)
                  for g in weeks}
            v = rank_verdict(values, g0, stop, top_pct.get(chip, 0.15))
        v["visible_to"] = gws[-1]
        v["beyond_view"] = bool(stop and stop > gws[-1])
        if chip == "freehit" and v["play"]:
            key = (g0, round(forecast[g0]["budget"], 1))
            v["squad"] = fh_cache[key][0] if key in fh_cache else \
                free_hit_squad(proj.df, g0, forecast[g0]["budget"], cfg)[0]
        out[chip] = v

    playing = [c for c in out if out[c]["play"]]
    if len(playing) > 1:
        keep = max(playing, key=lambda c: out[c]["regret"])
        for c in playing:
            if c != keep:
                out[c]["play"] = False
                out[c]["deferred_for"] = keep
    return out


def chip_line(chip, v):
    """One line of the chip table for the report."""
    name = CHIP_NAMES[chip]
    share = v["rank"] / v["n"]
    where = (f"this week #{v['rank']} of {v['n']} remaining (top {share:.0%}, "
             f"playing if in top {v['cutoff'] / v['n']:.0%})")
    if "gap_bar" in v:
        where = f"playing at {v['gap_bar']:+.1f} or more"
        if not v["play"] and v["now"] > 0:
            return f"  {name}: {v['now']:+.1f} now, {where} -> save - below the bar"
    if v["play"]:
        verdict = "PLAY" + (" - last week of the window" if v["last_chance"]
                            and v["rank"] > v["cutoff"] else "")
    elif v.get("deferred_for"):
        verdict = f"save - {CHIP_NAMES[v['deferred_for']]} loses more by waiting"
    elif v["now"] <= 0:
        verdict = "save - worth nothing this week"
    else:
        verdict = f"save - best remaining week is GW{v['best_gw']} ({v['values'][v['best_gw']]:+.1f})"
    note = ""
    if v["beyond_view"]:
        note = f"  (can't see past GW{v['visible_to']}; window runs to GW{v['stop']})"
    elif v["stop"] and v["stop"] - list(sorted(v["values"]))[0] <= 3:
        note = f"  ⚠ expires after GW{v['stop']}"
    return f"  {name}: {v['now']:+.1f} now, {where} -> {verdict}{note}"


# ---------------------------------------------------------------------
# 9c. THE WEEKLY DECISION
# ---------------------------------------------------------------------
@dataclass
class Decision:
    """One gameweek's decision, whoever is asking.

    The live report prints this; the backtest applies it and scores what really
    happened. Both get here the same way - previously each had its own copy of
    the sequence and they had already diverged on how the XI was picked.
    """
    plans: dict          # transfers made this week -> best multi-week plan starting that way
    best_n: int          # the number of transfers that scored highest
    plan: dict           # the plan being followed (the wildcard plan, if one is played)
    advice: dict         # chip -> chip_advice result
    chip: str = None     # chip being played this week, if any
    squad: list = None   # the 15 to field this week (a Free Hit squad if that chip is on)
    ins: list = field(default_factory=list)
    outs: list = field(default_factory=list)
    hits: int = 0
    bank: float = 0.0    # bank after this week's moves
    keep_squad: bool = False   # Free Hit: this week's squad is borrowed, don't keep it

    @property
    def n_transfers(self):
        return len(self.ins)


def choose_transfers(proj, gws, squad, bank, sell, cfg, pool=None, wc_ahead=None):
    """Best multi-week plan for every possible number of transfers this week.

    Returns (plans, best_n, pool). Solving each `n_first` separately is what lets
    the report show "0 transfers vs 1 vs 2" as real alternatives rather than one
    answer; the decision just takes the best objective. `wc_ahead` is how many
    weeks from now a wildcard is already planned, so nothing is bought just to be
    rebuilt away (and no hits are taken) in the weeks before it.
    """
    if pool is None:
        pool, _ = solver_pool(proj.df, squad, cfg)
    plans = {}
    for n in range(0, cfg["MAX_TRANSFERS"] + 1):
        p = plan_transfers(proj.df, gws, squad, bank, sell, cfg, n_first=n,
                           wc_week=wc_ahead, pool=pool)
        if p:
            plans[n] = p
    if not plans:
        raise RuntimeError("the transfer planner found no feasible plan")
    return plans, max(plans, key=lambda n: plans[n]["objective"]), pool


def decide_week(proj, gws, bs, squad, bank, sell, cfg, chips=(), forecast=None,
                fh_cache=None, force_wildcard=False, wc_ahead=None):
    """Plan this week's transfers, then decide whether a chip beats them.

    `force_wildcard` plays the wildcard this week whatever the timing rule says -
    the backtest uses it to measure what each week would really have paid.
    `wc_ahead` is passed on to `choose_transfers`.
    """
    plans, best_n, pool = choose_transfers(proj, gws, squad, bank, sell, cfg,
                                           wc_ahead=wc_ahead)
    plan = plans[best_n]
    advice = (chip_advice(proj, bs, chips, squad, bank, sell, cfg, forecast, plan_gws=gws,
                          fh_cache=fh_cache) if chips else {})
    chip = "wildcard" if force_wildcard else \
        next((c for c, v in advice.items() if v["play"]), None)

    d = Decision(plans=plans, best_n=best_n, plan=plan, advice=advice, bank=bank)

    if chip == "freehit":
        # A borrowed squad for one week: no permanent moves, so the real squad,
        # the bank and the saved free transfers are all untouched.
        d.chip, d.squad, d.keep_squad = "freehit", advice["freehit"]["squad"], False
        return d

    if chip == "wildcard":
        # Only now is the full planner run with the wildcard in place; the value
        # used to rank the weeks was a single cheap solve.
        wc = plan_transfers(proj.df, gws, squad, bank, sell, cfg, wc_week=0, pool=pool)
        if wc:
            d.chip, d.plan = "wildcard", wc
            wk = wc["weeks"][0]
            d.hits = 0                      # unlimited free transfers
        else:
            if "wildcard" in advice:
                advice["wildcard"]["play"] = False
            wk = plan["weeks"][0]
            d.hits = wk["hits"]
    else:
        wk = plan["weeks"][0]
        d.hits = wk["hits"]
        d.chip = chip if chip in ("3xc", "bboost") else None
    d.ins, d.outs, d.squad, d.keep_squad = wk["in"], wk["out"], wk["squad"], True
    d.bank = wk["bank"]
    return d


# ---------------------------------------------------------------------
# 9b. POINTS BREAKDOWN
# ---------------------------------------------------------------------
RATE_LABELS = {"saves": "Saves", "bonus": "Bonus", "expected_goals": "xG",
               "expected_assists": "xA", "defensive_contribution": "Def. actions"}
RATES_BY_POS = {1: ["saves", "bonus"],
                2: ["expected_goals", "expected_assists", "defensive_contribution", "bonus"],
                3: ["expected_goals", "expected_assists", "defensive_contribution", "bonus"],
                4: ["expected_goals", "expected_assists", "bonus"]}


def explain(proj, *names, gw=None):
    """Show where each player's projected points come from.
    Example: report.explain("Verbruggen", "Trafford")"""
    df, gws, ex = proj.df, proj.gws, proj.explain
    gw = gw or gws[0]
    ids = [find_player(df, n) for n in names]
    W = 16
    head = lambda label, vals: print(f"{label:<26}" + "".join(f"{v:>{W}}" for v in vals))
    print(f"\nPOINTS BREAKDOWN - GW{gw}\n" + "=" * (26 + W * len(ids)))
    head("", [df.loc[i, "name"] for i in ids])
    fx = []
    for i in ids:
        f_ = ex[i]["gws"][gw]["fixtures"]
        fx.append(" + ".join(f"{x['opp']} ({'H' if x['home'] else 'A'})" for x in f_) or "BLANK")
    head("Fixture", fx)
    head("Starts this season", [f"{ex[i]['starts']} of {ex[i]['team_games']}" for i in ids])
    head("Last 5 games (newest 1st)", [",".join(str(int(m)) for m in ex[i]["recent"][:5]) or "-"
                                       for i in ids])
    head("Chance of starting", [f"{ex[i]['start_rate']:.0%}" for i in ids])
    head("Chance of 60+ minutes", [f"{ex[i]['p60']:.0%}" for i in ids])
    head("Expected minutes", [f"{ex[i]['mpg']:.0f}" for i in ids])
    head("Availability", [f"{ex[i]['gws'][gw]['availability']:.0%}" for i in ids])
    print("-" * (26 + W * len(ids)))
    for k in COMPONENTS:
        vals = [ex[i]["gws"][gw]["components"][k] for i in ids]
        if all(abs(v) < 0.005 for v in vals):
            continue
        head(k, [f"{v:+.2f}" for v in vals])
    print("-" * (26 + W * len(ids)))
    head("TOTAL", [f"{df.loc[i, f'gw{gw}']:.2f}" for i in ids])

    print("\nBehind the numbers (single fixture shown for double gameweeks):")
    for label, key, fmt in [("Own defence (vs avg)*", "own_def", "{:.0%}"),
                            ("Opponent attack (vs avg)", "opp_att", "{:.0%}"),
                            ("Clean sheet chance", "cs_prob", "{:.0%}"),
                            ("Goals expected against", "exp_conceded", "{:.2f}"),
                            ("Saves expected", "exp_saves", "{:.1f}")]:
        vals = []
        for i in ids:
            f_ = ex[i]["gws"][gw]["fixtures"]
            vals.append(fmt.format(f_[0][key]) if f_ else "-")
        if key == "exp_saves" and all(ex[i]["pos"] != 1 for i in ids):
            continue
        head(label, vals)

    print("  * goals conceded vs league average: above 100% = leakier than average")
    print("\nWhere each per-90 rate comes from (this season | last season | position avg -> used):")
    for i in ids:
        d = ex[i]
        print(f"  {df.loc[i, 'name']}")
        for key in RATES_BY_POS[d["pos"]]:
            r = d["rates"].get(key)
            if not r:
                continue
            last = (f"{r['last']:.2f} (x{r['last_w']} min)" if r["last"] is not None
                    else "not used")
            print(f"    {RATE_LABELS[key]:<13} {r['this']:.2f} (x{r['this_w']} min) | {last} | "
                  f"{r['prior']:.2f} (x{r['prior_w']} min) -> {r['blended']:.2f}")
        if d["news"]:
            print(f"    News: {d['news']}")


def team_table(bs, fixtures, cfg):
    """Every team's defence rating next to what actually happened.
    Rating: 100% = league average, higher = leakier."""
    ts = team_strength(bs, fixtures, cfg)
    rows = sorted(ts["detail"].items(), key=lambda kv: -ts["def"][kv[0]])
    print(f"\nTEAM DEFENCE (league avg goals per team per game: {ts['avg']:.2f})")
    print(f"{'Team':<6}{'Games':>6}{'Conceded':>10}{'xG agst':>9}{'Pre-season':>12}"
          f"{'Keeper busy':>13}{'Adjust':>8}{'RATING':>9}")
    print(f"{'':<6}{'':>6}{'per game':>10}{'per game':>9}{'rating':>12}{'shots/90':>13}")
    for t, d in rows:
        ga = f"{d['ga_pg']:.2f}" if d["ga_pg"] is not None else "-"
        xga = f"{d['xga_pg']:.2f}" if d["xga_pg"] is not None else "-"
        busy = f"{d['keeper_busy']:.1f}" if d["keeper_busy"] is not None else "-"
        print(f"{ts['short'][t]:<6}{d['played']:>6}{ga:>10}{xga:>9}{d['prior']:>12.0%}"
              f"{busy:>13}{d['leak']:>8.2f}{ts['def'][t]:>9.0%}")
    print("Pre-season rating counts less as games are played. 'Adjust' is the keeper-workload\n"
          "multiplier (SAVES_LEAK): above 1 = busy keeper, defence treated as leakier.")


# ---------------------------------------------------------------------
# 10. REPORT
# ---------------------------------------------------------------------
@dataclass
class Report:
    """What `run()` produced. Replaces the old LAST_RUN global, so `explain` and
    `team_table` work off the run you name rather than whichever ran last."""
    proj: Projections
    plan: dict
    lineup: dict
    chips: dict
    squad: list
    bank: float
    cfg: dict
    bs: dict = field(repr=False, default=None)
    fixtures: list = field(repr=False, default=None)

    @property
    def df(self):
        return self.proj.df

    def explain(self, *names, gw=None):
        return explain(self.proj, *names, gw=gw)

    def team_table(self):
        return team_table(self.bs, self.fixtures, self.cfg)


def run(cfg=None, data=None, past=None):
    cfg = cfg or resolve_cfg()
    if data is None:
        bs, fixtures = fetch("bootstrap-static/"), fetch("fixtures/")
    else:
        bs, fixtures = data
    next_gw = next(e["id"] for e in bs["events"] if e.get("is_next"))
    proj = build_projections(bs, fixtures, cfg, next_gw)
    squad, bank, sell, chips = load_team(bs, cfg, next_gw, proj.df)
    for i in squad:
        if sell[i] is None:
            sell[i] = proj.df.loc[i, "price"]

    # second pass: last season + recent game-by-game minutes for players that matter
    if cfg.get("LAST_SEASON_STATS", True) or cfg.get("LAST_SEASON_STARTS", False) \
            or cfg.get("MINUTES_DECAY"):
        if past is None:
            ids, _ = solver_pool(proj.df, squad, cfg, per_pos=45)
            print(f"Loading player histories for {len(ids)} players...")
            past, recent = fetch_player_pages(ids)
            for e in bs["elements"]:
                if e["id"] in recent:
                    e["recent"] = recent[e["id"]]
        proj = build_projections(bs, fixtures, cfg, next_gw, past)

    df, gws, ts, fx_map, unscheduled = proj
    # The planner sees LONG_VIEW weeks; chips read the full horizon off `proj`.
    # The backtest splits these the same way, so the two cannot drift.
    plan_gws = gws[:cfg["LONG_VIEW"]]
    g0 = plan_gws[0]
    nm = lambda i: f"{df.loc[i, 'name']} ({df.loc[i, 'team']}, £{df.loc[i, 'price']:.1f}m)"
    L = "=" * 62
    fixture_str = proj.fixture_str

    print(L)
    print(f" FPL ASSISTANT - GW{g0}   plan GW{g0}-{g0 + cfg['PLAN_WEEKS'] - 1}, "
          f"long view to GW{plan_gws[-1]}, projections to GW{gws[-1]}")
    print(L)
    print(f"Bank £{bank:.1f}m | Free transfers {cfg['FREE_TRANSFERS']} | "
          f"Chips: {', '.join(CHIP_NAMES[c] for c in chips) or 'none'}\n")

    print("YOUR SQUAD")
    print(f"{'Player':<24}{'Sell':>6}{'Next':>6}{'12wk':>7}  Fixture / price outlook")
    for p in POS:
        for i in sorted([i for i in squad if df.loc[i, "pos"] == p],
                        key=lambda i: -df.loc[i, "xp_long"]):
            price = (" ↑ likely rise" if df.loc[i, "rise"] >= 0.8 else
                     " ↓ likely fall" if df.loc[i, "fall"] >= 0.8 else "")
            news = f"  ⚠ {df.loc[i, 'news']}" if df.loc[i, "news"] else ""
            print(f"{POS[p] + ' ' + df.loc[i, 'name']:<24}{sell[i]:>6.1f}{df.loc[i, 'xp_next']:>6.1f}"
                  f"{df.loc[i, 'xp_long']:>7.1f}  {fixture_str(df.loc[i, 'team_id'], g0)}{price}{news}")

    # ---- the decision (the same call the backtest makes)
    d = decide_week(proj, plan_gws, bs, squad, bank, sell, cfg, chips)
    plans, best_n, plan, advice, active = d.plans, d.best_n, d.plan, d.advice, d.chip
    w0 = plans[best_n]["weeks"][0]

    # ---- chips
    print(f"\n{L}\n CHIPS  (compared across every week of each chip's window)\n{L}")
    if not chips:
        print("  No chips available.")
    for chip, v in advice.items():
        print(chip_line(chip, v))
    if unscheduled:
        teams = sorted({ts["short"][t] for fx in unscheduled for t in (fx["team_h"], fx["team_a"])})
        print(f"  ℹ {len(unscheduled)} fixture(s) not yet scheduled ({', '.join(teams)}). "
              "These usually become double gameweeks - a reason to hold Bench Boost / "
              "Triple Captain if nothing stands out.")

    # ---- transfers
    print(f"\n{L}\n TRANSFERS\n{L}")
    lineup_squad = d.squad
    if active == "wildcard":
        print(f">>> PLAY WILDCARD - make these moves (no hits):")
        for o, i in pair_moves(df, d.outs, d.ins):
            print(f"   OUT {nm(o):<32} IN {nm(i)}")
        print(f"   Bank after: £{d.bank:.1f}m")
    elif active == "freehit":
        print(">>> PLAY FREE HIT - pick this one-week squad (your team returns next week):")
        for p_ in POS:
            print(f"   {POS[p_]}: " + ", ".join(nm(i) for i in lineup_squad
                                            if df.loc[i, "pos"] == p_))
        print("   No permanent transfers this week - your free transfer rolls over.")
    else:
        print("Options this week (each is the best multi-week plan starting that way):")
        base = plans[0]["objective"]
        for n, p in plans.items():
            wk = p["weeks"][0]
            hit = f"  (-{wk['hits'] * cfg['HIT_COST']} hit)" if wk["hits"] else ""
            print(f"\n {n} transfer(s){hit}: {p['objective'] - base:+.1f} pts vs rolling")
            for o, i in pair_moves(df, wk["out"], wk["in"]):
                print(f"   OUT {nm(o):<32} IN {nm(i)}")
        print(f"\n>>> RECOMMENDED: " + (
            "roll your free transfer" if best_n == 0 else
            f"{best_n} transfer(s)" + (f", taking a -{w0['hits'] * cfg['HIT_COST']} hit"
                                       if w0["hits"] else "")))
    if active != "freehit":
        print(f"\n The plan ahead (only this week is locked in - rerun every week):")
        for w in plan["weeks"]:
            moves = pair_moves(df, w["out"], w["in"])
            txt = ", ".join(f"{df.loc[o, 'name']} → {df.loc[i, 'name']}" for o, i in moves) or "roll"
            if w["wildcard"]:
                txt = "WILDCARD (" + str(len(moves)) + " moves)"
            hit = f" (-{w['hits'] * cfg['HIT_COST']})" if w["hits"] else ""
            print(f"   GW{w['gw']}: {w['ft']} FT → {txt}{hit}   bank £{w['bank']:.1f}m")

    # ---- lineup
    lu = pick_lineup(proj, lineup_squad, g0)
    col = f"gw{g0}"
    mult = 3 if active == "3xc" else 2
    print(f"\n{L}\n STARTING XI - GW{g0}\n{L}")
    print(f"  {'':<4}{'':<24}{'xP':>5}{'If plays':>10}{'Plays':>7}  Fixture")
    for p in POS:
        for i in [i for i in lu["xi"] if df.loc[i, "pos"] == p]:
            tag = (" (TC)" if mult == 3 else " (C)") if i == lu["cap"] else \
                  " (V)" if i == lu["vice"] else ""
            print(f"  {POS[p]:<4}{df.loc[i, 'name'] + tag:<24}{df.loc[i, col]:>5.1f}"
                  f"{lu['if_plays'][i]:>10.1f}{lu['play_chance'][i]:>7.0%}  "
                  f"{fixture_str(df.loc[i, 'team_id'], g0)}")
    print("  Bench (autosub order): " + ", ".join(
        f"{k + 1}. {df.loc[i, 'name']} ({lu['if_plays'][i]:.1f} if plays, {lu['play_chance'][i]:.0%})"
        for k, i in enumerate(lu["bench"])))
    if active == "bboost":
        total = sum(df.loc[i, col] for i in lineup_squad) + lu["cap_xp"]
    elif mult == 3:
        total = lu["total"] + lu["play_chance"][lu["cap"]] * lu["if_plays"][lu["cap"]]
    else:
        total = lu["total"]
    print(f"  Projected GW{g0} score (including autosubs): {total:.1f}")
    print("  Starters are picked on points IF they play - the bench covers anyone who gets 0 minutes.")

    # ---- value table
    print(f"\n{L}\n VALUE vs POSITION AVERAGE  (projected pts per £m, {len(plan_gws)} weeks)\n{L}")
    df["value"] = df["xp_long"] / df["price"]
    reg = df[(df["minutes"] >= 180) & (df["status"] != "u")]
    avg = reg.groupby("pos")["value"].mean()
    for p in POS:
        print(f"\n  {POS[p]} (average = 100%)")
        for i in [i for i in squad if df.loc[i, "pos"] == p]:
            print(f"    YOURS {df.loc[i, 'name']:<20}{df.loc[i, 'value'] / avg[p] * 100:>5.0f}%")
        for i, r in reg[reg["pos"] == p].sort_values("value", ascending=False).head(
                3 if p == 1 else 5).iterrows():
            print(f"    TOP   {r['name'] + ' (' + r['team'] + ', £' + format(r['price'], '.1f') + 'm)':<32}"
                  f"{r['value'] / avg[p] * 100:>5.0f}%")
    print(f"\n{L}\nEstimates only - check late team news before the deadline.")
    return Report(proj=proj, plan=plan, lineup=lu, chips=advice, squad=squad,
                  bank=bank, cfg=cfg, bs=bs, fixtures=fixtures)
