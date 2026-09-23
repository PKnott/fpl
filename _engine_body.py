# 2. SETUP
# ---------------------------------------------------------------------
import math, sys, subprocess, unicodedata, itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor
try:
    import pulp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pulp"])
    import pulp
import requests
import pandas as pd

BASE = "https://fantasy.premierleague.com/api/"
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POS_KEY = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
DC_THRESHOLD = {2: 10, 3: 12, 4: 12}
CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit",
              "bboost": "Bench Boost", "3xc": "Triple Captain"}
COMPONENTS = ["Playing time", "Goals", "Assists", "Clean sheet", "Goals conceded",
              "Saves", "Def. contributions", "Bonus", "Yellow cards"]
STAT_KEYS = ["expected_goals", "goals_scored", "expected_assists", "assists",
             "bonus", "saves", "defensive_contribution", "yellow_cards"]


def fetch(path):
    r = requests.get(BASE + path, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
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


EXPLAIN = {}      # filled by build_projections: per-player breakdown for explain()
LAST_RUN = {}     # filled by run(): lets explain() work after a report


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


def fetch_last_season(ids):
    return fetch_player_pages(ids)[0]


def use_last_season(cfg):
    """(stats_on, starts_on). Older settings with a single LAST_SEASON switch still work."""
    old = cfg.get("LAST_SEASON")
    if old is False:                       # old master switch off = everything off
        return False, False
    stats = cfg.get("LAST_SEASON_STATS", True)
    starts = cfg.get("LAST_SEASON_STARTS", bool(old))   # old "True" also meant starts on
    return bool(stats), bool(starts)


# ---------------------------------------------------------------------
# 5. PLAYER PROJECTIONS
# ---------------------------------------------------------------------
def build_projections(bs, fixtures, cfg, next_gw, past=None):
    past = past or {}
    ts = team_strength(bs, fixtures, cfg)
    sc = bs.get("game_config", {}).get("scoring", {})
    goal_pts = sc.get("goals_scored", {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4})
    cs_pts = sc.get("clean_sheets", {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})
    dc_pts = sc.get("defensive_contribution", {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2})
    ast_pts = sc.get("assists", 3)
    w = cfg["XG_WEIGHT"]
    gws = list(range(next_gw, min(next_gw + cfg["LONG_VIEW"], 39)))

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
    stats_on, starts_on = use_last_season(cfg)
    K_LAST = cfg.get("LAST_SEASON_MINUTES", 540) if stats_on else 0
    M0_ALL = cfg.get("PRIOR_MINUTES", 270)
    M0_GK = cfg.get("GK_PRIOR_MINUTES") or M0_ALL
    G_LAST = cfg.get("LAST_SEASON_STARTS_GAMES", 3)
    exact_saves = cfg.get("EXACT_SAVES", False)
    m_decay = cfg.get("MINUTES_DECAY")
    rows = []
    for e in bs["elements"]:
        pt, t, mins = e["element_type"], e["team"], e["minutes"]
        M0 = M0_GK if pt == 1 else M0_ALL
        n = max(ts["played"].get(t, 0), 1)
        last = past.get(e["id"]) if (stats_on or starts_on) else None
        last_mins = last.get("minutes", 0) if last else 0

        rate_info = {}

        def rate(key):
            # this season + position average (always) + last season (if on, added on top)
            obs = per90(e, key, mins)
            num, den = mins * obs + M0 * priors[pt][key], mins + M0
            info = {"this": obs, "this_w": mins, "prior": priors[pt][key], "prior_w": M0,
                    "last": None, "last_w": 0}
            if last and K_LAST and key in last:
                info["last"], info["last_w"] = per90(last, key, last_mins), K_LAST
                num += K_LAST * info["last"]
                den += K_LAST
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
        EXPLAIN[e["id"]] = {"pos": pt, "start_rate": start_rate, "starts": e.get("starts", 0),
                            "p60": p60, "p_any": p_any, "minutes_method": minutes_method,
                            "recent": [g["minutes"] for g in recent][::-1],
                            "team_games": n, "mpg": mpg, "p60": p60, "rates": rate_info,
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
    df["xp_long"] = sum(df[f"gw{g}"] * cfg["DECAY"] ** i for i, g in enumerate(gws))
    unscheduled = [fx for fx in fixtures if fx.get("event") is None]
    return df, gws, ts, fx_map, unscheduled


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


def chip_window(bs, chip, gw):
    for c in bs.get("chips", []):
        if c["name"] == chip and c["start_event"] <= gw <= c["stop_event"]:
            return c["stop_event"]
    return None if bs.get("chips") else 38


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
    B = pulp.LpVariable.dicts
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
        for p, need in {1: 2, 2: 5, 3: 5, 4: 3}.items():
            m += pulp.lpSum(x[i, g] for i in pool if pos[i] == p) == need
        for tm in set(team.values()):
            m += pulp.lpSum(x[i, g] for i in pool if team[i] == tm) <= 3
        m += pulp.lpSum(y[i, g] for i in pool) == 11
        m += pulp.lpSum(c[i, g] for i in pool) == 1
        for p, (lo, hi) in {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}.items():
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


MIN_IN_XI = {1: 1, 2: 3, 3: 2, 4: 1}   # FPL formation minimums (autosubs must respect them)


def play_chance(df, i, gw):
    """Chance the player gets ANY minutes (only 0 minutes triggers an autosub)."""
    e = EXPLAIN.get(i)
    if not e or gw not in e["gws"] or not e["gws"][gw]["fixtures"]:
        return 0.0
    return max(0.0, min(1.0, e["gws"][gw]["availability"] * e.get("p_any", 1.0)))


def pick_lineup(df, squad, gw, sims=4000, seed=1):
    """Choose XI, bench order and captain to maximise expected points INCLUDING autosubs.
    Each player plays (any minutes) with his play chance; if he plays he scores his
    points-if-he-plays. Non-players are replaced from the bench in order, keeping a legal
    formation; the vice gets the armband if the captain doesn't play."""
    col = f"gw{gw}"
    squad = list(squad)
    pos = {i: df.loc[i, "pos"] for i in squad}
    p = {i: play_chance(df, i, gw) for i in squad}
    v = {i: (df.loc[i, col] / p[i] if p[i] > 0.02 else 0.0) for i in squad}   # points if he plays
    rng = np.random.default_rng(seed)
    idx = {i: k for k, i in enumerate(squad)}
    plays = rng.random((sims, len(squad))) < np.array([p[i] for i in squad])
    vals = np.array([v[i] for i in squad])

    def score(xi, bench, cap, vice):
        xi_k = [idx[i] for i in xi]
        bench_k = [idx[i] for i in bench]
        total = np.zeros(sims)
        for s_ in range(sims):
            pl = plays[s_]
            on = [k for k in xi_k if pl[k]]
            count = {q: sum(1 for k in on if pos[squad[k]] == q) for q in POS}
            missing = [k for k in xi_k if not pl[k]]
            n_missing_out = sum(1 for k in missing if pos[squad[k]] != 1)
            # goalkeeper autosub
            if any(pos[squad[k]] == 1 for k in missing):
                gk_b = bench_k[0]
                if pl[gk_b]:
                    on.append(gk_b)
            # outfield autosubs in bench order, respecting formation minimums
            for b in bench_k[1:]:
                if n_missing_out == 0:
                    break
                if not pl[b]:
                    continue
                q = pos[squad[b]]
                need = {r: max(0, MIN_IN_XI[r] - count[r]) for r in (2, 3, 4)}
                spare = n_missing_out - sum(need.values())
                if need.get(q, 0) > 0 or spare > 0:
                    on.append(b); count[q] += 1; n_missing_out -= 1
            pts = vals[on].sum()
            c = idx[cap] if pl[idx[cap]] else (idx[vice] if pl[idx[vice]] else None)
            total[s_] = pts + (vals[c] if c is not None else 0)
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
    return {"xi": xi, "cap": cap, "vice": vice, "bench": bench, "total": exp_total,
            "if_plays": v, "play_chance": p,
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
    for p, need in {1: 2, 2: 5, 3: 5, 4: 3}.items():
        m += pulp.lpSum(x[i] for i in pool if df.loc[i, "pos"] == p) == need
    for t in set(df.loc[pool, "team_id"]):
        m += pulp.lpSum(x[i] for i in pool if df.loc[i, "team_id"] == t) <= 3
    m += pulp.lpSum(df.loc[i, "price"] * x[i] for i in pool) <= budget
    m += pulp.lpSum(y.values()) == 11
    m += pulp.lpSum(c.values()) == 1
    for p, (lo, hi) in {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}.items():
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
# 9. CHIP TIMING
# ---------------------------------------------------------------------
def chip_advice(df, gws, bs, chips, plan, squad, bank, sell, cfg, fx_map, unscheduled, pool):
    g0 = gws[0]
    squad_at = {w["gw"]: w["squad"] for w in plan["weeks"]}
    last_squad = plan["weeks"][-1]["squad"]
    sq = lambda g: squad_at.get(g, last_squad)
    budget = bank + sum(sell.values())
    results = {}

    def window(chip):
        stop = chip_window(bs, chip, g0)
        return stop, [g for g in gws if stop and g <= stop]

    def verdict(values, minimum, stop, higher_is_better=True):
        # later weeks are less certain (injuries, form, transfers): discount waiting
        wait = cfg.get("CHIP_WAIT_DECAY", 0.98)
        eff = {g: values[g] * wait ** (g - g0) for g in values}
        best_g = max(eff, key=eff.get)
        now = values[g0]
        beyond = stop and stop > gws[-1]
        last_chance = stop == g0
        # an unused chip is worth nothing: once the whole window is visible, just use
        # the best week - the minimum only applies while later weeks are out of view
        visible = bool(stop) and stop <= gws[-1]
        min_ok = now >= minimum or (visible and cfg.get("CHIP_IGNORE_MIN_IF_VISIBLE", True))
        play = (min_ok and now >= cfg["CHIP_RATIO"] * eff[best_g]) or (last_chance and now > 0)
        return play, best_g, now, beyond

    def weak(now, minimum, unit):
        if cfg.get("CHIP_IGNORE_MIN_IF_VISIBLE", True):
            return f"save. Best week in view is now, but later weeks outside the view may be better."
        return (f"save. This is the best week in view, but only {now:.1f} {unit} "
                f"(want {minimum}+ - usually a double or blank gameweek).")

    for chip in chips:
        stop, win = window(chip)
        if not win:
            results[chip] = (False, 0, "not usable this gameweek.")
            continue
        if chip == "3xc":
            vals = {g: best_xi(df, sq(g), g)["cap_xp"] for g in win}
            play, bg, now, beyond = verdict(vals, cfg["TC_MIN"], stop)
            cap = df.loc[best_xi(df, sq(bg), bg)["cap"], "name"]
            msg = (f"PLAY on {df.loc[best_xi(df, sq(g0), g0)['cap'], 'name']} ({now:.1f} xP)."
                   if play else weak(now, cfg["TC_MIN"], "captain xP") if bg == g0 else
                   f"save. Best week: GW{bg} ({cap}, {vals[bg]:.1f} xP) vs {now:.1f} now.")
            results[chip] = (play, now, msg)
        elif chip == "bboost":
            vals = {g: best_xi(df, sq(g), g)["bench_pts"] for g in win}
            play, bg, now, beyond = verdict(vals, cfg["BBOOST_MIN"], stop)
            msg = (f"PLAY - bench projects {now:.1f} pts." if play else
                   weak(now, cfg["BBOOST_MIN"], "bench pts") if bg == g0 else
                   f"save. Best week: GW{bg} (bench {vals[bg]:.1f}) vs {now:.1f} now.")
            results[chip] = (play, now, msg)
        elif chip == "freehit":
            vals, fh_squads = {}, {}
            for g in win:
                fh, fh_pts = free_hit_squad(df, g, budget, cfg)
                vals[g] = fh_pts - best_xi(df, sq(g), g)["total"]
                fh_squads[g] = fh
            play, bg, now, beyond = verdict(vals, cfg["FREEHIT_THRESHOLD"], stop)
            msg = (f"PLAY - a one-week squad gains {now:+.1f} pts." if play else
                   weak(now, cfg["FREEHIT_THRESHOLD"], "pts gain") if bg == g0 else
                   f"save. Best week: GW{bg} ({vals[bg]:+.1f} pts) vs {now:+.1f} now.")
            results[chip] = (play, now, msg, fh_squads.get(g0))
        elif chip == "wildcard":
            vals, plans = {}, {}
            for k, g in enumerate(plan["weeks"][:3]):
                if g["gw"] not in win:
                    continue
                p = plan_transfers(df, gws, squad, bank, sell, cfg, wc_week=k, pool=pool)
                if p:
                    vals[g["gw"]] = p["objective"] - plan["objective"]
                    plans[g["gw"]] = p
            if not vals:
                results[chip] = (False, 0, "couldn't evaluate.")
                continue
            bg = max(vals, key=vals.get)
            now = vals.get(g0, 0)
            play = bg == g0 and now >= cfg["WILDCARD_THRESHOLD"]
            msg = (f"PLAY - rebuilding gains {now:+.1f} weighted pts." if play else
                   f"save. Best timing: GW{bg} ({vals[bg]:+.1f} pts) vs {now:+.1f} now "
                   f"(needs +{cfg['WILDCARD_THRESHOLD']}).")
            results[chip] = (play, now, msg, plans.get(g0))
        if stop and stop - g0 <= 3:
            r = results[chip]
            results[chip] = (r[0], r[1], r[2] + f"  ⚠ Expires after GW{stop}!", *r[3:])
        if stop and stop > gws[-1]:
            r = results[chip]
            results[chip] = (r[0], r[1], r[2] + f" (can't see beyond GW{gws[-1]}; "
                             f"chip lasts to GW{stop})", *r[3:])

    # only one chip per gameweek: keep the most valuable
    playing = [c for c in results if results[c][0]]
    if len(playing) > 1:
        keep = max(playing, key=lambda c: results[c][1])
        for c in playing:
            if c != keep:
                r = results[c]
                results[c] = (False, r[1], f"save - {CHIP_NAMES[keep]} is worth more this week.",
                              *r[3:])
    return results


# ---------------------------------------------------------------------
# 9b. POINTS BREAKDOWN
# ---------------------------------------------------------------------
RATE_LABELS = {"saves": "Saves", "bonus": "Bonus", "expected_goals": "xG",
               "expected_assists": "xA", "defensive_contribution": "Def. actions"}
RATES_BY_POS = {1: ["saves", "bonus"],
                2: ["expected_goals", "expected_assists", "defensive_contribution", "bonus"],
                3: ["expected_goals", "expected_assists", "defensive_contribution", "bonus"],
                4: ["expected_goals", "expected_assists", "bonus"]}


def explain(*names, gw=None):
    """Show where each player's projected points come from. Run the report first.
    Example: explain("Verbruggen", "Trafford")"""
    if not LAST_RUN:
        print("Run the report first: df, plan, lineup, chips = run(CONFIG)")
        return
    df, gws = LAST_RUN["df"], LAST_RUN["gws"]
    gw = gw or gws[0]
    ids = [find_player(df, n) for n in names]
    W = 16
    head = lambda label, vals: print(f"{label:<26}" + "".join(f"{v:>{W}}" for v in vals))
    print(f"\nPOINTS BREAKDOWN - GW{gw}\n" + "=" * (26 + W * len(ids)))
    head("", [df.loc[i, "name"] for i in ids])
    fx = []
    for i in ids:
        f_ = EXPLAIN[i]["gws"][gw]["fixtures"]
        fx.append(" + ".join(f"{x['opp']} ({'H' if x['home'] else 'A'})" for x in f_) or "BLANK")
    head("Fixture", fx)
    head("Starts this season", [f"{EXPLAIN[i]['starts']} of {EXPLAIN[i]['team_games']}" for i in ids])
    head("Last 5 games (newest 1st)", [",".join(str(int(m)) for m in EXPLAIN[i]["recent"][:5]) or "-"
                                       for i in ids])
    head("Chance of starting", [f"{EXPLAIN[i]['start_rate']:.0%}" for i in ids])
    head("Chance of 60+ minutes", [f"{EXPLAIN[i]['p60']:.0%}" for i in ids])
    head("Expected minutes", [f"{EXPLAIN[i]['mpg']:.0f}" for i in ids])
    head("Availability", [f"{EXPLAIN[i]['gws'][gw]['availability']:.0%}" for i in ids])
    print("-" * (26 + W * len(ids)))
    for k in COMPONENTS:
        vals = [EXPLAIN[i]["gws"][gw]["components"][k] for i in ids]
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
            f_ = EXPLAIN[i]["gws"][gw]["fixtures"]
            vals.append(fmt.format(f_[0][key]) if f_ else "-")
        if key == "exp_saves" and all(EXPLAIN[i]["pos"] != 1 for i in ids):
            continue
        head(label, vals)

    print("  * goals conceded vs league average: above 100% = leakier than average")
    print("\nWhere each per-90 rate comes from (this season | last season | position avg -> used):")
    for i in ids:
        d = EXPLAIN[i]
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


def team_table(cfg=None):
    """Every team's defence rating next to what actually happened. Run the report first.
    Rating: 100% = league average, higher = leakier."""
    if not LAST_RUN:
        print("Run the report first: df, plan, lineup, chips = run(CONFIG)")
        return
    cfg = cfg or CONFIG
    ts = team_strength(LAST_RUN["bs"], LAST_RUN["fixtures"], cfg)
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
def run(cfg=CONFIG, data=None, past=None):
    if data is None:
        bs, fixtures = fetch("bootstrap-static/"), fetch("fixtures/")
    else:
        bs, fixtures = data
    next_gw = next(e["id"] for e in bs["events"] if e.get("is_next"))
    df, gws, ts, fx_map, unscheduled = build_projections(bs, fixtures, cfg, next_gw)
    squad, bank, sell, chips = load_team(bs, cfg, next_gw, df)
    for i in squad:
        if sell[i] is None:
            sell[i] = df.loc[i, "price"]

    # second pass: last season + recent game-by-game minutes for players that matter
    if any(use_last_season(cfg)) or cfg.get("MINUTES_DECAY"):
        if past is None:
            ids, _ = solver_pool(df, squad, cfg, per_pos=45)
            print(f"Loading player histories for {len(ids)} players...")
            past, recent = fetch_player_pages(ids)
            for e in bs["elements"]:
                if e["id"] in recent:
                    e["recent"] = recent[e["id"]]
        df, gws, ts, fx_map, unscheduled = build_projections(bs, fixtures, cfg, next_gw, past)

    g0 = gws[0]
    LAST_RUN.update(df=df, gws=gws, ts=ts, bs=bs, fixtures=fixtures)
    nm = lambda i: f"{df.loc[i, 'name']} ({df.loc[i, 'team']}, £{df.loc[i, 'price']:.1f}m)"
    L = "=" * 62

    def fixture_str(tid, gw):
        fl = fx_map.get((tid, gw), [])
        return " + ".join(f"{ts['short'][o]} ({'H' if h else 'A'})" for o, h in fl) or "BLANK"

    print(L)
    print(f" FPL ASSISTANT - GW{g0}   plan GW{g0}-{g0 + cfg['PLAN_WEEKS'] - 1}, "
          f"long view to GW{gws[-1]}")
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

    # ---- plan transfers (all options), then time chips against the best plan
    pool, _ = solver_pool(df, squad, cfg)
    plans = {}
    for n in range(0, cfg["MAX_TRANSFERS"] + 1):
        p = plan_transfers(df, gws, squad, bank, sell, cfg, n_first=n, pool=pool)
        if p:
            plans[n] = p
    best_n = max(plans, key=lambda n: plans[n]["objective"])
    plan = plans[best_n]
    w0 = plan["weeks"][0]
    advice = chip_advice(df, gws, bs, chips, plan, squad, bank, sell, cfg,
                         fx_map, unscheduled, pool) if chips else {}
    active = next((c for c, r in advice.items() if r[0]), None)

    # ---- chips
    print(f"\n{L}\n CHIPS  (compared across every week of each chip's window)\n{L}")
    if not chips:
        print("  No chips available.")
    for chip, r in advice.items():
        print(f"  {CHIP_NAMES[chip]}: {r[2]}")
    if unscheduled:
        teams = sorted({ts["short"][t] for fx in unscheduled for t in (fx["team_h"], fx["team_a"])})
        print(f"  ℹ {len(unscheduled)} fixture(s) not yet scheduled ({', '.join(teams)}). "
              "These usually become double gameweeks - a reason to hold Bench Boost / "
              "Triple Captain if nothing stands out.")

    # ---- transfers
    print(f"\n{L}\n TRANSFERS\n{L}")
    lineup_squad = w0["squad"]
    if active == "wildcard" and advice["wildcard"][3]:
        wp = advice["wildcard"][3]
        wk = wp["weeks"][0]
        lineup_squad = wk["squad"]
        print(f">>> PLAY WILDCARD - make these moves (no hits):")
        for o, i in pair_moves(df, wk["out"], wk["in"]):
            print(f"   OUT {nm(o):<32} IN {nm(i)}")
        print(f"   Bank after: £{wk['bank']:.1f}m")
        plan = wp
    elif active == "freehit" and advice["freehit"][3]:
        lineup_squad = advice["freehit"][3]
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
    lu = pick_lineup(df, lineup_squad, g0)
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
    print(f"\n{L}\n VALUE vs POSITION AVERAGE  (projected pts per £m, {len(gws)} weeks)\n{L}")
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
    return df, plan, lu, advice
