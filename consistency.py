"""Consistent performers vs one big game, and whether finishing is a skill.

Two questions, both answered from the archive, game by game:

1. SPIKES. The engine's rates are plain averages, so one big game moves them a
   lot - 79% of Bruno Fernandes' xG over GW1-5 of 2026-27 came in one match. Is
   a spike signal or noise? At every cutoff week, each method below estimates a
   player's rate from the games so far, and is scored on how well it predicts
   the same stat over the REST of that season:

     mean        total / minutes x 90                  (what the engine does)
     trim        the same, without the player's single best game
     median      the median game (games of 60+ minutes, per 90)
     capped      no single game counts above a cap (xG 0.8, xA 0.5, goals or
                 assists 1, points 10)
     mean x hit  the mean x the share of games with at least one goal (or
                 assist, or goal-or-assist for points). 5,0,0,0,0 scores
                 1.0 x 0.2; 1,1,1,1,1 scores 1.0 x 1.0.

   Methods are on different scales (mean x hit shrinks everyone), so each is
   first put on the target's scale by a straight-line fit on the OTHER seasons,
   then scored on the held-out season. Correlation is reported too, since it
   ignores scale altogether.

2. FINISHING. Does scoring above xG (or assisting above xA) in one stretch
   predict doing it again? Checked between halves of a season and between
   consecutive seasons, with xG per 90 alongside as a known signal.

3. ON TOP OF THE ENGINE. The engine projects from xG, xA and minutes, not from
   points, so 1 and 2 don't settle whether it is missing anything. This takes
   the engine's own projections from the backtest and asks whether the hit rate
   or assists-above-xA predict actual points better than the projection alone.

    python consistency.py            # parts 1 and 2 (seconds)
    python consistency.py engine     # part 3 (builds projections, ~1 min)
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get("BT_DATA", ROOT / "bt_data"))

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
CUTOFFS = range(4, 26)          # judge each method as it stands before these weeks
MIN_GAMES = 3                   # games played before the cutoff
MIN_REST = 450                  # rest-of-season minutes needed to score a row
CAPS = {"expected_goals": 0.8, "expected_assists": 0.5, "goals_scored": 1,
        "assists": 1, "total_points": 10}


def load(season):
    m = pd.read_csv(DATA / f"{season}_gws_merged_gw.csv")
    m = m[m["position"].isin(["DEF", "MID", "FWD"])]
    p = pd.read_csv(DATA / f"{season}_players_raw.csv").set_index("id")
    m["code"] = m["element"].map(p["code"])
    m["season"] = season
    keep = ["season", "element", "code", "position", "GW", "fixture", "minutes",
            "goals_scored", "assists", "expected_goals", "expected_assists", "total_points"]
    m = m[keep].copy()
    for k in ("expected_goals", "expected_assists"):
        m[k] = pd.to_numeric(m[k], errors="coerce").fillna(0.0)
    return m


# ---------------------------------------------------------------------
# 1. Spikes
# ---------------------------------------------------------------------
def estimates(mins, v, goals, assists, hit_stat, stat):
    """Every method's per-90 estimate from one player's games so far (arrays of
    the games he played in, oldest first)."""
    total_m = mins.sum()
    mean = v.sum() / total_m * 90
    best = int(np.argmax(v))
    rest_m = total_m - mins[best]
    trim = (v.sum() - v[best]) / rest_m * 90 if rest_m > 0 else mean
    full = mins >= 60
    median = float(np.median(v[full] / mins[full] * 90)) if full.any() else mean
    capped = np.minimum(v, CAPS[stat]).sum() / total_m * 90
    if hit_stat == "returns":
        hits = ((goals > 0) | (assists > 0)).mean()
    else:
        hits = ((goals if hit_stat == "goals_scored" else assists) > 0).mean()
    return {"mean": mean, "trim": trim, "median": median, "capped": capped,
            "mean x hit": mean * hits}


TARGETS = [
    # (target stat, estimating stat, stat whose "at least one" sets the hit rate)
    ("goals_scored", "expected_goals", "goals_scored"),
    ("goals_scored", "goals_scored", "goals_scored"),
    ("assists", "expected_assists", "assists"),
    ("assists", "assists", "assists"),
    ("total_points", "total_points", "returns"),
]


def spike_rows(all_games):
    """{(target, stat): rows} for every target, in one pass over the players."""
    out = {(t, s): [] for t, s, _ in TARGETS}
    cols = ["minutes", "goals_scored", "assists", "expected_goals",
            "expected_assists", "total_points"]
    for (season, el), g in all_games.groupby(["season", "element"], sort=False):
        g = g.sort_values(["GW", "fixture"])
        gw = g["GW"].to_numpy()
        a = {c: g[c].to_numpy(dtype=float) for c in cols}
        for t in CUTOFFS:
            b, r = gw < t, gw >= t
            pl = b & (a["minutes"] > 0)
            if pl.sum() < MIN_GAMES:
                continue
            rest_m = a["minutes"][r].sum()
            if rest_m < MIN_REST:
                continue
            for target, stat, hit in TARGETS:
                e = estimates(a["minutes"][pl], a[stat][pl], a["goals_scored"][pl],
                              a["assists"][pl], hit, stat)
                e.update(season=season, t=t, w=rest_m,
                         target=a[target][r].sum() / rest_m * 90)
                out[(target, stat)].append(e)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def score(d, method):
    """Held-out RMSE after a straight-line fit on the other seasons, and correlation."""
    se, wt = 0.0, 0.0
    for s in d["season"].unique():
        tr, te = d[d["season"] != s], d[d["season"] == s]
        b, a = np.polyfit(tr[method], tr["target"], 1, w=np.sqrt(tr["w"]))
        pred = a + b * te[method]
        se += (te["w"] * (pred - te["target"]) ** 2).sum()
        wt += te["w"].sum()
    return np.sqrt(se / wt), wcorr(d[method].to_numpy(), d["target"].to_numpy(), d["w"].to_numpy())


def part_spikes(all_games):
    print("1. SPIKES: which average best predicts the rest of the season?\n")
    methods = ["mean", "trim", "median", "capped", "mean x hit"]
    rows = spike_rows(all_games)
    for target, stat, hit in TARGETS:
        d = rows[(target, stat)]
        res = {m: score(d, m) for m in methods}
        base = res["mean"][0]
        label = f"{target} from {stat}"
        print(f"  {label}   ({len(d)} player-weeks)")
        print(f"    {'method':<12}{'held-out RMSE':>15}{'vs mean':>10}{'corr':>8}")
        for m in methods:
            r, c = res[m]
            print(f"    {m:<12}{r:>15.4f}{(r / base - 1) * 100:>+9.1f}%{c:>8.3f}")
        # by season, so one season can't carry the result
        per = []
        for s in sorted(d["season"].unique()):
            ds = d[d["season"] == s]
            per.append(f"{s[-5:]} " + " ".join(
                f"{m.split()[0][:4]}={wcorr(ds[m].to_numpy(), ds['target'].to_numpy(), ds['w'].to_numpy()):.3f}"
                for m in methods))
        print("    corr by season: " + "\n                    ".join(per) + "\n")


# ---------------------------------------------------------------------
# 2. Finishing
# ---------------------------------------------------------------------
def totals(games, by):
    t = games.groupby(by)[["minutes", "goals_scored", "expected_goals",
                           "assists", "expected_assists"]].sum()
    t["over_g90"] = (t["goals_scored"] - t["expected_goals"]) / t["minutes"] * 90
    t["over_a90"] = (t["assists"] - t["expected_assists"]) / t["minutes"] * 90
    t["xg90"] = t["expected_goals"] / t["minutes"] * 90
    t["xa90"] = t["expected_assists"] / t["minutes"] * 90
    return t


def persist(a, b, min_minutes, label):
    j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    j = j[(j["minutes_1"] >= min_minutes) & (j["minutes_2"] >= min_minutes)]
    out = [f"  {label:<44} n={len(j):>4}"]
    for k, name in (("over_g90", "goals - xG"), ("over_a90", "assists - xA"),
                    ("xg90", "xG (benchmark)"), ("xa90", "xA (benchmark)")):
        out.append(f"{name} {np.corrcoef(j[k + '_1'], j[k + '_2'])[0, 1]:+.2f}")
    print("   ".join(out))


def part_finishing(all_games):
    print("2. FINISHING: does beating xG / xA carry over? (correlation, + = it persists)\n")
    for s in SEASONS:
        g = all_games[all_games["season"] == s]
        persist(totals(g[g["GW"] <= 19], "code"), totals(g[g["GW"] >= 20], "code"),
                900, f"{s}: first half vs second half (900+ min)")
    for s1, s2 in zip(SEASONS, SEASONS[1:]):
        persist(totals(all_games[all_games["season"] == s1], "code"),
                totals(all_games[all_games["season"] == s2], "code"),
                1500, f"{s1} vs {s2} (1500+ min each)")
    # two seasons pooled against the next, for the biggest samples
    for s1, s2, s3 in zip(SEASONS, SEASONS[1:], SEASONS[2:]):
        a = totals(all_games[all_games["season"].isin([s1, s2])], "code")
        persist(a, totals(all_games[all_games["season"] == s3], "code"),
                2500, f"{s1[:4]}-{s2[-2:]} combined vs {s3} (2500+ min each)")
    print()


# ---------------------------------------------------------------------
# 3. On top of the engine
# ---------------------------------------------------------------------
ENGINE_SEASONS = ["2023-24", "2024-25", "2025-26"]    # 2022-23 has no fixtures file
AHEAD = 6
HIT_PRIOR_GAMES = 5


def engine_rows():
    os.environ.setdefault("BT_DATA", str(ROOT / "bt_data"))
    os.environ.setdefault("BT_CACHE", str(ROOT / "bt_cache"))
    sys.path.insert(0, str(ROOT))
    import run_backtest as R
    rows = []
    for season in ENGINE_SEASONS:
        pj = R.projector_for(season)
        src = pj.source
        games = load(season)
        by_el = {el: g.sort_values(["GW", "fixture"]) for el, g in games.groupby("element")}
        for t in CUTOFFS:
            df = pj.data[t][0].df
            df = df[df["can_select"] & df["pos"].isin([2, 3, 4])]
            for i, row in df.iterrows():
                g = by_el.get(i)
                if g is None:
                    continue
                b = g[(g["GW"] < t) & (g["minutes"] > 0)]
                if len(b) < MIN_GAMES:
                    continue
                act1 = src.actual.get((i, t))
                if act1 is None:
                    continue
                ws = [w for w in range(t, min(t + AHEAD, 39)) if f"gw{w}" in df.columns]
                acts = [src.actual.get((i, w)) for w in ws]
                hit = ((b["goals_scored"] > 0) | (b["assists"] > 0)).mean()
                n_games = len(b)
                mins = b["minutes"].sum()
                rows.append({
                    "season": season, "t": t, "pos": int(row["pos"]),
                    "proj1": row[f"gw{t}"], "act1": float(act1),
                    "proj6": sum(row[f"gw{w}"] for w in ws) if len(ws) == AHEAD else np.nan,
                    "act6": float(sum(acts)) if len(ws) == AHEAD and all(a is not None for a in acts) else np.nan,
                    "hit": hit, "n": n_games,
                    "over_a90": (b["assists"].sum() - b["expected_assists"].sum()) / mins * 90,
                })
    return pd.DataFrame(rows)


ENGINE_MODELS = {
    "projection only": ["P"],
    "+ hit rate": ["P", "P*hit"],
    "+ hit vs position": ["P", "P*hit_pos"],
    "+ hit vs pos, steadied": ["P", "P*hit_shr"],
    "+ position only": ["P", "P*mid", "P*fwd"],
    "+ assists above xA": ["P", "over_a"],
    "+ both": ["P", "P*hit", "over_a"],
}


def design(d, terms, horizon):
    P = d[f"proj{horizon}"].to_numpy()
    cols = {"P": P, "P*hit": P * (d["hit"].to_numpy() - 0.3),
            "P*hit_pos": P * (d["hit"] - d["pos_hit"]).to_numpy(),
            "P*hit_shr": P * (d["hit_shr"] - d["pos_hit"]).to_numpy(),
            "P*mid": P * (d["pos"] == 3).to_numpy(), "P*fwd": P * (d["pos"] == 4).to_numpy(),
            "over_a": d["over_a90"].to_numpy() * (horizon if horizon > 1 else 1)}
    return np.column_stack([np.ones(len(d))] + [cols[k] for k in terms])


def part_engine():
    print("3. ON TOP OF THE ENGINE: do they predict actual points better than the "
          "projection alone?", end="\n\n")
    d = engine_rows()
    # each player's hit rate against his position's, at the same point of the
    # same season; "steadied" adds HIT_PRIOR_GAMES of the position average
    d["pos_hit"] = d.groupby(["season", "t", "pos"])["hit"].transform("mean")
    d["hit_shr"] = (d["n"] * d["hit"] + HIT_PRIOR_GAMES * d["pos_hit"]) / (d["n"] + HIT_PRIOR_GAMES)
    for horizon in (1, AHEAD):
        dd = d.dropna(subset=[f"proj{horizon}", f"act{horizon}"])
        y = dd[f"act{horizon}"].to_numpy()
        print(f"  next {'week' if horizon == 1 else f'{AHEAD} weeks'}   ({len(dd)} player-weeks)")
        print(f"    {'model':<22}{'held-out RMSE':>15}{'vs proj':>10}   per season (RMSE change)   coefficients")
        base = None
        base_s = {}
        for name, terms in ENGINE_MODELS.items():
            se, per = 0.0, {}
            for s_ in ENGINE_SEASONS:
                tr, te = dd["season"] != s_, dd["season"] == s_
                X = design(dd, terms, horizon)
                beta, *_ = np.linalg.lstsq(X[tr.to_numpy()], y[tr.to_numpy()], rcond=None)
                err = (X[te.to_numpy()] @ beta - y[te.to_numpy()]) ** 2
                se += err.sum()
                per[s_] = np.sqrt(err.mean())
            rmse = np.sqrt(se / len(dd))
            beta_all, *_ = np.linalg.lstsq(design(dd, terms, horizon), y, rcond=None)
            if base is None:
                base, base_s = rmse, per
            cells = " ".join(f"{s_[-5:]} {(per[s_] / base_s[s_] - 1) * 100:+.2f}%" for s_ in ENGINE_SEASONS)
            coef = " ".join(f"{k}={b:+.3f}" for k, b in zip(["1"] + terms, beta_all))
            print(f"    {name:<22}{rmse:>15.4f}{(rmse / base - 1) * 100:>+9.2f}%   {cells}   {coef}")
        print()


def main():
    if sys.argv[1:] == ["engine"]:
        part_engine()
        return
    all_games = pd.concat([load(s) for s in SEASONS], ignore_index=True)
    part_spikes(all_games)
    part_finishing(all_games)


if __name__ == "__main__":
    main()
