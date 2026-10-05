"""When does a player's form become signal? Measured per stat from the archive.

The engine projects each per-90 rate as a minutes-weighted blend:

    rate = (this-season minutes x this-season rate
            + LAST_SEASON_MINUTES x last-season rate
            + PRIOR_MINUTES x position average) / (sum of the three weights)

so the two settings say how many minutes of this season's data the past is worth.
They are one value for every stat. This script asks the data instead: for every
player, at every cutoff week, how well does each choice of the two weights predict
the same stat over the REST of that season? Fitted per stat, scored by
leave-one-season-out, so a setting is never judged on the season it was fitted to.

    python signal_noise.py
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("BT_DATA", str(ROOT / "bt_data"))
os.environ.setdefault("BT_CACHE", str(ROOT / "bt_cache"))
sys.path.insert(0, str(ROOT))

import fpl_backtest as B                                         # noqa: E402

SEASONS = ["2023-24", "2024-25", "2025-26"]
CUTOFFS = range(2, 21)          # judge the blend as it stands before these weeks
MIN_REST = 450                  # rest-of-season minutes needed to score a row
# (stat, positions it is scored for). Saves are a keeper stat; defensive
# contributions only exist from 2025-26, with no previous season to blend.
STATS = [("expected_goals", (2, 3, 4)), ("goals_scored", (2, 3, 4)),
         ("expected_assists", (2, 3, 4)), ("assists", (2, 3, 4)),
         ("bonus", (2, 3, 4)), ("bonus", (1,)), ("saves", (1,)),
         ("yellow_cards", (2, 3, 4)), ("defensive_contribution", (2, 3, 4))]
K_GRID = np.array([0, 90, 180, 270, 360, 540, 720, 900, 1200, 1800, 2700, 3600, 5400, 8000])
M_GRID = np.array([0, 45, 90, 180, 270, 360, 540, 900, 1350, 2000, 3000, 4500, 6750, 10000])
CURRENT = {"outfield": (540, 90), "keeper": (540, 360)}


def team_names(season):
    """Each player's club name at the end of `season` (by player code)."""
    m = pd.read_csv(B.DATA / f"{season}_gws_merged_gw.csv")
    p = pd.read_csv(B.DATA / f"{season}_players_raw.csv").set_index("id")
    last = m.sort_values("GW").groupby("element")["team"].last()
    return {int(p.loc[i, "code"]): t for i, t in last.items() if i in p.index}


def rows_for(season, stat, positions):
    """One row per (player, cutoff): data so far, last season, and the target."""
    src = B.ArchiveSource(season)
    gw = src.gw
    if stat not in gw or gw[stat].abs().sum() == 0:
        return None
    # player x gameweek matrices, so every cutoff is a cumulative sum
    mins = gw.pivot_table(index="element", columns="GW", values="minutes",
                          aggfunc="sum", fill_value=0)
    vals = gw.pivot_table(index="element", columns="GW", values=stat,
                          aggfunc="sum", fill_value=0).reindex_like(mins).fillna(0)
    pids = mins.index.to_numpy()
    pos = src.players["element_type"].reindex(pids).to_numpy()
    cm, cv = mins.cumsum(axis=1).to_numpy(), vals.cumsum(axis=1).to_numpy()
    tot_m, tot_v = cm[:, -1], cv[:, -1]
    cols = list(mins.columns)

    past = src.past
    has_last = np.array([bool(past.get(p) and stat in past[p] and past[p]["minutes"] > 0)
                         for p in pids])
    last90 = np.array([past[p][stat] / past[p]["minutes"] * 90 if h else 0.0
                       for p, h in zip(pids, has_last)])
    # changed club since last season: last season's final club vs this season's first
    prev_team = team_names(src.prev_season)
    first_team = src.m.sort_values("GW").groupby("element")["team"].first()
    code = src.players["code"].reindex(pids)
    moved = np.array([h and prev_team.get(int(c)) is not None
                      and prev_team.get(int(c)) != first_team.get(p)
                      for p, c, h in zip(pids, code.fillna(-1), has_last)])

    out = []
    for t in CUTOFFS:
        k = sum(1 for c in cols if c < t)            # weeks of data before t
        m = cm[:, k - 1] if k else np.zeros(len(pids))
        v = cv[:, k - 1] if k else np.zeros(len(pids))
        rest_m, rest_v = tot_m - m, tot_v - v
        # The engine's prior exactly: 0.6 x the mean per-90 of players with 270+
        # minutes before t (zero when nobody has yet, i.e. the first weeks)
        pos90 = {}
        for p_ in (1, 2, 3, 4):
            sel = (pos == p_) & (m >= 270)
            pos90[p_] = 0.6 * float(np.mean(v[sel] / m[sel] * 90)) if sel.any() else 0.0
        keep = np.isin(pos, positions) & (rest_m >= MIN_REST)
        for i in np.where(keep)[0]:
            out.append((season, t, int(pos[i]), m[i], v[i] / m[i] * 90 if m[i] else 0.0,
                        has_last[i], last90[i], moved[i], pos90[int(pos[i])],
                        rest_v[i] / rest_m[i] * 90, rest_m[i]))
    return pd.DataFrame(out, columns=["season", "t", "pos", "m", "this90", "has_last", "last90",
                                      "moved", "pos90", "target90", "w"])


def sq_error(d, K, M0):
    """Rest-of-season-minutes-weighted squared error for every (K, M0) on the grid."""
    m = d["m"].to_numpy()[:, None, None]
    k = np.where(d["has_last"].to_numpy(), 1.0, 0.0)[:, None, None] * K[None, :, None]
    mo = M0[None, None, :]
    num = m * d["this90"].to_numpy()[:, None, None] + k * d["last90"].to_numpy()[:, None, None] \
        + mo * d["pos90"].to_numpy()[:, None, None]
    den = m + k + mo
    pred = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    # with no data and no prior, fall back to the position average
    pred = np.where(den > 0, pred, d["pos90"].to_numpy()[:, None, None])
    err = (pred - d["target90"].to_numpy()[:, None, None]) ** 2
    w = d["w"].to_numpy()[:, None, None]
    return (err * w).sum(axis=0) / w.sum()


def fit(d):
    e = sq_error(d, K_GRID, M_GRID)
    i, j = np.unravel_index(np.argmin(e), e.shape)
    return int(K_GRID[i]), int(M_GRID[j]), e


def main():
    print(__doc__.split("\n\n")[0], "\n")
    summary = []
    for stat, positions in STATS:
        group = "keeper" if positions == (1,) else "outfield"
        frames = [rows_for(s, stat, positions) for s in SEASONS]
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        seasons = sorted(d["season"].unique())
        K_all, M_all, e_all = fit(d)
        cur_K, cur_M = CURRENT[group]
        ci, cj = list(K_GRID).index(cur_K), list(M_GRID).index(cur_M)
        # leave-one-season-out: fit on the others, score on the held-out season
        loso_fit = loso_cur = 0.0
        if len(seasons) > 1:
            for s in seasons:
                tr, te = d[d["season"] != s], d[d["season"] == s]
                K, M, _ = fit(tr)
                e_te = sq_error(te, K_GRID, M_GRID)
                wt = te["w"].sum()
                loso_fit += e_te[list(K_GRID).index(K), list(M_GRID).index(M)] * wt
                loso_cur += e_te[ci, cj] * wt
            tot = d["w"].sum()
            loso_fit, loso_cur = loso_fit / tot, loso_cur / tot
        # Do club-changers' last seasons deserve less weight?
        movers = ""
        if d["moved"].sum() >= 100:
            # fit last season's weight separately, holding the position-average
            # weight at the pooled best
            j = list(M_GRID).index(M_all)
            Ks = int(K_GRID[np.argmin(sq_error(d[~d["moved"]], K_GRID, M_GRID)[:, j])])
            Km = int(K_GRID[np.argmin(sq_error(d[d["moved"] | ~d["has_last"]], K_GRID, M_GRID)[:, j])])
            movers = (f"last-season weight: stayed at club {Ks}, changed club {Km} "
                      f"({int(d['moved'].sum())} rows)")
        label = f"{stat}{' (GK)' if group == 'keeper' else ''}"
        change = (loso_fit / loso_cur - 1) * 100 if loso_cur else float("nan")
        summary.append((label, K_all, M_all, cur_K, cur_M, K_all + M_all, change, len(d), seasons))
        print(f"{label:<28} best last-season={K_all:>5}  pos-avg={M_all:>5}   "
              f"(now {cur_K}/{cur_M})   own data = half at {K_all + M_all:>5} min   "
              f"held-out error {change:+.1f}% vs now   rows={len(d)}  "
              f"seasons={','.join(x[-5:] for x in seasons)}")
        if movers:
            print(f"{'':<28} {movers}")
    return summary


if __name__ == "__main__":
    main()
