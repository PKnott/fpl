"""Are the projections over-confident, and which part of them?

Over six weeks, actual points come out at roughly 3.6 + 0.70 x projected: big
projections overshoot and small ones undershoot. A well-calibrated projection
has a slope of 1 (a player projected 2 points more really does score 2 more).
This splits every projection into the parts the engine builds it from - playing
time, goals, assists, clean sheets, bonus and so on - and checks each against
what actually happened, so the fix can go where the problem is.

For every player, at every week of three seasons, and for each of the next six
weeks, it compares the engine's projection (from the backtest's projection
cache) with the real points, scored with the same rules the engine assumed.

    python calibration.py         # where the over-confidence is
    python calibration.py grid    # candidate fixes, scored the same way
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

import fpl_engine as E                                           # noqa: E402
import run_backtest as R                                         # noqa: E402

SEASONS = ["2023-24", "2024-25", "2025-26"]
AHEAD = 6
# The engine's fallback scoring, which is what it uses on the archive (the
# archive has no game_config). Actual points are re-scored with the same values.
GOAL = {1: 10, 2: 6, 3: 5, 4: 4}
CS = {1: 4, 2: 4, 3: 1, 4: 0}
PARTS = ["Playing time", "Goals", "Assists", "Clean sheet", "Goals conceded", "Saves",
         "Def. contributions", "Bonus", "Yellow cards"]


def actual_parts(season):
    """Real points per (player, gameweek), split the way the engine splits them."""
    m = pd.read_csv(B_DATA() / f"{season}_gws_merged_gw.csv")
    pos = m["position"].map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4})
    m = m[pos.notna()].copy()
    m["pos"] = pos[pos.notna()].astype(int)
    dc = m["defensive_contribution"] if "defensive_contribution" in m else 0
    thresh = m["pos"].map(E.DC_THRESHOLD).fillna(np.inf)
    played = m["minutes"] > 0
    m["Playing time"] = np.where(m["minutes"] >= 60, 2, np.where(played, 1, 0))
    m["Goals"] = m["goals_scored"] * m["pos"].map(GOAL)
    m["Assists"] = m["assists"] * 3
    m["Clean sheet"] = m["clean_sheets"] * m["pos"].map(CS)
    m["Goals conceded"] = np.where(m["pos"] <= 2, -(m["goals_conceded"] // 2), 0)
    m["Saves"] = np.where(m["pos"] == 1, m["saves"] // 3, 0)
    m["Def. contributions"] = np.where((m["pos"] >= 2) & (dc >= thresh), 2, 0)
    m["Bonus"] = m["bonus"]
    m["Yellow cards"] = -m["yellow_cards"]
    m["Total"] = m["total_points"]
    return m.groupby(["element", "GW"])[PARTS + ["Total"]].sum()


def B_DATA():
    import fpl_backtest as B
    return B.DATA


def rows(cfg=None):
    import fpl_backtest as B
    out = []
    for season in SEASONS:
        pj = R.projector_for(season) if cfg is None else \
            B.Projector(B.ArchiveSource(season), cfg, cfg["PROJECTION_WEEKS"])
        act = actual_parts(season)
        act_idx = set(act.index)
        actd = act.to_dict("index")
        for t in range(2, 39):
            p = pj.data[t][0]
            df = p.df[p.df["can_select"]]
            for i in df.index:
                ex = p.explain.get(i)
                if not ex:
                    continue
                for k in range(AHEAD):
                    g = t + k
                    if g > 38 or (i, g) not in act_idx or g not in ex["gws"]:
                        continue
                    comp = ex["gws"][g]["components"]
                    a = actd[(i, g)]
                    r = {"season": season, "t": t, "id": i, "k": k + 1, "pos": int(df.loc[i, "pos"]),
                         "p_any": ex["p_any"], "proj_total": float(df.loc[i, f"gw{g}"]),
                         "act_total": a["Total"]}
                    for part in PARTS:
                        r[f"proj_{part}"] = comp.get(part, 0.0)
                        r[f"act_{part}"] = a[part]
                    out.append(r)
    return pd.DataFrame(out)


def fit(x, y):
    """Slope and intercept of actual on projected (1 and 0 = perfectly calibrated)."""
    b, a = np.polyfit(x, y, 1)
    return b, a


def main():
    d = rows()
    print(f"{len(d):,} player-week-target rows, 3 seasons\n")

    print("TOTAL POINTS by how far ahead (slope 1.00 = calibrated)")
    print(f"  {'weeks ahead':>11}{'slope':>8}{'intercept':>11}{'mean proj':>11}{'mean actual':>13}")
    for k in range(1, AHEAD + 1):
        dk = d[d["k"] == k]
        b, a = fit(dk["proj_total"], dk["act_total"])
        print(f"  {k:>11}{b:>8.2f}{a:>11.2f}{dk['proj_total'].mean():>11.2f}{dk['act_total'].mean():>13.2f}")

    print("\nTOTAL POINTS by season and position (next week)")
    d1 = d[d["k"] == 1]
    for s in SEASONS:
        cells = []
        for p_ in (1, 2, 3, 4):
            x = d1[(d1["season"] == s) & (d1["pos"] == p_)]
            cells.append(f"{E.POS[p_]} {fit(x['proj_total'], x['act_total'])[0]:.2f}")
        b, _ = fit(d1[d1["season"] == s]["proj_total"], d1[d1["season"] == s]["act_total"])
        print(f"  {s}: all {b:.2f}   " + "   ".join(cells))

    print("\nBY PART (next week; players with a fixture). Slope below 1 = that part "
          "is spread too wide.")
    print(f"  {'part':<20}{'slope':>8}{'intercept':>11}{'mean proj':>11}{'mean actual':>13}"
          f"{'share of proj':>15}")
    tot = d1["proj_total"].sum()
    for part in PARTS:
        x, y = d1[f"proj_{part}"], d1[f"act_{part}"]
        if x.abs().sum() == 0:
            continue
        b, a = fit(x, y)
        print(f"  {part:<20}{b:>8.2f}{a:>11.2f}{x.mean():>11.3f}{y.mean():>13.3f}"
              f"{x.sum() / tot:>14.0%}")

    print("\nBY PART, only players the engine is sure will play (chance of playing 95%+)")
    sure = d1[d1["p_any"] >= 0.95]
    for part in ["Playing time", "Goals", "Assists", "Clean sheet", "Bonus"]:
        b, a = fit(sure[f"proj_{part}"], sure[f"act_{part}"])
        print(f"  {part:<20}{b:>8.2f}{a:>11.2f}")
    b, a = fit(sure["proj_total"], sure["act_total"])
    print(f"  {'Total':<20}{b:>8.2f}{a:>11.2f}")

    print("\nDECILES of next-week projection: mean projected vs mean actual")
    d1 = d1.assign(bin=pd.qcut(d1["proj_total"].rank(method="first"), 10, labels=False))
    g = d1.groupby("bin")[["proj_total", "act_total"]].mean()
    for b_, r in g.iterrows():
        print(f"  decile {b_ + 1:>2}: projected {r['proj_total']:5.2f}   actual {r['act_total']:5.2f}"
              f"   {'over' if r['proj_total'] > r['act_total'] else 'under'} by "
              f"{abs(r['proj_total'] - r['act_total']):.2f}")


def summary(d):
    """Calibration slopes, raw error and six-week correlation for one config."""
    d1, d6 = d[d["k"] == 1], d[d["k"] == AHEAD]
    six = d.groupby(["season", "t", "id"]).agg(
        {"proj_total": "sum", "act_total": "sum", "k": "count"})
    six = six[six["k"] == AHEAD]
    out = {"slope 1wk": fit(d1["proj_total"], d1["act_total"])[0],
           "slope 6th wk": fit(d6["proj_total"], d6["act_total"])[0],
           "RMSE 1wk": np.sqrt(((d1["proj_total"] - d1["act_total"]) ** 2).mean()),
           "RMSE 6wk sum": np.sqrt(((six["proj_total"] - six["act_total"]) ** 2).mean()),
           "corr 6wk": np.corrcoef(six["proj_total"], six["act_total"])[0, 1]}
    for s_ in SEASONS:
        x = six.loc[s_]
        out[f"RMSE6 {s_[-5:]}"] = np.sqrt(((x["proj_total"] - x["act_total"]) ** 2).mean())
    return out


GRID = [
    ("current", {}),
    ("decay 0.98", {"AVAIL_DECAY": 0.98}),
    ("decay 0.96", {"AVAIL_DECAY": 0.96}),
    ("shrink 0.9 to 0.5", {"MINUTES_SHRINK": 0.9, "MINUTES_SHRINK_TO": 0.5}),
    ("shrink 0.8 to 0.5", {"MINUTES_SHRINK": 0.8, "MINUTES_SHRINK_TO": 0.5}),
    ("shrink 0.9 to 0.3", {"MINUTES_SHRINK": 0.9, "MINUTES_SHRINK_TO": 0.3}),
    ("shrink 0.9/0.5 + decay 0.98", {"MINUTES_SHRINK": 0.9, "MINUTES_SHRINK_TO": 0.5,
                                     "AVAIL_DECAY": 0.98}),
    ("shrink 0.8/0.5 + decay 0.97", {"MINUTES_SHRINK": 0.8, "MINUTES_SHRINK_TO": 0.5,
                                     "AVAIL_DECAY": 0.97}),
]


def grid():
    base = R.base_cfg()
    res = []
    for label, over in GRID:
        m = summary(rows(dict(base, **over)))
        res.append((label, m))
        print(f"  {label:<30}" + "  ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["grid"]:
        grid()
    else:
        main()
