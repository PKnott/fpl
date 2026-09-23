"""Stage 1 - which projection settings predict 2025/26 points best?"""
import sys, itertools, json, math
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from season import Season
import fpl_v2 as F

S = Season()
SNAP = {t: S.snapshot(t) for t in range(2, 39)}
TOPK = {1: 3, 2: 10, 3: 10, 4: 6}
FPL_XP_GWS = [g for g in range(2, 39) if S.fpl_xp.xs(g, level=1).abs().sum() > 0]


def evaluate(cfg, teams_override=None, label=""):
    rows, picks, caps, spear = [], [], [], []
    gk_rows, gk_picks = [], []
    for t, (bs, fx) in SNAP.items():
        if teams_override:
            bs = dict(bs, teams=teams_override)
        c = dict(F.CONFIG, **cfg, LONG_VIEW=1, AVAILABILITY_OVERRIDES={}, START_OVERRIDES={})
        df, *_ = F.build_projections(bs, fx, c, t, S.past)
        df = df[df["can_select"]].copy()
        df["actual"] = [S.actual.get((i, t), 0) for i in df.index]
        played = {e["id"] for e in bs["elements"] if e["minutes"] >= 90}
        rel = df[df.index.isin(played)]
        rows.append(pd.DataFrame({"xp": rel["xp_next"], "act": rel["actual"], "gw": t}))
        spear.append(spearmanr(rel["xp_next"], rel["actual"]).correlation)
        for p, k in TOPK.items():
            picks.append(df[df["pos"] == p].nlargest(k, "xp_next")["actual"].mean())
        caps.append(df.nlargest(1, "xp_next")["actual"].iloc[0])
        # keepers: accuracy, best-keeper picks, and the save-rate bias
        gk = df[df["pos"] == 1]
        gk_picks.append(gk.nlargest(3, "xp_next")["actual"].mean())
        for i in gk.index[gk.index.isin(played)]:
            ex = F.EXPLAIN[i]
            gk_rows.append({"xp": df.loc[i, "xp_next"], "act": df.loc[i, "actual"],
                            "played": S.minutes.get((i, t), 0) >= 60,
                            "nailed": ex["start_rate"] >= 0.9,
                            "save_rate": ex["rates"]["saves"]["blended"]})
    g = pd.DataFrame(gk_rows)
    g_err = g["xp"] - g["act"]
    s_ = g[g["played"] & g["nailed"]].copy()
    s_["grp"] = pd.qcut(s_["save_rate"], 3, labels=False, duplicates="drop")
    bias_by = (s_["xp"] - s_["act"]).groupby(s_["grp"]).mean()
    save_gap = float(bias_by.iloc[-1] - bias_by.iloc[0]) if len(bias_by) > 1 else float("nan")
    r = pd.concat(rows)
    err = r["xp"] - r["act"]
    return {"label": label, **cfg,
            "RMSE": math.sqrt((err ** 2).mean()), "MAE": err.abs().mean(), "bias": err.mean(),
            "rank_corr": float(np.mean(spear)), "top_picks_avg": float(np.mean(picks)),
            "captain_avg": float(np.mean(caps)),
            "gk_error": math.sqrt((g_err ** 2).mean()), "gk_bias": float(g_err.mean()),
            "gk_top_picks": float(np.mean(gk_picks)), "gk_save_gap": save_gap}


def fpl_benchmark():
    """FPL's own xP on the weeks it was recorded, vs our default, same weeks."""
    out = {}
    for src in ["FPL xP", "Our model (default)"]:
        errs, sp = [], []
        for t in FPL_XP_GWS:
            bs, fx = SNAP[t]
            c = dict(F.CONFIG, LONG_VIEW=1, AVAILABILITY_OVERRIDES={}, START_OVERRIDES={})
            df, *_ = F.build_projections(bs, fx, c, t, S.past)
            df = df[df["can_select"]]
            played = {e["id"] for e in bs["elements"] if e["minutes"] >= 90}
            df = df[df.index.isin(played)]
            act = np.array([S.actual.get((i, t), 0) for i in df.index])
            xp = (np.array([S.fpl_xp.get((i, t), 0) for i in df.index]) if src == "FPL xP"
                  else df["xp_next"].values)
            errs += list(xp - act)
            sp.append(spearmanr(xp, act).correlation)
        e = np.array(errs)
        out[src] = {"RMSE": float(np.sqrt((e ** 2).mean())), "rank_corr": float(np.mean(sp))}
    return out, FPL_XP_GWS



def run_stage1(grid):
    """grid = {setting: [values]}. LAST_SEASON_MINUTES 0 means last-season stats off."""
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Testing {len(combos)} combinations...", flush=True)
    res = []
    for vals in combos:
        cfg = dict(zip(keys, vals))
        if cfg.get("LAST_SEASON_MINUTES", 1) == 0:
            cfg["LAST_SEASON_STATS"] = False
        elif "LAST_SEASON_MINUTES" in cfg:
            cfg["LAST_SEASON_STATS"] = True
        r = evaluate(cfg)
        res.append(r)
        desc = ", ".join(f"{k} {v}" for k, v in zip(keys, vals))
        print(f"{desc}: error {r['RMSE']:.3f}, ranking {r['rank_corr']:.3f}, "
              f"top picks {r['top_picks_avg']:.3f} | keepers: error {r['gk_error']:.3f}, "
              f"top-3 {r['gk_top_picks']:.2f}, save gap {r['gk_save_gap']:+.2f}", flush=True)
    df = pd.DataFrame(res).drop(columns="label")
    df.to_csv("stage1_results.csv", index=False)
    bench, gws = fpl_benchmark()
    print("\nFPL's own xP vs our model (weeks FPL xP was recorded: %s)" % gws)
    for k, v in bench.items():
        print(f"  {k:<22} error {v['RMSE']:.3f}   ranking {v['rank_corr']:.3f}")
    print("  (FPL's numbers include injury/suspension news the archive doesn't give our model.)")
    return df
