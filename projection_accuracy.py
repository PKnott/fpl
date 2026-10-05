"""Do projected points predict real points? Compare configs on every player-week.

    python projection_accuracy.py
"""
import os, pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("BT_DATA", str(ROOT / "bt_data"))
os.environ.setdefault("BT_CACHE", str(ROOT / "bt_cache"))
sys.path.insert(0, str(ROOT))
import fpl_backtest as B                                         # noqa: E402
import run_backtest as R                                         # noqa: E402

SEASONS = ["2023-24", "2024-25", "2025-26"]
AHEAD = 6                       # also score the sum of the next 6 weeks


def score(season, cfg):
    src = B.ArchiveSource(season)
    pj = B.Projector(src, cfg, cfg["PROJECTION_WEEKS"])
    one, many = [], []
    for t in range(2, 39):
        df = pj.data[t][0].df
        df = df[df["can_select"]]
        for i, row in df.iterrows():
            act = src.actual.get((i, t))
            if act is None:
                continue
            one.append((row[f"gw{t}"], float(act)))
            ws = [g for g in range(t, min(t + AHEAD, 39)) if f"gw{g}" in df.columns]
            if len(ws) == AHEAD:
                a = [src.actual.get((i, g)) for g in ws]
                if all(x is not None for x in a):
                    many.append((sum(row[f"gw{g}"] for g in ws), float(sum(a))))
    out = {}
    for name, rows in (("next week", one), (f"next {AHEAD}", many)):
        p, a = np.array(rows).T
        out[name] = (np.sqrt(np.mean((p - a) ** 2)), np.corrcoef(p, a)[0, 1], len(rows))
    return out


def compare(configs):
    for season in SEASONS:
        print(season)
        for label, cfg in configs.items():
            r = score(season, cfg)
            print(f"  {label:<22}" + "   ".join(f"{k}: RMSE {v[0]:.4f} r {v[1]:.4f}"
                                                for k, v in r.items()), flush=True)


if __name__ == "__main__":
    base = R.base_cfg(PLAYABLE_CHIPS=["wildcard"])
    compare({"one weight for all": dict(base, STAT_PRIOR_MINUTES={}),
             "per-stat (DEFAULTS)": base})
