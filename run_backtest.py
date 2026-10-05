"""Record phase and chip tuning.

  python run_backtest.py record [--seasons 2023-24,...] [--squads 5] [--gw 38]
  python run_backtest.py tune   [--grid 0,0.05,...]

`record` runs one full season per (season, starting squad) with only the wildcard
playable, saving every chip's opportunity table and what that chip really paid in
that week. `tune` replays those tables for each candidate x, which takes seconds -
Triple Captain and Bench Boost never change the squad and Free Hit changes it for
one week, so their payoffs do not depend on when they were played.
"""
import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("BT_DATA", str(ROOT / "bt_data"))
os.environ.setdefault("BT_CACHE", str(ROOT / "bt_cache"))
sys.path.insert(0, str(ROOT))

import fpl_engine as E                                           # noqa: E402
import fpl_backtest as B                                         # noqa: E402

SEASONS = ["2023-24", "2024-25", "2025-26"]
RECORDS = ROOT / "bt_cache" / "records"
REPLAY_CHIPS = ["3xc", "bboost", "freehit"]
GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def base_cfg(**over):
    """The engine's own defaults, with the backtest's harness pins on top."""
    return E.resolve_cfg(
        dict(PRICE_CHANGES=False, AVAILABILITY_OVERRIDES={}, START_OVERRIDES={},
             LOCK=[], EXCLUDE=[], HIT_COST_REAL=4, SOLVER_SECONDS=20,
             FREE_TRANSFERS=1),
        over)


def record_one(args):
    season, squad_ix, last_gw = args
    cfg = base_cfg(PLAYABLE_CHIPS=["wildcard"])
    out = RECORDS / f"{season}_squad{squad_ix}_gw{last_gw}.json"
    if out.exists():
        return f"{season} squad {squad_ix}: already recorded, skipping"
    source = B.ArchiveSource(season)
    projector = B.Projector(source, cfg, cfg["PROJECTION_WEEKS"])
    squads = B.starting_squads(projector, cfg, n=8)
    t0 = time.time()
    r = B.simulate(projector, cfg, f"{season}/sq{squad_ix}",
                   start=list(squads[squad_ix]), last_gw=last_gw, record_chips=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"season": season, "squad": squad_ix, "total": r["total"],
                               "chips": r["chips"], "log": r["log"],
                               "tables": r["chip_tables"]}, default=str))
    return (f"{season} squad {squad_ix}: {r['total']} pts, wildcards {r['chips']}, "
            f"{len(r['chip_tables'])} weeks recorded ({time.time() - t0:.0f}s)")


def cmd_record(a):
    seasons = a.seasons.split(",") if a.seasons else SEASONS
    cfg = base_cfg(PLAYABLE_CHIPS=["wildcard"])
    for s in seasons:
        B.download(s)
        # Build each season's projection cache serially: the workers all share it,
        # and several of them writing the same pickle at once would corrupt it.
        t0 = time.time()
        B.Projector(B.ArchiveSource(s), cfg, cfg["PROJECTION_WEEKS"])
        print(f"{s}: projections ready ({time.time() - t0:.0f}s)", flush=True)
    jobs = [(s, i, a.gw) for s in seasons for i in range(a.squads)]
    print(f"{len(jobs)} record runs across {a.workers} workers\n")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(record_one, jobs):
            print(msg, flush=True)


def load_records():
    recs = []
    for p in sorted(RECORDS.glob("*.json")):
        d = json.loads(p.read_text())
        d["tables"] = {int(k): v for k, v in d["tables"].items()}
        # Records written before record_week cast `actual` to float hold numpy ints,
        # which json.dumps(default=str) saved as strings.
        for week in d["tables"].values():
            for info in week.values():
                if isinstance(info.get("actual"), str):
                    info["actual"] = float(info["actual"])
        recs.append(d)
    return recs


def cmd_tune(a):
    grid = [float(x) for x in a.grid.split(",")] if a.grid else GRID
    recs = load_records()
    if not recs:
        sys.exit("no records found - run `python run_backtest.py record` first")
    seasons = sorted({r["season"] for r in recs})
    print(f"{len(recs)} recorded runs over {len(seasons)} seasons: {', '.join(seasons)}\n")

    winners = {}
    for chip in REPLAY_CHIPS:
        print(f"{E.CHIP_NAMES[chip]}")
        print(f"  {'x':>6}{'pts/run':>10}{'se':>8}   per season")
        best = None
        for x in grid:
            # Paired by run: every x is measured on the same seasons and squads,
            # so the differences are not confounded by which squad got lucky.
            per_run, by_season = [], {}
            for rec in recs:
                played = B.replay_chip(rec["tables"], chip, x)
                got = sum(p["actual"] for p in played if p["actual"] is not None)
                per_run.append(got)
                by_season.setdefault(rec["season"], []).append(got)
            mean = statistics.mean(per_run)
            se = statistics.stdev(per_run) / len(per_run) ** 0.5 if len(per_run) > 1 else 0.0
            cells = "  ".join(f"{s[-5:]} {statistics.mean(v):>5.1f}"
                              for s, v in sorted(by_season.items()))
            mark = ""
            if best is None or mean > best[1]:
                best, mark = (x, mean), ""
            print(f"  {x:>6.2f}{mean:>10.1f}{se:>8.1f}   {cells}{mark}")
        winners[chip] = best[0]
        print(f"  -> best x = {best[0]:.2f} ({best[1]:.1f} pts/run)\n")

    print("Suggested CHIP_TOP_PCT (round these before adopting - see the standard errors):")
    print("  " + json.dumps(winners))
    print("\nCaveat: the archive's fixture list is final, so rescheduled double\n"
          "gameweeks are visible earlier here than they were at the time, which\n"
          "flatters waiting. Wildcard is not tunable by replay - it changes the\n"
          "squad trajectory, so it needs full simulations.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--seasons")
    r.add_argument("--squads", type=int, default=5)
    r.add_argument("--gw", type=int, default=38)
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.set_defaults(func=cmd_record)
    t = sub.add_parser("tune")
    t.add_argument("--grid")
    t.set_defaults(func=cmd_tune)
    a = p.parse_args()
    a.func(a)
