"""Record phase and chip tuning.

  python run_backtest.py record [--seasons 2023-24,...] [--squads 5] [--gw 38]
  python run_backtest.py tune   [--grid 0,0.05,...]
  python run_backtest.py wildcard [--seasons 2023-24,...]
  python run_backtest.py gaps
  python run_backtest.py bars [--chip bboost] [--bars 4,6,8,...]
  python run_backtest.py policy [--ranks 0.05,0.15,0.30] [--gaps 15,20,25,30] [--squads 0,1]

`record` runs one full season per (season, starting squad) with only the wildcard
playable, saving every chip's opportunity table and what that chip really paid in
that week. `tune` replays those tables for each candidate x, which takes seconds -
Triple Captain and Bench Boost never change the squad and Free Hit changes it for
one week, so their payoffs do not depend on when they were played.

`wildcard` can't work that way - a wildcard changes every week after it - so it
runs a full season for each candidate week instead, one half at a time.
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


def projector_for(season):
    """Projections don't depend on chip rules, so every command shares one cache."""
    cfg = base_cfg()
    return B.Projector(B.ArchiveSource(season), cfg, cfg["PROJECTION_WEEKS"])


def record_one(args):
    season, squad_ix, last_gw = args
    # No chips played: the squad path doesn't depend on any decision being tuned,
    # and the wildcard's signal is logged every week of the season.
    cfg = base_cfg(PLAYABLE_CHIPS=[])
    out = RECORDS / f"{season}_squad{squad_ix}_gw{last_gw}.json"
    if out.exists():
        return f"{season} squad {squad_ix}: already recorded, skipping"
    projector = projector_for(season)
    squads = B.starting_squads(projector, cfg, n=8)
    t0 = time.time()
    r = B.simulate(projector, cfg, f"{season}/sq{squad_ix}",
                   start=list(squads[squad_ix]), last_gw=last_gw, record_chips=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"season": season, "squad": squad_ix, "total": int(r["total"]),
                               "chips": r["chips"], "log": r["log"],
                               "tables": r["chip_tables"]}, default=str))
    return (f"{season} squad {squad_ix}: {r['total']} pts, wildcards {r['chips']}, "
            f"{len(r['chip_tables'])} weeks recorded ({time.time() - t0:.0f}s)")


def cmd_record(a):
    seasons = a.seasons.split(",") if a.seasons else SEASONS
    for s in seasons:
        B.download(s)
        # Build each season's projection cache serially: the workers all share it,
        # and several of them writing the same pickle at once would corrupt it.
        t0 = time.time()
        projector_for(s)
        print(f"{s}: projections ready ({time.time() - t0:.0f}s)", flush=True)
    jobs = [(s, i, a.gw) for s in seasons for i in range(a.squads)]
    print(f"{len(jobs)} record runs across {a.workers} workers\n")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(record_one, jobs):
            print(msg, flush=True)


WILDCARD_DIR = ROOT / "bt_cache" / "wildcard"
# Candidate weeks per half; None = never play it in that half. The other half is
# left to the timing rule, so each half is measured with everything else equal.
WILDCARD_WEEKS = {19: [None, 2, 5, 8, 11, 14, 17], 38: [None, 20, 23, 26, 29, 32, 35]}


def wildcard_one(args):
    season, stop, gw = args
    out = WILDCARD_DIR / f"{season}_stop{stop}_gw{gw or 'none'}.json"
    if out.exists():
        return f"{season} half to GW{stop}, wildcard GW{gw or '-'}: already run, skipping"
    # The projector is built from the recording config so its cache is shared;
    # only the simulation sees the forced week.
    rec_cfg = base_cfg(PLAYABLE_CHIPS=["wildcard"])
    projector = projector_for(season)
    start = B.starting_squads(projector, rec_cfg, n=1)[0]
    cfg = dict(rec_cfg, FORCE_WILDCARD={stop: gw})
    t0 = time.time()
    r = B.simulate(projector, cfg, f"{season}/wc{gw}", start=list(start))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"season": season, "stop": stop, "gw": gw, "total": int(r["total"]),
                               "chips": r["chips"], "log": r["log"]}, default=str))
    return (f"{season} half to GW{stop}, wildcard GW{gw or '-'}: {r['total']} pts "
            f"({r['chips'] or 'no wildcard'}, {time.time() - t0:.0f}s)")


def cmd_wildcard(a):
    seasons = a.seasons.split(",") if a.seasons else SEASONS
    for s in seasons:
        B.download(s)
        projector_for(s)
        print(f"{s}: projections ready", flush=True)
    jobs = [(s, stop, gw) for s in seasons for stop, weeks in WILDCARD_WEEKS.items()
            for gw in weeks]
    print(f"{len(jobs)} full-season runs across {a.workers} workers\n", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(wildcard_one, jobs):
            print(msg, flush=True)
    wildcard_report(seasons)


def wildcard_report(seasons):
    """Season totals by forced week, next to the same squad with no wildcard at all."""
    rule = {}
    for rec in load_records():
        if rec["squad"] == 0:
            rule[rec["season"]] = (int(rec["total"]), rec["chips"])
    for stop, weeks in WILDCARD_WEEKS.items():
        print(f"\nWildcard in the half ending GW{stop} (other half left to the rule)")
        print(f"  {'week':>6}" + "".join(f"{s:>10}" for s in seasons) + f"{'mean':>9}")
        for gw in weeks:
            cells = []
            for s in seasons:
                p = WILDCARD_DIR / f"{s}_stop{stop}_gw{gw or 'none'}.json"
                # int(): files written before the cast hold the total as a string
                cells.append(int(json.loads(p.read_text())["total"]) if p.exists() else None)
            got = [c for c in cells if c is not None]
            mean = f"{statistics.mean(got):>9.0f}" if got else f"{'':>9}"
            print(f"  {gw or 'none':>6}" + "".join(f"{c if c is not None else '-':>10}"
                                                 for c in cells) + mean)
        # The recordings play no chips, so this is the floor: no wildcard in
        # either half. The rule's own choice shows up as the row where the forced
        # week matches what the rule plays in the other table.
        print(f"  {'no wc':>6}" + "".join(f"{rule.get(s, ('-',))[0]:>10}" for s in seasons))


def cmd_bars(a):
    """Replay the bar rule (play once this week's value reaches the bar) per chip,
    next to the current top-x% rule."""
    recs = load_records()
    if not recs:
        sys.exit("no records found - run `python run_backtest.py record` first")
    chip = a.chip
    seasons = sorted({r["season"] for r in recs})
    x_now = E.DEFAULTS["CHIP_TOP_PCT"][chip]
    print(f"{E.CHIP_NAMES[chip]}: bar rule vs top-{x_now:.0%} rule, "
          f"{len(recs)} runs over {len(seasons)} seasons\n")
    print(f"  {'rule':>10}{'pts/run':>10}{'se':>8}   per season")
    rules = [(f"top {x_now:.2f}", dict(bar=None))] + \
            [(f"bar {b:g}", dict(bar=b)) for b in [float(v) for v in a.bars.split(",")]]
    for label, kw in rules:
        per_run, by_season = [], {}
        for rec in recs:
            played = B.replay_chip(rec["tables"], chip, x_now, **kw)
            got = sum(p["actual"] for p in played if p["actual"] is not None)
            per_run.append(got)
            by_season.setdefault(rec["season"], []).append(got)
        se = statistics.stdev(per_run) / len(per_run) ** 0.5
        cells = "  ".join(f"{s[-5:]} {statistics.mean(v):>5.1f}" for s, v in sorted(by_season.items()))
        print(f"  {label:>10}{statistics.mean(per_run):>10.1f}{se:>8.1f}   {cells}")


def cmd_gaps(a):
    """How big the wildcard gap gets, by stage of season - the history a bar is set from."""
    recs = load_records()
    rows = [(r["season"], t, w["wildcard"]["gap"]) for r in recs
            for t, w in r["tables"].items() if "wildcard" in w]
    if not rows:
        sys.exit("no wildcard gaps recorded - re-run `record`")
    import pandas as pd
    d = pd.DataFrame(rows, columns=["season", "gw", "gap"])
    d["stage"] = pd.cut(d["gw"], [0, 5, 10, 15, 19, 25, 30, 38],
                        labels=["2-5", "6-10", "11-15", "16-19", "20-25", "26-30", "31-38"])
    q = d.groupby("stage", observed=True)["gap"].quantile([0.5, 0.75, 0.9, 1.0]).unstack()
    q.columns = ["median", "p75", "p90", "max"]
    print("Wildcard gap (planner points a rebuild adds), no chips played:\n")
    print(q.round(1).to_string())
    print("\nBy season, weeks 2-19 / 20-38 (median, p90):")
    for s, g in d.groupby("season"):
        h1, h2 = g[g["gw"] <= 19]["gap"], g[g["gw"] >= 20]["gap"]
        print(f"  {s}: {h1.median():.1f}, {h1.quantile(.9):.1f}  /  "
              f"{h2.median():.1f}, {h2.quantile(.9):.1f}")


POLICY_DIR = ROOT / "bt_cache" / "policy"


def policy_one(args):
    season, label, over, squad_ix = args
    sq = f"_sq{squad_ix}" if squad_ix else ""     # squad 0 keeps its original name
    out = POLICY_DIR / f"{season}_{label}{sq}.json"
    if out.exists():
        return f"{season} {label}: already run, skipping"
    cfg = base_cfg(PLAYABLE_CHIPS=["wildcard"], **over)
    projector = projector_for(season)
    start = B.starting_squads(projector, cfg, n=squad_ix + 1)[squad_ix]
    t0 = time.time()
    r = B.simulate(projector, cfg, f"{season}/{label}{sq}", start=list(start))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"season": season, "label": label, "over": over, "squad": squad_ix,
                               "total": int(r["total"]), "chips": r["chips"], "log": r["log"]},
                              default=str))
    return f"{season} {label}{sq}: {r['total']} pts ({r['chips'] or 'no wildcard'}, {time.time() - t0:.0f}s)"


def cmd_policy(a):
    """Full seasons under each wildcard rule: the rank rule at several shares, and
    the gap rule at several bars."""
    seasons = a.seasons.split(",") if a.seasons else SEASONS
    for s in seasons:
        B.download(s)
        projector_for(s)
    pct = E.DEFAULTS["CHIP_TOP_PCT"]
    # WILDCARD_GAP=None, or the default bar would override the rank rule
    policies = [(f"rank{x:.2f}", {"CHIP_TOP_PCT": dict(pct, wildcard=x), "WILDCARD_GAP": None})
                for x in [float(v) for v in a.ranks.split(",")]]
    policies += [(f"gap{g:g}", {"WILDCARD_GAP": g}) for g in [float(v) for v in a.gaps.split(",")]]
    squads = [int(v) for v in a.squads.split(",")]
    jobs = [(s, label, over, q) for s in seasons for q in squads for label, over in policies]
    print(f"{len(jobs)} full-season runs across {a.workers} workers\n", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(policy_one, jobs):
            print(msg, flush=True)
    for q in squads:
        sq = f"_sq{q}" if q else ""
        print(f"\n  starting squad {q}")
        print(f"  {'policy':<12}" + "".join(f"{s:>10}" for s in seasons) + f"{'mean':>9}   wildcards")
        for label, _ in policies:
            rs = [POLICY_DIR / f"{s}_{label}{sq}.json" for s in seasons]
            rs = [json.loads(p.read_text()) if p.exists() else None for p in rs]
            tot = [int(r["total"]) for r in rs if r]
            print(f"  {label:<12}" + "".join(f"{int(r['total']) if r else '-':>10}" for r in rs)
                  + (f"{statistics.mean(tot):>9.0f}" if tot else "")
                  + "   " + "; ".join(f"{r['season'][-5:]} {r['chips'] or '-'}" for r in rs if r))


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
    br = sub.add_parser("bars")
    br.add_argument("--chip", default="bboost")
    br.add_argument("--bars", default="4,6,8,10,12,14,16,18,20")
    br.set_defaults(func=cmd_bars)
    g = sub.add_parser("gaps")
    g.set_defaults(func=cmd_gaps)
    pol = sub.add_parser("policy")
    pol.add_argument("--seasons")
    pol.add_argument("--ranks", default="0.05,0.15,0.30")
    pol.add_argument("--gaps", default="5,10,15,20")
    pol.add_argument("--squads", default="0", help="starting squads, e.g. 0,1,2,3,4")
    pol.add_argument("--workers", type=int, default=os.cpu_count())
    pol.set_defaults(func=cmd_policy)
    w = sub.add_parser("wildcard")
    w.add_argument("--seasons")
    w.add_argument("--workers", type=int, default=os.cpu_count())
    w.set_defaults(func=cmd_wildcard)
    a = p.parse_args()
    a.func(a)
