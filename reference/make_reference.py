"""Freeze reference outputs from the CURRENT, unmodified code (_ref_current/).

Steps 2 and 3 of the refactor must reproduce these exactly. To make that a fair
test, everything is pinned explicitly here: the notebook left MINUTES_GAMES,
MAX_TRANSFERS, TAIL_WEIGHT, FT_END_VALUE, BENCH_WEIGHT and every chip setting
falling through to fpl_v2.CONFIG, which is the drifted copy. Merging the engines
changes those defaults on purpose, so the reference config names all of them and
ref_cfg.json is what both old and new code are run against.

Usage:  python reference/make_reference.py [--quick]
"""
import hashlib
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CUR = ROOT / "_ref_current"

# Projection snapshots to freeze: early season (thin data), mid, the second-half
# boundary where chip windows reset, and the last week.
REF_GWS = [2, 10, 20, 30, 38]

# ---------------------------------------------------------------------
# The reference configuration, materialised in full.
# ---------------------------------------------------------------------
# From the backtest notebook, cell 1.
STAGE2_PROJECTION = dict(
    XG_WEIGHT=0.85, PRIOR_MINUTES=90, LAST_SEASON_STATS=True, LAST_SEASON_MINUTES=540,
    LAST_SEASON_STARTS=False, GK_PRIOR_MINUTES=360, SAVES_LEAK=0.5,
    SAVES_LEAK_MODE="shots", TEAM_PRIOR_GAMES=8, MINUTES_DECAY=0.6)
BASE = dict(HIT_COST=6, DECAY=0.8, PLAN_WEEKS=5, LONG_VIEW=19)
REF_LABEL = "hit6 decay0.8"

# From sim.py lines 13-16: what the harness pins on top of everything else.
MAX_VIEW = 18
SIM_PINS = dict(PRICE_CHANGES=False, AVAILABILITY_OVERRIDES={}, START_OVERRIDES={},
                LOCK=[], EXCLUDE=[], HIT_COST_REAL=4, SOLVER_SECONDS=20)

# The closed set of settings that can reach a projection - every cfg key read by
# build_projections, team_strength and use_last_season (fpl_v2.py 178-238, 264-273,
# 277-434). Nothing outside this set can change a projected point, so pinning these
# pins the projections. sim.py names its pickle after the *values* of whatever it is
# given, so handing it the whole config would produce a filename over the 255-byte
# limit; this keeps it short without leaving anything unpinned.
PROJ_KEYS = [
    "XG_WEIGHT", "PRIOR_MINUTES", "MINUTES_DECAY", "MINUTES_GAMES", "SKIP_BEFORE_DEBUT",
    "LAST_SEASON", "LAST_SEASON_STATS", "LAST_SEASON_MINUTES", "LAST_SEASON_STARTS",
    "LAST_SEASON_STARTS_GAMES", "GK_PRIOR_MINUTES", "GK_PRIOR_SCALE", "TEAM_PRIOR_GAMES",
    "TEAM_XG_WEIGHT", "SAVES_LEAK", "SAVES_LEAK_MODE", "EXACT_SAVES", "DECAY",
    "LONG_VIEW", "AVAILABILITY_OVERRIDES", "START_OVERRIDES",
]


def build_ref_cfg(engine_config):
    """Full explicit config = engine defaults <- stage 2 projection <- BASE <- sim pins."""
    cfg = dict(engine_config)
    cfg.update(STAGE2_PROJECTION)
    cfg.update(BASE)
    cfg.update(SIM_PINS)
    # Irrelevant to the sim (it never calls load_team) and not worth freezing.
    for k in ("TEAM_ID", "MY_SQUAD", "BANK", "SELLING_PRICES", "CHIPS_AVAILABLE"):
        cfg.pop(k, None)
    return cfg


# sim.py passes these three as its own explicit kwargs, so they must not also arrive
# via PROJ_SETTINGS. MAX_VIEW and SIM_PINS pin them to the same values either way.
SIM_OWNED = {"LONG_VIEW", "AVAILABILITY_OVERRIDES", "START_OVERRIDES"}


def projection_settings(cfg):
    """The PROJ_SETTINGS payload sim.py should be given for `cfg`. Shared with verify.py
    so the reference and the check can't drift apart."""
    return {k: cfg[k] for k in PROJ_KEYS if k in cfg and k not in SIM_OWNED}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def main():
    quick = "--quick" in sys.argv
    os.chdir(CUR)
    sys.path.insert(0, ".")

    import fpl_v2 as F
    ref_cfg = build_ref_cfg(F.CONFIG)
    (HERE / "ref_cfg.json").write_text(json.dumps(ref_cfg, indent=2, sort_keys=True))
    print(f"wrote ref_cfg.json ({len(ref_cfg)} settings)")

    # sim.py reads its projection settings from the environment at import time,
    # so these have to be in place before `import sim`.
    proj_settings = projection_settings(ref_cfg)
    os.environ["PROJ_SETTINGS"] = json.dumps(proj_settings)
    os.environ["MAX_LONG_VIEW"] = str(MAX_VIEW)
    cache_name = "proj_cache_%s_view%d.pkl" % (
        "_".join(str(v) for v in proj_settings.values()), MAX_VIEW)
    assert len(cache_name.encode()) < 250, f"cache filename too long: {len(cache_name)}"

    from season import Season
    S = Season()

    # ---- projections ----
    proj_cfg = dict(ref_cfg, LONG_VIEW=MAX_VIEW)
    manifest = {"config": "ref_cfg.json", "max_view": MAX_VIEW, "projections": {}}
    for t in REF_GWS:
        bs, fx = S.snapshot(t)
        t0 = time.time()
        df, gws, ts, fm, un = F.build_projections(bs, fx, proj_cfg, t, S.past)
        out = HERE / f"proj_gw{t}.csv"
        df.round(6).to_csv(out)
        manifest["projections"][t] = {"file": out.name, "rows": len(df),
                                      "sha256_16": sha(out), "gws": [gws[0], gws[-1]]}
        print(f"  GW{t:<3} {len(df)} rows  gws {gws[0]}-{gws[-1]}  "
              f"sha {manifest['projections'][t]['sha256_16']}  ({time.time() - t0:.1f}s)")

    # ---- reference sim ----
    import sim
    last_gw = 6 if quick else 38
    start = sim.initial_squad()
    t0 = time.time()
    res = sim.simulate(ref_cfg, label=REF_LABEL, start=list(start), last_gw=last_gw)
    secs = time.time() - t0
    print(f"\nreference sim: total={res['total']} transfers={res['transfers']} "
          f"hits={res['hits']} chips={res.get('chips')} ({secs:.0f}s)")

    manifest["sim"] = {"label": REF_LABEL, "last_gw": last_gw, "total": res["total"],
                       "transfers": res["transfers"], "hits": res["hits"],
                       "chips": res.get("chips"), "log": res["log"], "secs": round(secs, 1)}
    name = "manifest_quick.json" if quick else "manifest.json"
    (HERE / name).write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {name}")


if __name__ == "__main__":
    main()
