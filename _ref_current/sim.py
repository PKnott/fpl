"""Stage 2 - replay 2025/26 with the planner making every decision."""
import sys, time, pickle, os, json
sys.path.insert(0, ".")
import pandas as pd, numpy as np
from season import Season
import fpl_v2 as F

S = Season()
PROJ_SETTINGS = json.loads(os.environ.get("PROJ_SETTINGS", "{}")) or dict(
    XG_WEIGHT=0.85, PRIOR_MINUTES=90, LAST_SEASON_STATS=True, LAST_SEASON_MINUTES=540,
    LAST_SEASON_STARTS=False, GK_PRIOR_MINUTES=360, SAVES_LEAK=0.5,
    SAVES_LEAK_MODE="shots", TEAM_PRIOR_GAMES=8, MINUTES_DECAY=0.6)
MAX_VIEW = int(os.environ.get("MAX_LONG_VIEW", "18"))   # longest LONG_VIEW any run uses
PROJ_CFG = dict(F.CONFIG, **PROJ_SETTINGS, LONG_VIEW=MAX_VIEW, PRICE_CHANGES=False,
                AVAILABILITY_OVERRIDES={}, START_OVERRIDES={}, LOCK=[], EXCLUDE=[],
                HIT_COST_REAL=4, SOLVER_SECONDS=20)
CACHE = "proj_cache_%s_view%d.pkl" % ("_".join(str(v) for v in PROJ_SETTINGS.values()), MAX_VIEW)


def projections():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    out = {}
    for t in range(1, 39):
        bs, fx = S.snapshot(t)
        # GW1 has no data yet, so lean on last season for the opening squad only
        cfg = dict(PROJ_CFG)
        df, gws, ts, fm, un = F.build_projections(bs, fx, cfg, t, S.past)
        out[t] = (df, gws, bs)
    pickle.dump(out, open(CACHE, "wb"))
    return out


PROJ = projections()


def initial_squad():
    df, gws, bs = PROJ[1]
    d = df.copy()
    d["gwinit"] = d["xp_long"]
    squad, _ = F.free_hit_squad(d, "init", 100.0, PROJ_CFG)
    return squad


def actual_score(squad, xi, cap, vice, bench, t, chip):
    mins = lambda i: S.minutes.get((i, t), 0)
    pts = lambda i: S.actual.get((i, t), 0)
    pos = PROJ[t][0]["pos"]
    if chip == "bboost":
        players = list(squad)
    else:
        players = list(xi)
        for b in bench:                               # auto-subs, in bench order
            for s in [s for s in players if mins(s) == 0]:
                if mins(b) == 0:
                    break
                trial = [p for p in players if p != s] + [b]
                cnt = {k: sum(pos[p] == k for p in trial) for k in (1, 2, 3, 4)}
                if cnt[1] == 1 and cnt[2] >= 3 and cnt[3] >= 2 and cnt[4] >= 1:
                    players = trial
                    break
    total = sum(pts(i) for i in players)
    armband = cap if mins(cap) > 0 else vice
    total += (2 if chip == "3xc" else 1) * pts(armband)
    return total


def simulate(cfg_over, label="", verbose=False, start=None, last_gw=38):
    cfg = dict(PROJ_CFG, **cfg_over)
    squad = start or initial_squad()
    val = lambda i, t: S.value.get((i, t), None)
    purchase = {i: PROJ[1][0].loc[i, "price"] for i in squad}
    bank = round(100.0 - sum(purchase.values()), 1)
    ft, used, total, log = 1, set(), 0, []
    hits_total = 0
    view = min(cfg.get("LONG_VIEW", 12), MAX_VIEW)
    if cfg.get("PLAN_WEEKS", 5) > view:
        raise ValueError(f"PLAN_WEEKS ({cfg['PLAN_WEEKS']}) can't exceed LONG_VIEW ({view})")
    for t in range(1, last_gw + 1):
        df, gws, bs = PROJ[t]
        df = df.copy()
        gws = gws[:view]
        df["xp_long"] = sum(df[f"gw{g}"] * cfg["DECAY"] ** i for i, g in enumerate(gws))
        chip = None
        if t == 1:                                    # opening squad, no transfers
            lu = F.best_xi(df, squad, t)
            total += actual_score(squad, lu["xi"], lu["cap"], lu["vice"], lu["bench"], t, None)
            continue
        now_price = {i: df.loc[i, "price"] for i in squad}
        sell = {i: F.selling_price(purchase[i], now_price[i]) for i in squad}
        half = 1 if t <= 19 else 2
        chips = [c for c in F.CHIP_NAMES if (c, half) not in used]
        cfg["FREE_TRANSFERS"] = ft
        pool, _ = F.solver_pool(df, squad, cfg)
        plans = {}
        for n in range(0, cfg["MAX_TRANSFERS"] + 1):
            p = F.plan_transfers(df, gws, squad, bank, sell, cfg, n_first=n, pool=pool)
            if p:
                plans[n] = p
        best_n = max(plans, key=lambda n: plans[n]["objective"])
        plan = plans[best_n]
        advice = F.chip_advice(df, gws, bs, chips, plan, squad, bank, sell, cfg, {}, [], pool) \
            if cfg.get("USE_CHIPS", True) and chips else {}
        chip = next((c for c, r in advice.items() if r[0]), None)

        if chip == "freehit" and advice["freehit"][3]:
            fh = advice["freehit"][3]
            lu = F.best_xi(df, fh, t)
            total += actual_score(fh, lu["xi"], lu["cap"], lu["vice"], lu["bench"], t, None)
            used.add(("freehit", half))
            ft = min(5, ft)                           # saved transfers kept
            log.append((t, "FREE HIT", 0))
            continue
        if chip == "wildcard" and advice["wildcard"][3]:
            wk = advice["wildcard"][3]["weeks"][0]
            hits = 0
            used.add(("wildcard", half))
        else:
            wk = plan["weeks"][0]
            hits = wk["hits"]
            if chip == "wildcard":
                chip = None
        outs, ins = wk["out"], wk["in"]
        for o in outs:
            bank += sell[o]; purchase.pop(o, None)
        for i in ins:
            bank -= df.loc[i, "price"]; purchase[i] = df.loc[i, "price"]
        bank = round(bank, 1)
        squad = wk["squad"]
        if chip == "wildcard":
            ft = min(5, ft)
        else:
            ft = min(5, max(1, ft - (len(ins) - hits) + 1))
        if chip in ("3xc", "bboost"):
            used.add((chip, half))
        lu = F.best_xi(df, squad, t)
        gw_pts = actual_score(squad, lu["xi"], lu["cap"], lu["vice"], lu["bench"], t, chip)
        total += gw_pts - cfg["HIT_COST_REAL"] * hits
        hits_total += hits
        log.append((t, chip or "", len(ins), hits))
        if verbose:
            print(t, chip, len(ins), hits, gw_pts, total, bank, flush=True)
    chip_weeks = {}
    for l in log:
        if l[1]:
            chip_weeks.setdefault(l[1], []).append(l[0])
    chips_txt = ", ".join(f"{F.CHIP_NAMES.get(c, c)} GW{'/'.join(str(g) for g in w)}"
                          for c, w in sorted(chip_weeks.items()))
    return {"label": label, **cfg_over, "total": total, "hits": hits_total,
            "transfers": sum(l[2] for l in log if len(l) > 3), "chips": chips_txt, "log": log}



def run_grid(runs, results_file="stage2_results.json", last_gw=38):
    start = initial_squad()
    out = json.load(open(results_file)) if os.path.exists(results_file) else []
    key = lambda label, c: json.dumps([label, c, PROJ_SETTINGS], sort_keys=True, default=str)
    done = {r.get("key") for r in out if "total" in r}
    for label, c in runs:
        if key(label, c) in done:
            print(f"{label}: already done with these exact settings, skipping", flush=True)
            continue
        t0 = time.time()
        try:
            r = simulate(c, label, start=list(start), last_gw=last_gw)
        except Exception as e:
            r = {"label": label, "error": repr(e)}
        r["secs"] = round(time.time() - t0)
        r["key"] = key(label, c)
        out = [o for o in out if o.get("key") != r["key"]] + [r]
        json.dump(out, open(results_file, "w"), default=str)
        print(f"{label}: {r.get('total')} pts, {r.get('transfers')} transfers, "
              f"{r.get('hits')} hits  ({r['secs']}s)", flush=True)
    return out
