"""Replay past seasons with the engine making every decision.

Nothing in here knows how to project points, plan transfers or time chips - that
is all fpl_engine. This module only supplies data as it looked before a past
deadline (`ArchiveSource`), applies what the engine decided, and scores what
really happened.
"""
import hashlib
import json
import os
import pathlib
import pickle
import time

import numpy as np
import pandas as pd

import fpl_engine as E

DATA = pathlib.Path(os.environ.get("BT_DATA", "bt_data"))
CACHE = pathlib.Path(os.environ.get("BT_CACHE", "bt_cache"))
ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"

STATS = ["minutes", "starts", "expected_goals", "goals_scored", "expected_assists",
         "assists", "bonus", "saves", "goals_conceded", "defensive_contribution",
         "yellow_cards", "expected_goals_conceded", "total_points"]

# The season before each one, for the last-season stats blend.
PREV_SEASON = {"2023-24": "2022-23", "2024-25": "2023-24", "2025-26": "2024-25"}

# Chip windows. The archive doesn't record them, and older seasons ran under
# different rules (one of each chip per season, not per half). Every season is
# deliberately replayed under the CURRENT rules - a fresh set per half - because
# chips only change how actual points are counted, and it makes the seasons
# comparable. Not a faithful replay of 2023-24 or 2024-25.
CHIP_WINDOWS = [(1, 19), (20, 38)]


def download(season):
    """Fetch this season's archive files (and its predecessor's) if not already local."""
    DATA.mkdir(parents=True, exist_ok=True)
    wanted = {f"{season}_gws_merged_gw.csv": f"{season}/gws/merged_gw.csv",
              f"{season}_fixtures.csv": f"{season}/fixtures.csv",
              f"{season}_teams.csv": f"{season}/teams.csv",
              f"{season}_players_raw.csv": f"{season}/players_raw.csv"}
    prev = PREV_SEASON.get(season)
    if prev:
        wanted[f"{prev}_gws_merged_gw.csv"] = f"{prev}/gws/merged_gw.csv"
        wanted[f"{prev}_players_raw.csv"] = f"{prev}/players_raw.csv"
    import urllib.request
    for local, remote in wanted.items():
        if not (DATA / local).exists():
            urllib.request.urlretrieve(ARCHIVE + remote, DATA / local)
            print("downloaded", local)


class ArchiveSource:
    """Rebuilds what the FPL API would have shown before each deadline of a past season.

    Same contract as the live API: `snapshot(gw)` returns (bootstrap, fixtures).
    Only data from strictly before gameweek `gw` is included, so a decision made
    here could have been made on the day.
    """

    def __init__(self, season="2025-26", prev_season=None):
        self.season = season
        self.prev_season = prev_season or PREV_SEASON.get(season)
        m = pd.read_csv(DATA / f"{season}_gws_merged_gw.csv")
        self.teams = pd.read_csv(DATA / f"{season}_teams.csv")
        self.fx = pd.read_csv(DATA / f"{season}_fixtures.csv")
        self.players = pd.read_csv(DATA / f"{season}_players_raw.csv").set_index("id")
        name2id = dict(zip(self.teams["name"], self.teams["id"]))
        m["team_id"] = m["team"].map(name2id)
        assert m["team_id"].notna().all(), f"{season}: unmapped team names"
        self.m = m
        # per player per GW (double gameweeks summed)
        agg = {k: "sum" for k in STATS if k in m}
        agg.update(value="last", team_id="last")
        self.gw = m.groupby(["element", "GW"]).agg(agg).reset_index()
        self.actual = self.gw.set_index(["element", "GW"])["total_points"]
        self.minutes = self.gw.set_index(["element", "GW"])["minutes"]
        self.value = self.gw.set_index(["element", "GW"])["value"]
        self.fpl_xp = m.groupby(["element", "GW"])["xP"].sum()   # FPL's own prediction
        self.past = self._last_season()
        # game-by-game minutes & starts (one row per fixture, oldest first)
        mm = m.sort_values(["GW", "fixture"])[["element", "GW", "minutes", "starts"]]
        self.games = {pid: (g["GW"].to_numpy(), g["minutes"].to_numpy(), g["starts"].to_numpy())
                      for pid, g in mm.groupby("element")}

    def _last_season(self):
        if not self.prev_season:
            return {}
        m = pd.read_csv(DATA / f"{self.prev_season}_gws_merged_gw.csv")
        p = pd.read_csv(DATA / f"{self.prev_season}_players_raw.csv").set_index("id")
        keys = [k for k in STATS if k in m
                and k not in ("total_points", "expected_goals_conceded")]
        tot = m.groupby("element")[keys].sum()
        tot["code"] = p.loc[tot.index, "code"].values
        code2id = dict(zip(self.players["code"], self.players.index))
        past = {}
        for _, r in tot.iterrows():
            pid = code2id.get(r["code"])
            if pid is not None and r["minutes"] >= 600:
                past[pid] = {k: float(r[k]) for k in keys}
        return past

    def snapshot(self, t):
        """bootstrap-static + fixtures as known before the GW t deadline."""
        hist = self.gw[self.gw["GW"] < t]
        cum = hist.groupby("element")[[k for k in STATS if k in hist]].sum()
        now = self.gw[self.gw["GW"] == t].set_index("element")
        last = self.gw[self.gw["GW"] <= t].sort_values("GW").groupby("element").last()
        # Crude injury proxy: FPL's flags aren't in the archive, so a player who
        # has been playing but missed his team's last 1-2 games is called doubtful.
        gone = set(self.gw.groupby("element")["GW"].max().loc[lambda x: x < t].index)
        prev = {g: self.gw[self.gw["GW"] == g].set_index("element")["minutes"]
                for g in (t - 1, t - 2)}
        els = []
        for pid in last.index:
            if pid not in self.players.index:
                continue
            pl = self.players.loc[pid]
            c = cum.loc[pid] if pid in cum.index else None
            row = now.loc[pid] if pid in now.index else last.loc[pid]
            cop = None
            if c is not None and c["starts"] >= 2:
                m1, m2 = prev[t - 1].get(pid), prev[t - 2].get(pid)
                if m1 == 0 and m2 == 0:
                    cop = 25
                elif m1 == 0:
                    cop = 75
            e = {"id": int(pid), "web_name": pl["web_name"],
                 "element_type": int(pl["element_type"]), "team": int(row["team_id"]),
                 "now_cost": int(row["value"]),
                 "status": "u" if pid in gone else ("a" if cop is None else "d"),
                 "chance_of_playing_next_round": cop, "news": "",
                 "can_select": pid in now.index, "cost_change_start": 0,
                 "points_per_game": 0}
            for k in STATS:
                e[k] = (float(c[k]) if c is not None and k in c else 0)
            e["minutes"] = int(e["minutes"])
            e["starts"] = int(e["starts"])
            if pid in self.games:
                gws_, mins_, st_ = self.games[pid]
                k = int(np.searchsorted(gws_, t))          # games before GW t only
                e["recent"] = [{"minutes": int(x), "starts": int(y)}
                               for x, y in zip(mins_[max(0, k - 12):k], st_[max(0, k - 12):k])]
            els.append(e)
        teams = self.teams[["id", "name", "short_name", "strength"]].to_dict("records")
        fixtures = []
        for _, f in self.fx.iterrows():
            done = f["event"] < t
            fixtures.append({"event": int(f["event"]), "team_h": int(f["team_h"]),
                             "team_a": int(f["team_a"]), "finished": bool(done),
                             "team_h_score": int(f["team_h_score"]) if done else None,
                             "team_a_score": int(f["team_a_score"]) if done else None})
        chips = [{"name": c, "start_event": s, "stop_event": e}
                 for c in E.CHIP_NAMES for s, e in CHIP_WINDOWS]
        events = [{"id": g, "is_next": g == t} for g in range(1, 39)]
        return {"teams": teams, "elements": els, "events": events, "chips": chips}, fixtures


# ---------------------------------------------------------------------
# Projections, built once per season and reused by every run
# ---------------------------------------------------------------------
def cache_key(payload):
    """Stable across dict ordering - the old key joined `.values()` in insertion
    order, so reordering a settings dict silently reused the wrong cache."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Projector:
    """All 38 gameweeks of projections for one season, cached to disk."""

    def __init__(self, source, proj_cfg, weeks):
        self.source = source
        self.cfg = dict(proj_cfg, LONG_VIEW=weeks)
        self.weeks = weeks
        CACHE.mkdir(parents=True, exist_ok=True)
        key = cache_key({"season": source.season, "weeks": weeks, "cfg": self.cfg})
        self.path = CACHE / f"proj_{source.season}_{key}.pkl"
        self.data = self._build()

    def _build(self):
        if self.path.exists():
            with open(self.path, "rb") as fh:
                return pickle.load(fh)
        out = {}
        for t in range(1, 39):
            bs, fx = self.source.snapshot(t)
            out[t] = (E.build_projections(bs, fx, self.cfg, t, self.source.past), bs)
        with open(self.path, "wb") as fh:
            pickle.dump(out, fh)
        return out

    def week(self, t, view, decay):
        """This week's projections, with long-term value re-weighted for one run.

        The returned Projections keeps the FULL projection horizon, because chips
        look further ahead than the planner: valuing a wildcard near the end of a
        window needs weeks the planner's LONG_VIEW slice does not contain. The
        planner is handed `gws[:view]` separately.
        """
        base, bs = self.data[t]
        df = base.df.copy()
        plan_gws = base.gws[:view]
        df["xp_long"] = sum(df[f"gw{g}"] * decay ** i for i, g in enumerate(plan_gws))
        proj = E.Projections(df, base.gws, base.ts, base.fx_map, base.unscheduled,
                             base.explain)
        return proj, bs, plan_gws


# ---------------------------------------------------------------------
# Scoring what really happened
# ---------------------------------------------------------------------
def actual_score(source, pos, squad, xi, cap, vice, bench, t, chip):
    """Real FPL points for a fielded team, including autosubs and the armband.

    `pos` is the position lookup for the week. Bench Boost plays all 15; otherwise
    anyone on 0 minutes is replaced from the bench in order, provided the result is
    still a legal formation.
    """
    mins = lambda i: source.minutes.get((i, t), 0)
    pts = lambda i: source.actual.get((i, t), 0)
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


# ---------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------
def lineup(proj, squad, t, cfg):
    """Pick the XI the way the live report does.

    Set LINEUP="best_xi" to fall back to the old deterministic pick, which ignores
    each player's chance of playing. That is what the backtest used to do at every
    call site while the live report used pick_lineup, so the lineup logic that
    actually runs on a Saturday was never tested. Kept only for before/after work.
    """
    if cfg.get("LINEUP") == "best_xi":
        return E.best_xi(proj.df, squad, t)
    return E.pick_lineup(proj, squad, t)


def initial_squad(projector, cfg):
    """GW1 opening squad: the best £100.0m 15 on long-term projected points."""
    base, _ = projector.data[1]
    d = base.df.copy()
    d["gwinit"] = d["xp_long"]
    squad, _ = E.free_hit_squad(d, "init", 100.0, cfg)
    return squad


def simulate(projector, cfg, label="", start=None, last_gw=38, verbose=False):
    """Replay a season. The engine decides; this function only applies and scores.

    Every decision comes from `E.decide_week`, the same call the live report makes.
    Previously this loop restated the whole sequence itself and the two had already
    drifted - most visibly, the report picked its XI with the autosub-aware
    `pick_lineup` while this used `best_xi`, so the backtest never tested the
    lineup logic that actually runs.
    """
    cfg = dict(cfg)                  # FREE_TRANSFERS is rewritten each week
    source = projector.source
    squad = list(start) if start else initial_squad(projector, cfg)
    view = min(cfg.get("LONG_VIEW", 12), projector.weeks)
    if cfg.get("PLAN_WEEKS", 5) > view:
        raise ValueError(f"PLAN_WEEKS ({cfg['PLAN_WEEKS']}) can't exceed LONG_VIEW ({view})")

    purchase = {i: projector.data[1][0].df.loc[i, "price"] for i in squad}
    bank = round(100.0 - sum(purchase.values()), 1)
    ft, used, total, hits_total, log = 1, set(), 0, 0, []
    # Best one-week squads depend only on (gameweek, budget), so they are shared
    # across every week of the season rather than re-solved for each.
    fh_cache = {}
    chip_tables = {}

    for t in range(1, last_gw + 1):
        proj, bs, plan_gws = projector.week(t, view, cfg["DECAY"])
        df = proj.df
        pos = df["pos"]

        if t == 1:                                    # opening squad, no transfers
            lu = lineup(proj, squad, t, cfg)
            total += actual_score(source, pos, squad, lu["xi"], lu["cap"], lu["vice"],
                                  lu["bench"], t, None)
            continue

        sell = {i: E.selling_price(purchase[i], df.loc[i, "price"]) for i in squad}
        chips = E.chips_left(bs, t, used) if cfg.get("USE_CHIPS", True) else []
        cfg["FREE_TRANSFERS"] = ft

        d = E.decide_week(proj, plan_gws, bs, squad, bank, sell, cfg, chips,
                          fh_cache=fh_cache)
        chip = d.chip
        if d.advice:
            # Keep every chip's opportunity table: tuning the top-x% rule later
            # replays these instead of re-running the season for each value.
            chip_tables[t] = {c: {"values": dict(v["values"]), "rank": v["rank"],
                                  "n": v["n"], "played": c == chip}
                              for c, v in d.advice.items()}
        if chip:
            used.add((chip, E.chip_window(bs, chip, t)))

        if chip == "freehit":
            # A borrowed squad: score it, then hand the real one back untouched.
            lu = lineup(proj, d.squad, t, cfg)
            total += actual_score(source, pos, d.squad, lu["xi"], lu["cap"], lu["vice"],
                                  lu["bench"], t, None)
            ft = E.free_transfers_after(ft, 0, 0, chip="freehit")
            log.append((t, "FREE HIT", 0))
            continue

        for o in d.outs:
            bank += sell[o]
            purchase.pop(o, None)
        for i in d.ins:
            bank -= df.loc[i, "price"]
            purchase[i] = df.loc[i, "price"]
        bank = round(bank, 1)
        squad = d.squad
        ft = E.free_transfers_after(ft, len(d.ins), d.hits, chip=chip)

        lu = lineup(proj, squad, t, cfg)
        gw_pts = actual_score(source, pos, squad, lu["xi"], lu["cap"], lu["vice"],
                              lu["bench"], t, chip)
        total += gw_pts - cfg["HIT_COST_REAL"] * d.hits
        hits_total += d.hits
        log.append((t, chip or "", len(d.ins), d.hits))
        if verbose:
            print(t, chip, len(d.ins), d.hits, gw_pts, total, bank, flush=True)

    chip_weeks = {}
    for l in log:
        if l[1]:
            chip_weeks.setdefault(l[1], []).append(l[0])
    chips_txt = ", ".join(f"{E.CHIP_NAMES.get(c, c)} GW{'/'.join(str(g) for g in w)}"
                          for c, w in sorted(chip_weeks.items()))
    return {"label": label, "total": total, "hits": hits_total,
            "transfers": sum(l[2] for l in log if len(l) > 3),
            "chips": chips_txt, "log": log, "chip_tables": chip_tables}


def run_grid(projector, runs, base_cfg, results_file, last_gw=38):
    """Run a list of (label, overrides), saving after each so a drop can resume."""
    results_file = pathlib.Path(results_file)
    start = initial_squad(projector, base_cfg)
    out = json.loads(results_file.read_text()) if results_file.exists() else []
    done = {r.get("key") for r in out if "total" in r}
    for label, over in runs:
        cfg = dict(base_cfg, **over)
        key = cache_key({"label": label, "cfg": cfg, "season": projector.source.season,
                         "last_gw": last_gw})
        if key in done:
            print(f"{label}: already done with these exact settings, skipping", flush=True)
            continue
        t0 = time.time()
        try:
            r = simulate(projector, cfg, label, start=list(start), last_gw=last_gw)
        except Exception as e:
            r = {"label": label, "error": repr(e)}
        r["secs"] = round(time.time() - t0)
        r["key"] = key
        r["season"] = projector.source.season
        out = [o for o in out if o.get("key") != key] + [r]
        results_file.write_text(json.dumps(out, default=str))
        print(f"{label}: {r.get('total')} pts, {r.get('transfers')} transfers, "
              f"{r.get('hits')} hits  ({r['secs']}s)", flush=True)
    return out
