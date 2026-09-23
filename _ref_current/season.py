"""Rebuild FPL data as it looked at each 2025/26 deadline (no future leakage)."""
import pandas as pd, numpy as np

import os
D = os.environ.get("BT_DATA", "bt_data/")
STATS = ["minutes", "starts", "expected_goals", "goals_scored", "expected_assists", "assists",
         "bonus", "saves", "goals_conceded", "defensive_contribution", "yellow_cards", "expected_goals_conceded",
         "total_points"]


class Season:
    def __init__(self):
        m = pd.read_csv(D + "2025-26_gws_merged_gw.csv")
        self.teams = pd.read_csv(D + "2025-26_teams.csv")
        self.fx = pd.read_csv(D + "2025-26_fixtures.csv")
        self.players = pd.read_csv(D + "2025-26_players_raw.csv").set_index("id")
        name2id = dict(zip(self.teams["name"], self.teams["id"]))
        m["team_id"] = m["team"].map(name2id)
        assert m["team_id"].notna().all()
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
        # game-by-game minutes & starts (one row per fixture, oldest first) for the minutes model
        mm = m.sort_values(["GW", "fixture"])[["element", "GW", "minutes", "starts"]]
        self.games = {pid: (g["GW"].to_numpy(), g["minutes"].to_numpy(), g["starts"].to_numpy())
                      for pid, g in mm.groupby("element")}

    def _last_season(self):
        m = pd.read_csv(D + "2024-25_gws_merged_gw.csv")
        p = pd.read_csv(D + "2024-25_players_raw.csv").set_index("id")
        keys = [k for k in STATS if k in m and k not in ("total_points", "expected_goals_conceded")]
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
        # crude injury proxy (FPL's flags aren't in the archive): a player who has
        # been playing but missed his team's last 1-2 games is flagged doubtful.
        gone = set(self.gw.groupby("element")["GW"].max().loc[lambda x: x < t].index)
        prev = {g: self.gw[self.gw["GW"] == g].set_index("element")["minutes"] for g in (t - 1, t - 2)}
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
            e = {"id": int(pid), "web_name": pl["web_name"], "element_type": int(pl["element_type"]),
                 "team": int(row["team_id"]), "now_cost": int(row["value"]),
                 "status": "u" if pid in gone else ("a" if cop is None else "d"), "chance_of_playing_next_round": cop,
                 "news": "", "can_select": pid in now.index, "cost_change_start": 0,
                 "points_per_game": 0}
            for k in STATS:
                e[k] = (float(c[k]) if c is not None and k in c else 0)
            e["minutes"] = int(e["minutes"]); e["starts"] = int(e["starts"])
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
                 for c in ["wildcard", "freehit", "bboost", "3xc"] for s, e in [(1, 19), (20, 38)]]
        events = [{"id": g, "is_next": g == t} for g in range(1, 39)]
        return {"teams": teams, "elements": els, "events": events, "chips": chips}, fixtures
