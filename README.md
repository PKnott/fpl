# FPL decision engine

An end-to-end decision system for Fantasy Premier League. It projects every
player's points for the next 24 gameweeks, plans transfers several weeks ahead
with integer programming, picks the lineup and captain using Monte Carlo
simulation that allows for autosubs, and decides when to play each chip.

The same engine runs in two places:

- **Live:** each week it reads the official FPL API and produces a report for my team.
- **Backtest:** it replays full past seasons using only the data available before
  each deadline, so I can measure and tune decisions against what actually happened.

**Stack:** Python · pandas · NumPy · PuLP/CBC (mixed-integer programming) ·
Jupyter. About 2,400 lines across the engine, the backtest harness and the
verification tools.

---

## What it produces

An excerpt from a live weekly report (GW6, 2026-27):

```
 CHIPS  (compared across every week of each chip's window)
  Bench Boost: +9.4 now, this week #13 of 14 remaining (top 93%, playing if in top 21%)
               -> save - best remaining week is GW17 (+17.1)

 TRANSFERS
 0 transfer(s): +0.0 pts vs rolling
 1 transfer(s): +3.8 pts vs rolling
   OUT Gakpo (LIV, £7.2m)               IN Mbeumo (MUN, £7.9m)
 2 transfer(s)  (-6 hit): +2.9 pts vs rolling
 ...
>>> RECOMMENDED: 1 transfer(s)

 The plan ahead (only this week is locked in - rerun every week):
   GW6: 1 FT → Gakpo → Mbeumo        bank £1.3m
   GW7: 1 FT → João Pedro → Barry    bank £3.2m
   ...

 STARTING XI - GW6                 xP  If plays  Plays
  MID Mbeumo (C)                  5.8       5.8   100%
  FWD João Pedro                  2.3       5.4    42%
  ...
  Projected GW6 score (including autosubs): 56.4
```

The report also covers:

- **Projection breakdowns** for any player. It shows each points component and
  where every per-90 rate came from: this season, last season or the position
  average, with the weight each one got.
- **A team-defence table** comparing each club's rating with what has actually
  happened.
- **Value per £m** against the average for each position.

---

## How it works

### 1. Points projection

Each player's expected points are built from the FPL scoring rules, one
component at a time:

- **Goals and clean sheets are Poisson.** A team's expected goals in a fixture
  combine its attack with the opponent's defence. Clean-sheet odds and the
  goals-conceded penalty come straight from that rate. The defensive-contribution
  bonus (10+ or 12+ actions) is a Poisson threshold too.
- **Attacking output blends xG with actual goals** (85/15, a setting tuned by
  backtest). Raw output alone overreacts to finishing luck.
- **Small samples are pulled towards a prior.** Every per-90 rate is weighted
  together with last season's rate and the position average, each counted as a
  fixed number of minutes. A player with 90 minutes is mostly prior; one with
  1,500 is mostly himself. Goalkeepers get a stronger prior, because saves and
  bonus are very noisy.
- **Minutes use recency weighting.** Recent games are weighted with exponential
  decay (0.6 per game back, also tuned by backtest). This gives each player's
  chances of starting, of playing 60+ minutes and of appearing at all. The
  official injury flags (live) or a recent-absence rule (backtest) then scale those
  chances.
- **Team strength starts from FPL's pre-season rating**, and that rating counts
  for less as real games come in.
- **Busy keepers make a defence look leakier.** A keeper facing a lot of shots is
  taken as extra evidence that his defence is weak, which lowers its clean-sheet
  odds.

### 2. Transfer planning (mixed-integer program)

Transfers are solved jointly over a 5-week horizon with PuLP/CBC. The binary
variables cover squad membership, buys, sells and the starting XI for each week.
The model enforces:

- the budget, using real selling prices (FPL keeps half of any profit)
- squad shape (2/5/5/3) and the limit of 3 players per club
- legal formations
- free-transfer roll-over, plus a cost for each extra transfer. FPL charges 4
  points; the planner charges 6, as a margin against projection noise.

The objective is the discounted expected points of the XI. On top of that it adds
a weighted "tail" for 19 weeks of longer-term squad value, the expected value of
price changes, and a value for each free transfer left at the end of the horizon.

The report shows the best plan for each number of transfers this week, from 0 to
11. Only the first week is acted on, and the plan is re-solved every week.

### 3. Lineup and captaincy (Monte Carlo)

Picking the XI on expected points is wrong when players might not play. A
doubtful player's expected points are low, but if he does play he scores his full
amount, and the bench covers him if he doesn't. The lineup step:

- simulates 4,000 draws of who plays
- applies FPL's autosub rules, with formation constraints
- passes the armband to the vice-captain when the captain doesn't play
- chooses the XI, bench order and captain that maximise expected points over all
  the draws

The draws are vectorised in NumPy, which runs about 50× faster than the original
loop. `lineup_equivalence.py` proves it gives the same answers: every difference
across 200 samples is an exact tie broken the other way.

### 4. Chip timing

A chip is worth nothing if it's never used, so the right question isn't "is this
week good?" It is "is this week good enough, given how many chances are left?"

- Each chip is valued in **every remaining week of its window**. Each week's value
  is measured against a **forecast of the squad I'd actually own then**: the
  planner steps forward one week at a time and keeps only its first week's moves.
- This week is ranked among those weeks. The chip is played if it lands in the top
  `CHIP_TOP_PCT` share. The bar tightens by itself as the window runs out and
  reaches certainty in the last week, so no hand-set point thresholds are needed.
- If two chips qualify in the same week, the one that loses most by waiting is
  played.

---

## Backtesting without look-ahead

`ArchiveSource` rebuilds what the FPL API would have shown before each deadline
of a past season, from the [vaastav archive](https://github.com/vaastav/Fantasy-Premier-League).
It only includes data from before that gameweek, so every decision could have
been made on the day. The simulator then applies the engine's decisions and
scores what really happened. It calls `decide_week`, the same function the live
report uses, so the backtest tests the program that actually runs.

**Tuning chips by recording and replaying.** Sweeping a chip threshold by
re-simulating whole seasons would cost hours per setting. Instead:

1. **Record:** simulate each (season × starting squad) once, with only the
   wildcard playable. For every week, save each chip's full opportunity table and
   what playing it there would *really* have scored.
2. **Replay:** for each candidate threshold, walk those saved tables. This takes
   seconds for the whole grid.

This is valid because Triple Captain and Bench Boost never change the squad, and
Free Hit changes it for one week only, so their payoffs don't depend on when they
were played. The wildcard does change the squad's path and is left out of the
replay. Every threshold is compared on the same seasons and squads (a paired
comparison), and results are reported with standard errors and broken down by
season.

### Results

There are 15 recorded seasons: 3 seasons × 5 starting squads, with only the
wildcard playable. Points are season totals before Triple Captain, Bench Boost
and Free Hit are added:

| Season | Season total (5 squads) | Wildcards played |
|---|---|---|
| 2023-24 | 2,245 – 2,249 | GW2, GW22 |
| 2024-25 | 2,236 – 2,245 | GW2, GW20 |
| 2025-26 | 2,025 – 2,068 | GW2, GW20 |

The next table shows the real points each chip added per season (both halves
combined) under different thresholds, replayed from those records. A threshold
`x` means "play when this week ranks in the top `x` share of the weeks left".
At `x = 0` only the single best remaining week qualifies.

| Chip | Best threshold | Points per season | At a loose threshold (0.30) |
|---|---|---|---|
| Triple Captain | 0.10 – 0.25 (flat) | 28.7 | 19.7 |
| Bench Boost | 0.00 – 0.25 (flat) | 23.9 | 19.7 |
| Free Hit | 0.00 – 0.05 | 27.7 | −10.7 |

What this shows:

- **Free Hit is the chip that rewards patience.** Holding it for the single best
  week earned about 28 points per season. Loosening the threshold to 0.15 cut
  that to about 1, and in two of the three seasons it lost points outright. The
  pattern holds in every season, so the Free Hit threshold is tightened to 0.05.
- **Triple Captain and Bench Boost are flat between 0.10 and 0.25.** The existing
  0.15 already sits in that range, so they are unchanged.
- **The samples are fewer than they look.** Every run plays its wildcard in GW2,
  which rebuilds the squad, so the five starting squads for a season end up
  nearly identical. The 15 runs are closer to 3 independent ones, one per season,
  and the standard errors in `run_backtest.py tune` overstate the precision.
  That's why thresholds only change when the effect is consistent across all
  three seasons, and why values are rounded rather than set to the exact
  grid-optimal number.
- **Final fixture lists flatter waiting**, as described under the limitations
  below. Free Hit is the most exposed, because blank gameweeks show up in the
  archive earlier than they did at the time. That's a reason to use 0.05 rather
  than 0.

---

## Engineering notes

- **One engine, no copies.** The live assistant and the backtest used to carry
  their own copies of the engine. The settings had drifted apart (planning horizon
  5 vs 6 weeks, maximum transfers 11 vs 3, and others), so the backtest was tuning
  a different program from the one being run. Now both import `fpl_engine.py`.
  Settings are split into `DEFAULTS` (the model, which the backtest tunes) and
  `USER` (team ID, free transfers, injury overrides).
- **Refactors proven to change nothing.** `verify.py` runs the engine against
  frozen reference outputs: projections for five gameweeks across a season, plus a
  full-season simulation. Every setting is pinned from `reference/ref_cfg.json`, so
  a deliberate change of defaults can't be mistaken for a refactor bug. The
  projection snapshots still match the reference exactly.
- **Deliberate behaviour changes are recorded with their effect.** When the
  simulator switched to the autosub-aware lineup the live report uses, the
  2025-26 reference season moved from 2,022 to 2,030 points
  (`reference/baseline.md`). The old behaviour can still be reproduced with a
  config flag.
- **Caching.** All 38 gameweeks of projections for a season are cached to disk,
  keyed by a hash of the projection settings. Recording runs in parallel across
  processes, and finished runs are skipped when a recording is resumed.

## Known limitations

- The archive's fixture list is the final one, so rescheduled double gameweeks are
  visible earlier in the backtest than they were at the time. That flatters
  waiting to play a chip.
- 2023-24 and 2024-25 are replayed under the **current** chip rules (a fresh set
  each half), so the seasons are comparable. This isn't a faithful replay of how
  those seasons worked.
- Those two seasons also have no defensive-contribution data. Results are
  consistent within each season but not across them, so they are reported per
  season as well as pooled.
- The archive has no injury flags. In the backtest, a regular starter who missed
  his team's last one or two games is treated as doubtful. Live runs use FPL's own
  flags.

---

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy "pulp==3.3.2" scipy requests pyarrow jupyter
```

PuLP is pinned because 4.0 no longer includes the CBC solver.

**Weekly report:** open `FPL_Weekly_Assistant.ipynb`, set `TEAM_ID` and
`FREE_TRANSFERS`, and run all cells.

**Backtest:**

```bash
.venv/bin/python run_backtest.py record --squads 5 --workers 4   # ~10 min per run, cached
.venv/bin/python run_backtest.py tune                            # seconds
```

**Verification:**

```bash
.venv/bin/python verify.py --projections-only   # projections vs reference/
.venv/bin/python lineup_equivalence.py 200      # vectorised lineup vs the original loop
```

## Repository layout

```
fpl_engine.py               projection model, planner, lineup, chips, weekly report
fpl_backtest.py             archive data source, season simulator, chip record/replay
run_backtest.py             command-line tool: record runs and tune chip thresholds
FPL_Weekly_Assistant.ipynb  the weekly report
FPL_Backtest.ipynb          backtest and tuning walkthrough
verify.py                   regression check against frozen reference outputs
lineup_equivalence.py       vectorised lineup vs the original loop
reference/                  frozen outputs and config used by verify.py
_ref_current/               the pre-refactor code, kept as the verification baseline
```

`bt_data/` (archive CSVs) and `bt_cache/` (about 500 MB of projection caches
and records) are gitignored and rebuilt on demand.
