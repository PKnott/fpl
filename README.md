# FPL decision engine

An end-to-end decision system for Fantasy Premier League. It projects every
player's points to the end of the season, plans transfers several weeks ahead
with integer programming, picks the lineup and captain using Monte Carlo
simulation that allows for autosubs, and decides when to play each chip.

The same engine runs in two places:

- **Live:** each week it reads the official FPL API and produces a report for my team.
- **Backtest:** it replays full past seasons using only the data available before
  each deadline, so I can measure and tune decisions against what actually happened.

**Stack:** Python · pandas · NumPy · PuLP/CBC (mixed-integer programming) ·
Jupyter. About 3,000 lines across the engine, the backtest harness and the
analysis and verification tools.

---

## What it produces

An excerpt from a live weekly report (GW6, 2026-27):

```
 CHIPS  (compared across every week of each chip's window)
  Bench Boost: +8.6 now, this week #11 of 14 remaining (top 79%, playing if in top 43%)
               -> save - best remaining week is GW19 (+16.8)

 TRANSFERS
 0 transfer(s): +0.0 pts vs rolling
 1 transfer(s): +3.2 pts vs rolling
   OUT Gakpo (LIV, £7.2m)               IN Mbeumo (MUN, £7.9m)
 2 transfer(s)  (-6 hit): +1.7 pts vs rolling
 ...
>>> RECOMMENDED: 1 transfer(s)

 The plan ahead (only this week is locked in - rerun every week):
   GW6: 1 FT → Gakpo → Mbeumo        bank £1.3m
   GW7: 1 FT → João Pedro → Barry    bank £3.2m
   ...

 STARTING XI - GW6                 xP  If plays  Plays
  MID Mbeumo (C)                  5.8       5.8   100%
  DEF Konsa                       2.8       3.9    71%
  ...
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
- **Small samples are pulled towards a prior.** Every per-90 rate is a
  minutes-weighted blend of this season, last season and the position average.
  How much weight the past gets is set **per stat**, from measurement (see
  [When does form become signal?](#when-does-form-become-signal)): xG settles fast,
  while goals, assists, bonus and cards are mostly luck over a few weeks and
  lean much harder on last season.
- **Minutes use recency weighting.** Recent games are weighted with exponential
  decay (0.6 per game back, also tuned by backtest). This gives each player's
  chances of starting, of playing 60+ minutes and of appearing at all. Before a
  player's first game, last season's starts stand in. The official injury flags
  (live) or a recent-absence rule (backtest) then scale those chances.
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

The objective is the discounted expected points of the XI (each week counts 0.8
of the one before). On top of that it adds a half-weight "tail" out to 19 weeks
for longer-term squad value, the expected value of price changes, and a value for
each free transfer left at the end of the horizon.

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

FPL gives one of each chip per half-season (GW1–19 and GW20–38), and an unused
chip expires. So the question isn't "is this week good?" but "is this week good
enough, given the chances left?" Two rules answer it:

- **Triple Captain, Bench Boost and Free Hit: rank against the rest of the window.**
  Each is valued in every remaining week of its window, against a forecast of the
  squad I'd actually own then (the planner stepped forward one week at a time).
  The chip is played if this week ranks in the top `CHIP_TOP_PCT` share of what's
  left. The bar tightens by itself as the window runs out, and the chip is always
  played in its last week rather than wasted.
- **Wildcard: play when the squad has fallen far enough behind.** Each week the
  planner solves two plans: rebuild freely now, or carry on with normal transfers.
  The wildcard is played once the rebuild is ahead by `WILDCARD_GAP` (20 planner
  points — roughly 5 points a week of squad quality). A first-half wildcard is
  only credited up to GW19, because from GW20 a fresh one can rebuild anyway.

  The wildcard used the ranking rule until the backtest showed it was broken: the
  forecast assumes no injuries or form swings, so it always makes "now" look like
  the best time to rebuild. It played GW2 and GW20 in every season, at every
  threshold. See [the wildcard results](#the-wildcard).
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

Three seasons are replayed (2023-24 to 2025-26), from five different starting
squads each.

**Chips that don't change the squad are tuned by record and replay.** Re-simulating
a season for every candidate threshold would take hours per setting. Instead:

1. **Record:** simulate each (season × starting squad) once, playing no chips. For
   every week, save each chip's full opportunity table, what playing it there
   would *really* have scored, and the wildcard gap.
2. **Replay:** for each candidate rule, walk those saved tables. A whole grid
   takes seconds.

This is valid because Triple Captain and Bench Boost never change the squad, and
Free Hit changes it for one week only, so their payoffs don't depend on when they
were played. Every rule is compared on the same seasons and squads (a paired
comparison), with standard errors and a per-season breakdown.

**The wildcard needs full simulations**, because it changes every week after it.
Two experiments:

- **Forced week:** play it in a chosen week (every third week of each half, plus
  never), with the other half left to the rule. The planner is told in advance,
  so it doesn't take hits the week before a rebuild.
- **Rule comparison:** full seasons under each candidate rule.

---

## Results

### Chip thresholds

Real points each chip added per season, replayed from the 15 recordings (3 seasons ×
5 starting squads):

| Chip | Rule | Points per season | Notes |
|---|---|---|---|
| Free Hit | top 5% | 28.7 | At 0.15, two of three seasons lost points with it. |
| Triple Captain | top 15% | 27.7 | Anywhere from 0.10 to 0.25 is as good; the differences come down to one or two captains. Looser than 0.30 drops to 11. |
| Bench Boost | top 40% | 24.3 | At 0.15 it was nearly always played in the last week or two of its window, because the forecast always makes the bench look stronger later. 0.40 beat 0.15 in all three seasons. |

A fixed points bar ("play when worth at least N") did worse than the ranking rule
for all three chips (`run_backtest.py bars`).

### The wildcard

**No single week is reliably best.** Season totals with the first-half wildcard
forced into each week:

| Week | 2023-24 | 2024-25 | 2025-26 | Mean |
|---|---|---|---|---|
| Never | 2,258 | 2,376 | 2,059 | 2,231 |
| GW2 | **2,300** | 2,291 | 2,109 | 2,233 |
| GW5 | 2,247 | 2,145 | 2,069 | 2,154 |
| GW8 | 2,250 | 2,364 | **2,177** | 2,264 |
| GW11 | 2,278 | 2,336 | 2,083 | 2,232 |
| GW14 | 2,256 | **2,402** | 1,988 | 2,215 |
| GW17 | 2,272 | 2,355 | 1,991 | 2,206 |

Each season names a different best week, and one wildcard can move a season by
over 250 points, because it sends everything after it down a different path. In
the second half, GW20 had the lowest average of any option, below never playing
it at all.

**A signal beats the calendar.** Full seasons under each rule (starting squad 0):

| Rule | 2023-24 | 2024-25 | 2025-26 | Mean | Wildcards played |
|---|---|---|---|---|---|
| Rank against the window (any threshold) | 2,300 | 2,291 | 2,109 | 2,233 | GW2 and GW20, every time |
| Gap ≥ 15 | 2,300 | 2,266 | 2,155 | 2,240 | GW2, then GW20–24 |
| **Gap ≥ 20** | **2,383** | **2,314** | **2,178** | **2,292** | GW2–4, then GW23–33 |
| Gap ≥ 25 | 2,353 | 2,250 | 2,154 | 2,252 | GW2–5, second sometimes never |
| Gap ≥ 30 | 2,330 | 2,353 | 2,086 | 2,256 | often only fires once |

Gap ≥ 20 beat the old rule in all three seasons, by about 58 points a season. It
still rebuilds early, when the pre-season squad really is out of date, but holds
the second wildcard until the squad has actually drifted.

**It holds on the other starting squads.** Gap ≥ 20 against the old rule, all
five starting squads:

| Season | Gap 20 minus old rule, squads 0–4 | Mean |
|---|---|---|
| 2023-24 | +83, +85, +41, +55, +2 | +53 |
| 2024-25 | +23 ×5 (the early rebuild makes the squads identical) | +23 |
| 2025-26 | +69, +32, +67, −35, +70 | +41 |

It won 14 of 15 runs, by about **39 points a season** on average.

One caution: 20 was chosen from four bars on squad 0's seasons, which flatters
that table's margin. The other four squads weren't used to choose it, and the
gain there (+34 a season) is close to the overall figure. What makes it credible
is that it won in every season, on almost every squad, and the mechanism is
understood.

### When does form become signal?

`signal_noise.py` asks the archive directly. For each stat, at every week of
three seasons, it finds the weights for last season and the position average
that best predict that stat over the **rest** of the season, scored on held-out
seasons:

| Stat | Last-season weight (minutes) | Own data counts for half after | Error vs one shared weight |
|---|---|---|---|
| xG, xA | 720 (the default 540 kept: no measurable gain) | ~900 min (10 games) | about the same |
| Goals, assists | 2,700 | ~3,600–4,000 min | −22 to −25% |
| Bonus (outfield) | 1,800 | ~2,700 min | −23% |
| Yellow cards | 2,700 | ~4,000 min | −38% |
| Saves (GK) | 900 | ~1,000 min | −13% |

xG settles within about ten games; a player's goal or bonus tally over a few
weeks is mostly luck, and last season predicts the rest of this one better. The
fitted weights are in `DEFAULTS` as `STAT_PRIOR_MINUTES`.

The effect on **points** is small but consistent: six-week projection error falls
0.2–0.7% in every season (`projection_accuracy.py`). Points are driven mostly by
xG and minutes, which were already well calibrated, so the noisy stats only move
the margins.

### Season totals

With no chips at all, the engine scores 2,249–2,302 (2023-24), 2,342–2,400
(2024-25) and 2,002–2,038 (2025-26) across the five starting squads.

---

## What the backtest caught

Running the backtest end to end found problems that no single piece of code
showed on its own:

- **Every player projected zero at GW1.** The minutes model learns from this
  season's games, and before GW1 there are none, so the opening squad was
  effectively random and a GW2 wildcard replaced 14 of 15 players. Last season's
  starts now stand in until a player has played.
- **Later wildcard weeks lost by construction.** Projections stopped 24 weeks out,
  so a GW17 wildcard seen from GW2 was credited with 9 weeks of benefit and a GW3
  one with 19. Projections now run to the end of the season.
- **The forecast assumes no news.** Valuing future weeks from today's projections
  means nothing ever goes wrong in the future, so "now" always looks best for a
  wildcard and "later" always looks best for a Bench Boost. That is why the
  wildcard moved to a gap rule and the Bench Boost to a looser threshold.
- **A planner that didn't know a wildcard was coming** took four hits the week
  before a forced rebuild. Forced weeks are now planned in advance.

## Engineering notes

- **One engine, no copies.** The live assistant and the backtest used to carry
  their own copies of the engine. The settings had drifted apart (planning horizon
  5 vs 6 weeks, maximum transfers 11 vs 3, and others), so the backtest was tuning
  a different program from the one being run. Now both import `fpl_engine.py`.
  Settings are split into `DEFAULTS` (the model, which the backtest tunes) and
  `USER` (team ID, free transfers, injury overrides).
- **Refactors proven to change nothing.** `verify.py` runs the engine against
  frozen reference outputs: projections for five gameweeks across a season. Every
  setting is pinned from `reference/ref_cfg.json`, so a deliberate change of
  defaults can't be mistaken for a refactor bug, and the snapshots still match
  exactly. Its full-season check only applied until the chip rules were
  deliberately rewritten, so `--projections-only` is the check to run now.
- **Deliberate behaviour changes are recorded with their effect.** When the
  simulator switched to the autosub-aware lineup the live report uses, the
  2025-26 reference season moved from 2,022 to 2,030 points
  (`reference/baseline.md`). The old behaviour can still be reproduced with a
  config flag.
- **Caching and resumable runs.** All 38 gameweeks of projections for a season
  are cached to disk, keyed by a hash of the projection settings, and shared by
  every experiment. Simulations run in parallel across processes; each finished
  run is saved, so an interrupted experiment resumes where it stopped.

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
- Three seasons is a small sample for a decision made twice a season. The wildcard
  results are consistent across seasons and squads, but the size of the gain is
  an estimate, not a precise figure.
- The forecast behind the ranking rule still assumes no bad news. For the Bench
  Boost that bias is offset with a looser threshold rather than removed.

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
.venv/bin/python run_backtest.py record --squads 5 --workers 4   # 15 seasons, ~10 min each
.venv/bin/python run_backtest.py tune                            # chip thresholds, seconds
.venv/bin/python run_backtest.py bars --chip bboost              # points-bar rule, seconds
.venv/bin/python run_backtest.py gaps                            # wildcard gap by stage of season
.venv/bin/python run_backtest.py wildcard --workers 4            # forced-week experiment
.venv/bin/python run_backtest.py policy --gaps 15,20,25,30       # wildcard rule comparison
```

**Analysis:**

```bash
.venv/bin/python signal_noise.py          # per-stat weights for last season vs this season
.venv/bin/python projection_accuracy.py   # projected vs actual points, per config
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
run_backtest.py             command-line tool: record, tune and the wildcard experiments
signal_noise.py             when a player's form becomes signal, per stat
projection_accuracy.py      projected vs actual points for competing configs
FPL_Weekly_Assistant.ipynb  the weekly report
FPL_Backtest.ipynb          backtest and tuning walkthrough
verify.py                   regression check against frozen reference outputs
lineup_equivalence.py       vectorised lineup vs the original loop
reference/                  frozen outputs and config used by verify.py
_ref_current/               the pre-refactor code, kept as the verification baseline
```

`bt_data/` (archive CSVs) and `bt_cache/` (projection caches, recordings and
experiment results, a couple of GB) are gitignored and rebuilt on demand.
