# FPL

One engine, two front ends.

```
fpl_engine.py               the only engine code that exists
fpl_backtest.py             archive data, simulation, chip record/replay
run_backtest.py             record phase + chip tuning driver
FPL_Weekly_Assistant.ipynb  your weekly report
FPL_Backtest.ipynb          tuning
verify.py                   proves a refactor changed nothing
lineup_equivalence.py       checks the vectorised lineup against the original
reference/                  frozen outputs the verification compares against
_ref_current/               the pre-refactor code, kept only as a reference
```

Both notebooks download `fpl_engine.py` from this repo, so what the backtest tunes
is what runs on a Saturday. Previously each notebook carried its own copy; the
bodies were identical but the settings had drifted (`PLAN_WEEKS` 5 vs 6,
`LONG_VIEW` 19 vs 18, `DECAY` 0.8 vs 0.75, `MAX_TRANSFERS` 11 vs 3, `XG_WEIGHT`
0.85 vs 0.9, `PRIOR_MINUTES` 90 vs 270, `MINUTES_GAMES` 5 vs 10), so the backtest
was tuning a different program than the one being run.

## Settings

`DEFAULTS` is the model, the planner and the chip rules — what the backtest tunes.
`USER` is your team id, free transfers, injury overrides, locks and excludes — what
you edit each week. `resolve_cfg()` merges them.

Three horizons that used to be one setting:

| | |
|---|---|
| `PLAN_WEEKS` 5 | weeks of transfers solved together |
| `LONG_VIEW` 19 | weeks used for a squad's long-term value |
| `PROJECTION_WEEKS` 24 | how far points are projected at all |

Chips use none of these — a chip's horizon is its expiry. 24 is needed because
valuing a wildcard near a window's end looks past the planner's slice.

## Chips

A chip is worth nothing unused, so the question is not "is this week good?" but
"is it good enough, given how many chances are left?". Each chip is valued the
same way in every remaining week of its window, this week is ranked among them,
and it is played if it lands in the top `CHIP_TOP_PCT[chip]`. The bar tightens by
itself as the window runs out and reaches certainty in the last week. When two
chips qualify at once, the one that loses most by waiting is played.

Values are measured against `forecast_squads` — the team you would actually own in
each future week, stepped forward one planner solve at a time.

## Running things

```bash
python -m venv .venv && .venv/bin/pip install pandas numpy pulp scipy requests
.venv/bin/python run_backtest.py record --squads 5 --workers 8   # ~10 min per run
.venv/bin/python run_backtest.py tune                            # seconds
```

Recording runs one season per (season, starting squad) with only the wildcard
playable, saving each chip's opportunity table *and* what it really paid. Tuning
replays those tables. That works because Triple Captain and Bench Boost never
change the squad and Free Hit changes it for a single week — so their payoffs do
not depend on when they were played. The wildcard does change the trajectory and
needs full simulations.

## Verification

```bash
.venv/bin/python verify.py --new-backtest   # projections + season total vs reference/
.venv/bin/python lineup_equivalence.py 200  # vectorised lineup vs the original loop
```

`verify.py` pins every setting from `reference/ref_cfg.json` so a deliberate change
of defaults is never mistaken for a refactor bug. It was used to prove steps 2 and
3 changed nothing; after step 4 the season total moves by design (2022 → 2030),
and `LINEUP="best_xi"` reproduces the old number.

## Caveats

- The archive's fixture list is the final one, so rescheduled double gameweeks are
  visible earlier here than they were at the time. That flatters waiting.
- 2023-24 and 2024-25 are replayed under **current** chip rules (a fresh set each
  half). It makes the seasons comparable; it is not a faithful replay of those
  seasons.
- Those two seasons also have no defensive-contribution data, and did not score
  it. Consistent within each season, not comparable across them — results are
  reported per season as well as pooled.
- The archive has no injury flags, so a player who has been playing and missed his
  team's last one or two games is treated as doubtful. Live runs get FPL's real
  flags.
