# Sim baselines, 2025-26, reference config

| After step | Sim total | What changed |
|---|---|---|
| step 1-3 | 2022 | frozen reference; refactor verified to change nothing |
| step 4 | **2030** | sim switched from `best_xi` to the autosub-aware `pick_lineup` the live report uses |

Reproduce the old number at any time with `LINEUP="best_xi"` in the run config.
Every setting tuned before step 4 is provisional: they were tuned against a sim
that picked its XI differently from the program being run.
