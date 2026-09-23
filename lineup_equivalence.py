"""Check the vectorised pick_lineup against the original loop implementation.

Not a rewrite compared to itself: this imports the untouched _ref_current/fpl_v2.py
and runs its `pick_lineup` (the 4,000-iteration Python loop) beside the new one on
the same squads and gameweeks, with the same random draws.

Usage:  python lineup_equivalence.py [n_samples]
"""
import json
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("BT_DATA", str(ROOT / "bt_data"))
os.environ.setdefault("BT_CACHE", str(ROOT / "bt_cache"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_ref_current"))

import numpy as np                                            # noqa: E402
import fpl_engine as E                                        # noqa: E402
import fpl_backtest as B                                      # noqa: E402
import fpl_v2 as OLD                                          # noqa: E402


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ref_cfg = json.loads((ROOT / "reference" / "ref_cfg.json").read_text())
    max_view = json.loads((ROOT / "reference" / "manifest.json").read_text())["max_view"]

    source = B.ArchiveSource("2025-26")
    projector = B.Projector(source, ref_cfg, max_view)
    rng = random.Random(0)

    # Sample real squads: the opening squad, then squads the planner actually
    # reaches, so the comparison covers lineups with genuine injury doubts.
    base_squad = B.initial_squad(projector, ref_cfg)

    rows, xi_same, cap_same, worst = [], 0, 0, 0.0
    t_old = t_new = 0.0
    for n in range(n_samples):
        gw = rng.randint(2, 38)
        proj, bs = projector.week(gw, max_view, ref_cfg["DECAY"])
        # Perturb the squad so we are not measuring the same 15 every time, while
        # keeping it legal: swap a few players for others in the same position.
        squad = list(base_squad)
        for _ in range(rng.randint(0, 6)):
            k = rng.randrange(15)
            p = proj.df.loc[squad[k], "pos"]
            cands = proj.df[(proj.df["pos"] == p) & proj.df["can_select"]].index
            pick = int(rng.choice(list(cands)))
            if pick not in squad:
                squad[k] = pick

        # The old implementation reads the module-level EXPLAIN global, so it has
        # to be filled by the old engine's own build_projections call.
        bs_s, fx_s = source.snapshot(gw)
        OLD.build_projections(bs_s, fx_s, dict(ref_cfg, LONG_VIEW=max_view), gw, source.past)

        t0 = time.perf_counter()
        old = OLD.pick_lineup(proj.df, squad, gw)
        t_old += time.perf_counter() - t0

        t0 = time.perf_counter()
        new = E.pick_lineup(proj, squad, gw)
        t_new += time.perf_counter() - t0

        d = abs(old["total"] - new["total"])
        worst = max(worst, d)
        xi_same += set(old["xi"]) == set(new["xi"])
        cap_same += (old["cap"], old["vice"]) == (new["cap"], new["vice"])
        rows.append((gw, d, set(old["xi"]) == set(new["xi"])))

    print(f"samples                    {n_samples}")
    print(f"identical XI               {xi_same}/{n_samples}  ({xi_same / n_samples:.1%})")
    print(f"identical captain & vice   {cap_same}/{n_samples}  ({cap_same / n_samples:.1%})")
    print(f"max |difference| in xP     {worst:.2e}")
    print(f"time  loop {t_old:.1f}s   vectorised {t_new:.1f}s   "
          f"speedup {t_old / max(t_new, 1e-9):.0f}x")

    # A different XI is only acceptable when the two score the same to floating-point
    # noise: the summation order changed, so exact ties can break either way. A real
    # disagreement would show up as a gap far larger than 1e-9.
    TIE = 1e-9
    differing = [r for r in rows if not r[2]]
    ties = [r for r in differing if r[1] < TIE]
    if differing:
        print(f"\nXI differed in {len(differing)} sample(s), of which {len(ties)} are exact "
              f"ties (gap < {TIE:g}); largest gap among them {max(r[1] for r in differing):.2e}")
    # The armband can swap for the same reason, so judge on value, not on labels:
    # if both implementations agree on expected points everywhere, they agree.
    ok = worst < TIE and len(ties) == len(differing)
    print("\n" + ("PASS - equivalent; every difference is a tie broken the other way."
                  if ok else "REVIEW - differences above are not pure float noise."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
