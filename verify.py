"""Prove the refactored engine behaves exactly like the code it replaced.

NOTE: the full-season sim comparison is meaningful only up to step 3, the last
point at which behaviour was required to be identical. Step 4 changed the lineup
on purpose (2022 -> 2030) and step 6 replaced the chip rules, so the sim check
will differ from the reference after those. `--projections-only` stays valid
throughout and is the part to keep running: the projection model itself has not
changed and must not.

The comparison holds everything else still: it runs the ORIGINAL backtest harness
(_ref_current/sim.py, unmodified except for the import line) against the NEW
fpl_engine, using reference/ref_cfg.json so no setting falls through to a default.
Any difference is therefore an engine change, which at this stage there should be
none of.

Usage:  python verify.py [--quick] [--projections-only]
"""
import json
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
REF = ROOT / "reference"
CUR = ROOT / "_ref_current"
WORK = ROOT / "bt_cache" / "verify"


def make_harness():
    """Copy the original sim/season harness, repointed at the new engine.

    Only the import line and the config lookup change - the simulation loop,
    the scoring and the free-transfer accounting are byte-identical to the code
    that produced the reference.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copy(CUR / "season.py", WORK / "season.py")
    shutil.copy(ROOT / "fpl_engine.py", WORK / "fpl_engine.py")

    sim = (CUR / "sim.py").read_text()
    swaps = [("import fpl_v2 as F", "import fpl_engine as F"),
             ("dict(F.CONFIG,", "dict(F.resolve_cfg(),")]
    for old, new in swaps:
        assert old in sim, f"harness no longer contains {old!r}"
        sim = sim.replace(old, new)
    (WORK / "sim.py").write_text(sim)

    link = WORK / "bt_data"
    if not link.exists():
        link.symlink_to(ROOT / "bt_data")
    return swaps


def compare_projections(manifest, ref_cfg, S, F):
    import pandas as pd

    max_view = manifest["max_view"]
    proj_cfg = dict(ref_cfg, LONG_VIEW=max_view)
    ok = True
    for gw_str, meta in manifest["projections"].items():
        t = int(gw_str)
        bs, fx = S.snapshot(t)
        proj = F.build_projections(bs, fx, proj_cfg, t, S.past)
        got = proj.df.round(6)
        want = pd.read_csv(REF / meta["file"], index_col="id")
        got = got[want.columns]                     # column order only
        # Empty text ("news" for a fit player) survives the write as "" but reads
        # back as NaN, so normalise both sides rather than chase a false difference.
        for col in ("name", "team", "status", "news"):
            if col in want.columns:
                want[col] = want[col].fillna("").astype(str)
                got[col] = got[col].fillna("").astype(str)
        try:
            pd.testing.assert_frame_equal(got, want, check_dtype=False, rtol=0, atol=0)
            print(f"  GW{t:<3} MATCH   ({len(got)} rows x {len(got.columns)} cols)")
        except AssertionError as e:
            ok = False
            first = str(e).split("\n")[0:6]
            print(f"  GW{t:<3} DIFFERS\n      " + "\n      ".join(first))
    return ok


def main():
    quick = "--quick" in sys.argv
    proj_only = "--projections-only" in sys.argv
    manifest_name = "manifest_quick.json" if quick else "manifest.json"
    manifest = json.loads((REF / manifest_name).read_text())
    ref_cfg = json.loads((REF / "ref_cfg.json").read_text())

    swaps = make_harness()
    print("harness repointed at the new engine:")
    for old, new in swaps:
        print(f"    {old}  ->  {new}")

    # Build the PROJ_SETTINGS payload the same way the reference did.
    sys.path.insert(0, str(REF))
    from make_reference import projection_settings

    os.chdir(WORK)
    sys.path.insert(0, str(WORK))
    os.environ["PROJ_SETTINGS"] = json.dumps(projection_settings(ref_cfg))
    os.environ["MAX_LONG_VIEW"] = str(manifest["max_view"])

    import fpl_engine as F
    from season import Season
    S = Season()

    print("\nPROJECTIONS (must be identical)")
    proj_ok = compare_projections(manifest, ref_cfg, S, F)
    if proj_only:
        sys.exit(0 if proj_ok else 1)

    want = manifest["sim"]
    if "--new-backtest" in sys.argv:
        # Step 3 onwards: the rewritten harness, which calls E.decide_week instead
        # of restating the weekly decision. Same engine, same config, same answer.
        print("\nFULL-SEASON SIM via fpl_backtest.simulate (must be identical)")
        os.environ["BT_DATA"] = str(ROOT / "bt_data")
        os.environ["BT_CACHE"] = str(ROOT / "bt_cache")
        sys.path.insert(0, str(ROOT))
        import fpl_backtest as B
        source = B.ArchiveSource("2025-26")
        projector = B.Projector(source, ref_cfg, manifest["max_view"])
        t0 = time.time()
        got = B.simulate(projector, ref_cfg, label=want["label"],
                         start=list(B.initial_squad(projector, ref_cfg)),
                         last_gw=want["last_gw"])
        secs = time.time() - t0
    else:
        print("\nFULL-SEASON SIM via the original harness (must be identical)")
        import sim
        t0 = time.time()
        got = sim.simulate(ref_cfg, label=want["label"], start=list(sim.initial_squad()),
                           last_gw=want["last_gw"])
        secs = time.time() - t0

    log_match = [tuple(x) for x in got["log"]] == [tuple(x) for x in want["log"]]
    rows = [("total", want["total"], got["total"]),
            ("transfers", want["transfers"], got["transfers"]),
            ("hits", want["hits"], got["hits"]),
            ("chips", want["chips"], got.get("chips"))]
    sim_ok = log_match
    print(f"  {'':<12}{'reference':>28}{'new engine':>28}")
    for name, w, g in rows:
        # The manifest went through json(default=str), so numbers may have been
        # written as strings. Compare on the text, which is what we display.
        same = str(w) == str(g)
        sim_ok = sim_ok and same
        flag = "" if same else "   <-- DIFFERS"
        print(f"  {name:<12}{str(w)[:26]:>28}{str(g)[:26]:>28}{flag}")
    print(f"  {'week log':<12}{'':>28}{'identical' if log_match else 'DIFFERS':>28}")
    print(f"  ({secs:.0f}s)")

    print("\n" + ("PASS - the refactor changed nothing." if proj_ok and sim_ok
                  else "FAIL - see the differences above."))
    sys.exit(0 if proj_ok and sim_ok else 1)


if __name__ == "__main__":
    main()
