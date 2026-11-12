"""Cross-check against circulax's ThermalPhaseShifter (forward configuration only).

tests/data/circulax_heater.json is written by tools/circulax_heater.py in the cspdk
venv (jax + circulax + sax). Circulax's heater is a VCVS, non-reciprocal at o2, so
only the forward MZI is comparable.
"""

from __future__ import annotations

import json
import pathlib

from test_heater import deck

DATA = pathlib.Path(__file__).parent / "data" / "circulax_heater.json"
TOL = 1e-7  # circulax's g_leak (1e-9) and Newton tolerances limit its accuracy


def vacask_powers(run_op, mod_dir, vh):
    v = run_op(
        deck(
            mod_dir,
            "s1 (a_i a_q) src p=1",
            "c1 (a_i a_q x_i x_q top_i top_q bot_i bot_q) cpl coupling=0.5",
            "t1 (x_i x_q) term",
            "h1 (top_i top_q m1_i m1_q le 0) heater",
            "w1 (bot_i bot_q m2_i m2_q) wg length=320 neff=2.38",
            "c2 (m1_i m1_q m2_i m2_q y1_i y1_q y2_i y2_q) cpl coupling=0.5",
            "d1 (y1_i y1_q p1 0) pd",
            "d2 (y2_i y2_q p2 0) pd",
            "r1 (p1 0) r r=1",
            "r2 (p2 0) r r=1",
            f"vh (le 0) vsrc dc={vh!r}",
        )
    )
    return [v["p1"], v["p2"]]


def test_heater_mzi_matches_circulax(run_op, vacask_mod_dir):
    ref = json.loads(DATA.read_text())
    worst = 0.0
    for vh, want in zip(ref["vh"], ref["p"], strict=True):
        got = vacask_powers(run_op, vacask_mod_dir, vh)
        worst = max(worst, *(abs(g - w) for g, w in zip(got, want, strict=True)))
    print(f"max |VACASK - circulax| over the sweep: {worst:.3e}")
    assert worst < TOL
