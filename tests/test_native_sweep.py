"""VACASK's own sweeps over socklight parameters.

A sweep reads each swept parameter back through OSDI access() before it starts and
writes it back afterwards, so ungiven parameters must hold their real values (the
signature default, or the model line's value) rather than 0 / NULL.
"""

from __future__ import annotations

from conftest import analyse, elaborate
from test_osdi import MMI2X2, TOL, bench, deck, expected_column, ports_of


def point(res: dict, i: int) -> dict[str, float]:
    return {vec: float(v[i]) for vec, v in res.items()}


def check(v: dict[str, float], name: str, tag: str = "a", driven: str = "o1", **kw) -> None:
    for p, want in expected_column(name, driven, **kw).items():
        got = complex(v[f"{tag}{p}_i"], v[f"{tag}{p}_q"])
        assert abs(got - want) < TOL, (name, kw, p, got, want)


def test_wl_sweep_with_wl_on_the_model_line(run_op, vacask_mod_dir):
    cir, _keep = elaborate(deck(vacask_mod_dir, {"m": f"'{MMI2X2}' wl=1.52"}, bench("a", "m", ports_of(MMI2X2))))
    wls = [1.50, 1.55, 1.60]
    res = analyse(cir, {"instance": "xa", "parameter": "wl", "values": wls})
    for i, wl in enumerate(wls):
        check(point(res, i), MMI2X2, wl=wl)
    check(point(analyse(cir), 0), MMI2X2, wl=1.52)  # restored to the model line's value


def test_op_after_sweeping_an_ungiven_param_is_unchanged(run_op, vacask_mod_dir):
    cir, _keep = elaborate(deck(vacask_mod_dir, {"m": f"'{MMI2X2}'"}, bench("a", "m", ports_of(MMI2X2))))
    before = point(analyse(cir), 0)
    check(before, MMI2X2)
    res = analyse(cir, {"instance": "xa", "parameter": "fwhm", "values": [0.1, 0.3]})
    check(point(res, 0), MMI2X2, fwhm=0.1)
    check(point(res, 1), MMI2X2, fwhm=0.3)
    after = point(analyse(cir), 0)
    assert all(abs(after[k] - before[k]) < TOL for k in before)


def test_string_sweep_first_point(run_op, vacask_mod_dir):
    ports = ("in0", "out0", "out1")
    cir, _keep = elaborate(deck(vacask_mod_dir, {"sp": "'t.m:sorted_ports'"}, bench("a", "sp", ports)))
    res = analyse(cir, {"instance": "xa", "parameter": "mode", "values": ["b", "a"]})
    assert abs(res["aout0_i"][0] - 0.4) < TOL  # the default is "a" (0.5)
    assert abs(res["aout0_i"][1] - 0.5) < TOL


def test_opt_param_stays_omitted_after_a_sweep(run_op, vacask_mod_dir):
    # t.m:optional: S21 = 0.5 when y is None, else 0.1 y (so y=0 is not None)
    cir, _keep = elaborate(deck(vacask_mod_dir, {"o": "'t.m:optional'"}, bench("a", "o", ("o1", "o2"))))
    assert abs(analyse(cir)["ao2_i"][0] - 0.5) < TOL
    res = analyse(cir, {"instance": "xa", "parameter": "y", "values": [1.0, 2.0]})
    assert abs(res["ao2_i"][0] - 0.1) < TOL
    assert abs(res["ao2_i"][1] - 0.2) < TOL
    assert abs(analyse(cir)["ao2_i"][0] - 0.5) < TOL  # y is None again
