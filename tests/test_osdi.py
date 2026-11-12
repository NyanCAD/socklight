"""socklight.osdi loaded by VACASK: one device per registered model.

Each bench drives one port with an ideal source (``dc=1`` on ``_i``, ``0`` on
``_q``) and terminates every other port in its matched 1 ohm load on both
rails, so ``a_k = 0`` off the driven port ``d`` and ``V_d = a_d (1 + S_dd) = 1``:
``V_k = b_k = S_kd / (1 + S_dd)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import MANY_PORTS, NPARAMS_MANY, VacaskError, elaborate, many_column, two_port

import socklight
from socklight import registry

TOL = 1e-9
MMI2X2 = "socklight.models:mmi2x2"


def bench(tag: str, model: str, ports, params: str = "", driven: str | None = None) -> str:
    driven = driven or ports[0]
    nodes = " ".join(f"{tag}{p}_i {tag}{p}_q" for p in ports)
    lines = [
        f"x{tag} ({nodes}) {model} {params}",
        f"v{tag}i ({tag}{driven}_i 0) vsrc dc=1",
        f"v{tag}q ({tag}{driven}_q 0) vsrc dc=0",
    ]
    for p in ports:
        if p != driven:
            lines += [f"r{tag}{p}i ({tag}{p}_i 0) r r=1", f"r{tag}{p}q ({tag}{p}_q 0) r r=1"]
    return "\n".join(lines)


def deck(mod_dir: str, models: dict[str, str], *benches: str) -> str:
    """``models`` maps a VACASK model name to ``"'resource:path' params"``."""
    head = [
        "socklight test",
        "ground 0",
        f'load "{socklight.osdi_path()}"',
        f'load "{mod_dir}/resistor.osdi"',
        "model vsrc vsource",
        "model r resistor",
    ]
    head += [f"model {name} {rest}" for name, rest in models.items()]
    return "\n".join([*head, *benches]) + "\n"


def wave(v: dict[str, float], node: str) -> complex:
    return complex(v[f"{node}_i"], v[f"{node}_q"])


def expected_column(name: str, driven: str = "o1", **kw) -> dict[str, complex]:
    spec, fn = registry._models[name]
    s = fn(**kw)
    sdd = complex(np.asarray(s.get((driven, driven), 0.0)).reshape(-1)[0])
    return {
        p: complex(np.asarray(s.get((p, driven), 0.0)).reshape(-1)[0]) / (1 + sdd)
        for p in spec.ports
        if p != driven
    }


def check_column(v, tag, name, driven="o1", **kw):
    for p, want in expected_column(name, driven, **kw).items():
        got = wave(v, f"{tag}{p}")
        assert abs(got - want) < TOL, (name, p, got, want)


def ports_of(name):
    return registry._models[name][0].ports


def test_loads_and_parses(run_op, vacask_mod_dir):
    ports = ports_of(MMI2X2)
    elaborate(deck(vacask_mod_dir, {"mmi": f"'{MMI2X2}'"}, bench("a", "mmi", ports, "wl=1.55")))


@pytest.mark.parametrize("model", ["mmi2x2", "coupler", "mmi1x2", "straight"])
def test_column_matches_model(run_op, vacask_mod_dir, model):
    name = f"socklight.models:{model}"
    v = run_op(deck(vacask_mod_dir, {"dev": f"'{name}'"}, bench("a", "dev", ports_of(name), "wl=1.55")))
    check_column(v, "a", name, wl=1.55)


def test_port_order_is_sorted(run_op, vacask_mod_dir):
    assert ports_of("t.m:sorted_ports") == ("in0", "out0", "out1")
    v = run_op(deck(vacask_mod_dir, {"sp": "'t.m:sorted_ports'"}, bench("a", "sp", ("in0", "out0", "out1"))))
    assert abs(wave(v, "aout0") - 0.5) < TOL
    assert abs(wave(v, "aout1") - 0.25j) < TOL


def test_str_and_int_params(run_op, vacask_mod_dir):
    ports = ("in0", "out0", "out1")
    v = run_op(deck(vacask_mod_dir, {"sp": "'t.m:sorted_ports'"}, bench("a", "sp", ports, 'mode="b" n=2')))
    assert abs(wave(v, "aout0") - 0.4) < TOL
    assert abs(wave(v, "aout1") - 0.5j) < TOL


def test_param_typo_is_parse_error(run_op, vacask_mod_dir):
    ports = ports_of(MMI2X2)
    with pytest.raises(VacaskError):
        elaborate(deck(vacask_mod_dir, {"mmi": f"'{MMI2X2}'"}, bench("a", "mmi", ports, "loss_dbx=0.3")))


def test_two_instances_independent(run_op, vacask_mod_dir):
    ports = ports_of(MMI2X2)
    v = run_op(
        deck(
            vacask_mod_dir,
            {"mmi": f"'{MMI2X2}'"},
            bench("a", "mmi", ports, "wl=1.50"),
            bench("b", "mmi", ports, "wl=1.55"),
        )
    )
    check_column(v, "a", MMI2X2, wl=1.50)
    check_column(v, "b", MMI2X2, wl=1.55)
    assert abs(wave(v, "ao3") - wave(v, "bo3")) > 1e-6


def test_model_line_defaults_and_override(run_op, vacask_mod_dir):
    ports = ports_of(MMI2X2)
    v = run_op(
        deck(
            vacask_mod_dir,
            {"m": f"'{MMI2X2}' wl=1.50 loss_db=0.5"},
            bench("a", "m", ports),
            bench("b", "m", ports, "wl=1.55"),
        )
    )
    check_column(v, "a", MMI2X2, wl=1.50, loss_dB=0.5)
    check_column(v, "b", MMI2X2, wl=1.55, loss_dB=0.5)


def test_reregister_same_signature_takes_effect(run_op, vacask_mod_dir):
    net = deck(vacask_mod_dir, {"f": "'t.m:f'"}, bench("a", "f", ("o1", "o2")))
    assert abs(wave(run_op(net), "ao2") - 0.5) < TOL
    try:
        socklight.register("t.m:f", two_port(0.25))
        assert abs(wave(run_op(net), "ao2") - 0.25) < TOL
    finally:
        socklight.register("t.m:f", two_port(0.5))


def test_setup_failure_sets_last_error(run_op, vacask_mod_dir):
    def net(x):
        return deck(vacask_mod_dir, {"b": "'t.m:bad'"}, bench("a", "b", ("o1", "o2"), f"x={x}"))

    with pytest.raises(VacaskError):
        run_op(net(2))
    err = socklight.last_error()
    assert err["model"] == "t.m:bad"
    assert err["params"] == {"x": 2.0}
    assert "bad" in err["traceback"]
    assert abs(wave(run_op(net(0.5)), "ao2") - 0.5) < TOL


def test_many_ports_and_params(run_op, vacask_mod_dir):
    p = [0.0] * NPARAMS_MANY
    p[45] = 2.0  # model line
    p[5], p[25], p[99] = 10.0, 3.0, 7.0  # instance
    v = run_op(deck(vacask_mod_dir, {"mp": "'t.m:many' p45=2 p5=-1"}, bench("a", "mp", MANY_PORTS, "p5=10 p25=3 p99=7")))
    col = many_column(p)
    for k in range(1, len(MANY_PORTS)):
        assert abs(wave(v, f"a{MANY_PORTS[k]}") - col[k]) < TOL, k


def test_load_error_none_when_ok(run_op, vacask_mod_dir):
    run_op(deck(vacask_mod_dir, {"f": "'t.m:f'"}, bench("a", "f", ("o1", "o2"))))
    assert socklight.load_error() is None


def test_missing_terminals_fail(run_op, vacask_mod_dir):
    nodes = " ".join(f"n{k}" for k in range(6))
    net = deck(vacask_mod_dir, {"mmi": f"'{MMI2X2}'"}, f"xa ({nodes}) mmi\nvai (n0 0) vsrc dc=1")
    with pytest.raises(VacaskError):
        run_op(net)
    err = socklight.last_error()
    assert err["model"] == MMI2X2
    assert "connected 6 of 8 terminals" in err["traceback"]


def test_non_finite_s_fails(run_op, vacask_mod_dir):
    with pytest.raises(VacaskError):
        run_op(deck(vacask_mod_dir, {"nf": "'t.m:nonfinite'"}, bench("a", "nf", ("o1", "o2"), "x=1")))
    err = socklight.last_error()
    assert err["model"] == "t.m:nonfinite"
    assert "('o1', 'o2') is not finite" in err["traceback"] or "('o2', 'o1') is not finite" in err["traceback"]
