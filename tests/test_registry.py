import numpy as np
import pytest

import socklight
from socklight import _bridge, registry
from socklight.models import STDLIB


@pytest.fixture(autouse=True)
def _fresh_registry():
    registry._reset()
    socklight.register_models("socklight.models", STDLIB)
    yield
    registry._reset()
    socklight.register_models("socklight.models", STDLIB)


def _spec(name):
    return {s[0]: s for s in _bridge.snapshot()}[name]


def test_stdlib_registered():
    assert _spec("socklight.models:mmi2x2")[1] == ("o1", "o2", "o3", "o4")


def test_param_kinds():
    kinds = {p[0]: p[1] for p in _spec("socklight.models:mmi2x2")[2]}
    assert kinds["loss_dB_cross"] == "opt"
    assert kinds["wl"] == "real"
    assert kinds["fwhm"] == "real"


def test_int_bool_str_kinds():
    def m(wl=1.55, n=2, flag=True, mode="a", arr=(1, 2)):
        return {("a", "b"): 1.0}

    socklight.register("t.m:m", m)
    kinds = {p[0]: p[1] for p in _spec("t.m:m")[2]}
    assert kinds == {"wl": "real", "n": "int", "flag": "int", "mode": "str"}


@pytest.mark.parametrize("name", ["nocolon", "1a:b", "a:", "a:b-c", "a:b..c", ":b"])
def test_bad_name_raises(name):
    with pytest.raises(ValueError):
        socklight.register(name, STDLIB["straight"])


def test_case_collision_raises():
    socklight.register("a.b:X", STDLIB["straight"])
    with pytest.raises(ValueError):
        socklight.register("a.b:x", STDLIB["straight"])


def test_param_case_collision_raises():
    def f(L=1.0, l=2.0):
        return {("a", "b"): 1.0}

    with pytest.raises(ValueError):
        socklight.register("a.b:f", f)


def test_required_param_raises():
    def f(x):
        return {("a", "b"): 1.0}

    with pytest.raises(ValueError):
        socklight.register("a.b:f", f)


def test_probe_failure_raises():
    def f(x=1.0):
        raise RuntimeError("boom")

    with pytest.raises(Exception):
        socklight.register("a.b:f", f)
    assert "a.b:f" not in {s[0] for s in _bridge.snapshot()}


def test_port_order_is_sorted():
    def f(wl=1.55):
        return {("out1", "in0"): 1.0, ("out0", "out1"): 2.0, ("in0", "out0"): 3.0}

    socklight.register("a.b:f", f)
    spec = _spec("a.b:f")
    assert spec[1] == ("in0", "out0", "out1")
    v = _bridge.evaluate("a.b:f", {"wl": 1.55})
    assert len(v) == 2 * 9
    assert v[2 * (2 * 3 + 0)] == 1.0  # (out1, in0)
    assert v[2 * (1 * 3 + 2)] == 2.0  # (out0, out1)
    assert v[2 * (0 * 3 + 1)] == 3.0  # (in0, out0)


def test_evaluate_matches_model():
    v = _bridge.evaluate("socklight.models:mmi2x2", {"wl": 1.55})
    expected = STDLIB["mmi2x2"](wl=1.55)[("o1", "o3")]
    idx = 2 * (0 * 4 + 2)
    assert v[idx] == pytest.approx(complex(expected).real)
    assert v[idx + 1] == pytest.approx(complex(expected).imag)
    assert all(type(x) is float for x in v)


def test_evaluate_opt_and_int():
    seen = {}

    def f(wl=1.55, k=None, n=3):
        seen.update(k=k, n=n)
        return {("a", "b"): 1.0}

    socklight.register("a.b:f", f)
    _bridge.evaluate("a.b:f", {"wl": 1.5, "n": 4.0})
    assert seen == {"k": None, "n": 4} and isinstance(seen["n"], int)
    _bridge.evaluate("a.b:f", {"k": 0.5})
    assert seen["k"] == 0.5


def test_evaluate_port_mismatch_raises():
    state = {"extra": False}

    def f(wl=1.55):
        s = {("a", "b"): 1.0}
        if state["extra"]:
            s[("a", "c")] = 1.0
        return s

    socklight.register("a.b:f", f)
    state["extra"] = True
    with pytest.raises(ValueError):
        _bridge.evaluate("a.b:f", {})


def test_register_models_aliases():
    socklight.register_models("p.q", {"alias": STDLIB["straight"]})
    assert "p.q:alias" in {s[0] for s in _bridge.snapshot()}


def test_frozen_new_name_raises():
    _bridge.snapshot()
    with pytest.raises(RuntimeError, match="restart"):
        socklight.register("a.b:new", STDLIB["straight"])


def test_frozen_changed_signature_raises():
    def f(wl=1.55):
        return {("a", "b"): 1.0}

    socklight.register("a.b:f", f)
    _bridge.snapshot()

    def g(wl=1.55, extra=1.0):
        return {("a", "b"): 1.0}

    with pytest.raises(RuntimeError):
        socklight.register("a.b:f", g)


def test_frozen_same_signature_replaces():
    def f(wl=1.55):
        return {("a", "b"): 1.0}

    def g(wl=1.55):
        return {("a", "b"): 2.0}

    socklight.register("a.b:f", f)
    _bridge.snapshot()
    socklight.register("a.b:f", g)
    assert _bridge.evaluate("a.b:f", {})[2] == 2.0  # (a, b)


def test_evaluate_error_sets_last_error():
    state = {"bad": False}

    def exploding_model(wl=1.55):
        if state["bad"]:
            raise KeyError("nope")
        return {("a", "b"): 1.0}

    socklight.register("a.b:f", exploding_model)
    assert socklight.last_error() is None
    state["bad"] = True
    with pytest.raises(KeyError):
        _bridge.evaluate("a.b:f", {"wl": 1.4})
    err = socklight.last_error()
    assert err["model"] == "a.b:f"
    assert err["params"] == {"wl": 1.4}
    assert "exploding_model" in err["traceback"]


def test_load_error():
    assert socklight.load_error() is None
    _bridge.set_load_error("bad")
    assert socklight.load_error() == "bad"
