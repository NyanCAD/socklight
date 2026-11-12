"""VACASK helpers and the test-only models the OSDI tests load.

A process loads ``socklight.osdi`` once, and its constructor snapshots the
registry then, so every model an OSDI test uses is registered here, by the
session fixture ``osdi_models``, before the first circuit loads the file.
Tests that use ``run_op`` run last (see ``pytest_collection_modifyitems``):
``test_registry.py`` resets the registry around each of its tests, which must
not happen after the snapshot.
"""

from __future__ import annotations

import os

import pytest

import socklight


class VacaskError(RuntimeError):
    """A netlist failed to parse, elaborate or simulate."""


# ── test-only models ───────────────────────────────────────────────────────────


def sorted_ports(wl=1.55, mode="a", n=1):
    """Ports declared out of order; ``mode``/``n`` exercise str and int parameters."""
    t = 0.5 if mode == "a" else 0.4
    return {
        ("out1", "in0"): 0.25j * n, ("in0", "out1"): 0.25j * n,
        ("out0", "in0"): t, ("in0", "out0"): t,
    }


def two_port(s21):
    def f(wl=1.55):
        return {("o1", "o2"): s21, ("o2", "o1"): s21}

    return f


def bad(x=0.0):
    if x > 1:
        raise ValueError("bad model: x must be <= 1")
    return {("o1", "o2"): 0.5, ("o2", "o1"): 0.5}


def optional(y=None):
    t = 0.5 if y is None else 0.1 * y
    return {("o1", "o2"): t, ("o2", "o1"): t}


def nonfinite(x=0.0):
    t = float("nan") if x > 0 else 0.5
    return {("o1", "o2"): t, ("o2", "o1"): t}


NPORTS_MANY = 20
NPARAMS_MANY = 100
MANY_PORTS = [f"o{k:02d}" for k in range(NPORTS_MANY)]


def many_column(p):
    """S[o_k, o00] for the many-port model, given its 100 parameter values."""
    return [0.0] + [0.03 * k + 0.001 * sum(p[k::NPORTS_MANY]) for k in range(1, NPORTS_MANY)]


def _many(**p):
    col = many_column([p[f"p{j}"] for j in range(NPARAMS_MANY)])
    s = {(MANY_PORTS[k], MANY_PORTS[0]): col[k] for k in range(1, NPORTS_MANY)}
    s.update({(MANY_PORTS[0], MANY_PORTS[k]): col[k] for k in range(1, NPORTS_MANY)})
    s[(MANY_PORTS[1], MANY_PORTS[1])] = 0.0  # every port appears in some key
    for k in range(2, NPORTS_MANY):
        s[(MANY_PORTS[k], MANY_PORTS[k])] = 0.0
    return s


# A real signature (inspect.signature) with p0..p99, all defaulting to 0.0.
exec(  # noqa: S102
    "def many(" + ", ".join(f"p{j}=0.0" for j in range(NPARAMS_MANY)) + "):\n    return _many(" + ", ".join(f"p{j}=p{j}" for j in range(NPARAMS_MANY)) + ")\n"
)

TEST_MODELS = {
    "t.m:sorted_ports": sorted_ports,
    "t.m:f": two_port(0.5),
    "t.m:bad": bad,
    "t.m:nonfinite": nonfinite,
    "t.m:optional": optional,
    "t.m:many": many,  # noqa: F821  (defined by exec above)
}


@pytest.fixture(scope="session")
def osdi_models():
    for name, fn in TEST_MODELS.items():
        socklight.register(name, fn)


def pytest_configure(config):
    # Registered here, not in pyproject.toml: the Pyodide wheel test copies only tests/.
    config.addinivalue_line("markers", "native: needs a native interpreter (subprocess, a jax venv); skipped in Pyodide")


def pytest_collection_modifyitems(config, items):
    items.sort(key=lambda item: "run_op" in getattr(item, "fixturenames", ()))


# ── VACASK ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def vacask_mod_dir():
    import vacask

    return os.path.join(os.path.dirname(vacask.__file__), "mod")


def elaborate(netlist: str, variables: dict | None = None):
    """Parses and elaborates ``netlist``. ``variables`` declares circuit variables (what a
    ``var`` line in a control block does under the vacask CLI) before elaboration."""
    from vacask import Circuit, OpenvafCompiler, Parser, ParserTables, Status, Value
    from vacask import simulator as sim

    sim.setup()
    s = Status()
    tab = ParserTables("models")
    if not Parser(tab).parseNetlistString(netlist):
        raise VacaskError("netlist does not parse")
    if not tab.verify(s):
        raise VacaskError(s.message())
    compiler = OpenvafCompiler()
    cir = Circuit(tab, compiler, s)
    if not cir.isValid():
        raise VacaskError(s.message())
    for name, value in (variables or {}).items():
        if not cir.setVariable(name, Value(value))[0]:
            raise VacaskError(f"cannot set circuit variable {name!r}")
    if not cir.elaborate([], "__topdef__", "__topinst__", status=s):
        raise VacaskError(s.message())
    # The circuit refers to the parser tables and the compiler but does not keep them alive.
    return cir, (tab, compiler, s)


def analyse(cir, sweep: dict | None = None, name: str = "op1") -> dict:
    """Runs an operating point on an elaborated circuit, optionally under a VACASK sweep.

    ``sweep`` holds the sweep's parameters, e.g. ``{"instance": "xa", "parameter": "wl",
    "values": [1.5, 1.55]}``. Returns ``{vector: numpy array}``, one entry per point.
    """
    from vacask import PV, Analysis, PTAnalysis, PTSave, PTSweep, Status
    from vacask.rawfile import rawread

    s = Status()
    desc = PTAnalysis(name, "op")  # the analysis refers to it: keep it alive until run() is done
    if sweep:
        sw = PTSweep("sweep1")
        for key, value in sweep.items():
            sw.add(PV(key, value))
        desc.add(sw)
    analysis = Analysis.create(desc, cir, s)
    if not analysis:
        raise VacaskError(s.message())
    analysis.add(PTSave("default"))
    ok, _ = analysis.run(s)
    if not ok:
        raise VacaskError(s.message())
    res = rawread(name + ".raw").get(sweeps=1 if sweep else 0)
    return {vec: res[vec] for vec in res.names}


def _op(netlist: str) -> dict[str, float]:
    cir, _keep = elaborate(netlist)
    return {vec: float(v[0]) for vec, v in analyse(cir).items()}


@pytest.fixture
def run_op(osdi_models, tmp_path, monkeypatch):
    """``run_op(netlist) -> {node: value}``: operating point, run in ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    return _op
