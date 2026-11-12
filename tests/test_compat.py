"""socklight.compat: jax and sax served from numpy and socklight, for PDK model code without jax."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

import socklight
import socklight.compat as compat

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
CHILD = os.path.join(HERE, "compat_pdk", "child.py")
SHIM_NAMES = ("jax", "jax.numpy", "jax.typing", "jaxtyping", "sax", "sax.models")


@pytest.fixture(autouse=True)
def _clean():
    """Every test starts and ends without the shim and without compat_pdk imported."""
    real = [n for n in ("jax", "jaxlib", "sax") if n in sys.modules]
    if real:
        pytest.skip(f"real {real} imported in this process")
    yield
    compat.uninstall()
    for name in [n for n in sys.modules if n == "compat_pdk" or n.startswith("compat_pdk.")]:
        del sys.modules[name]


def test_import_socklight_does_not_install():
    assert not compat.installed()
    assert not any(n in sys.modules for n in SHIM_NAMES)


@pytest.mark.parametrize("name", ["jax", "jaxlib", "sax"])
def test_refuses_when_real_module_imported(monkeypatch, name):
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    with pytest.raises(RuntimeError, match=f"real {name} already imported"):
        compat.install()
    assert not compat.installed()


def test_idempotent():
    compat.install()
    import jax

    compat.install()
    assert sum(isinstance(f, compat._Finder) for f in sys.meta_path) == 1
    assert importlib.import_module("jax") is jax


def test_uninstall():
    compat.install()
    import jax.numpy as jnp
    import sax.models  # noqa: F401

    assert jnp is np
    compat.uninstall()
    assert not compat.installed()
    assert not any(n in sys.modules for n in SHIM_NAMES)
    assert np.__spec__.name == "numpy"
    compat.uninstall()  # again: a no-op


def test_jax_surface():
    compat.install()
    import jax
    import jax.numpy as jnp
    from jax.typing import ArrayLike  # noqa: F401
    from jaxtyping import Array, Float  # noqa: F401

    assert jnp is np and jax.numpy is np
    assert np.__spec__.name == "numpy"

    def f(x=1.0):
        return x

    assert jax.jit(f) is f
    assert jax.jit(static_argnames=("x",))(f) is f
    jax.config.update("jax_enable_x64", True)
    assert jax.config.jax_enable_x64 is True
    with jax.ensure_compile_time_eval():
        pass
    assert not isinstance(np.zeros(2), jax.Array)
    with pytest.raises(AttributeError, match="jax.grad is not provided"):
        jax.grad  # noqa: B018
    with pytest.raises(ImportError, match="jax.scipy is not provided"):
        import jax.scipy  # noqa: F401
    with pytest.raises(ImportError, match="jaxlib is not provided"):
        import jaxlib  # noqa: F401


def test_sax_surface():
    compat.install()
    import sax

    from socklight import _sax

    assert sax.reciprocal is _sax.reciprocal
    assert sax.EPS == _sax.EPS and sax.WL_C == 1.55 and sax.C_M_S == 299792458.0
    assert sax.SDict != sax.SType
    assert float | sax.FloatArray is not None  # usable in annotations at runtime
    assert sax.models.straight(wl=1.55)[("in0", "out0")] is not None  # `import sax` brings sax.models
    with pytest.raises(NotImplementedError, match="composed by VACASK"):
        sax.circuit({}, models={})
    from sax import circuit  # importing it works; calling it does not

    with pytest.raises(NotImplementedError):
        circuit()
    with pytest.raises(ImportError, match="composed by VACASK"):
        import sax.circuits  # noqa: F401
    with pytest.raises(AttributeError, match="sax.into is not provided"):
        sax.into  # noqa: B018
    with pytest.raises(AttributeError, match="RF models"):
        sax.models.capacitor  # noqa: B018
    with pytest.raises(ImportError, match="sax.models.rf is not provided"):
        import sax.models.rf  # noqa: F401


def _ports(s):
    return sorted({p for k in s for p in k})


def test_port_naming_is_sax_default_and_separate_from_socklight():
    compat.install()
    import sax
    import sax.models as sm

    from socklight.models import STDLIB

    assert sax.get_port_naming_strategy() == "inout"  # sax's default
    assert _ports(sm.mmi1x2()) == ["in0", "out0", "out1"]
    assert sax.PortNamer(1, 1).in0 == "in0"
    sax.set_port_naming_strategy("optical")
    assert _ports(sm.mmi1x2()) == ["o1", "o2", "o3"]
    assert sax.PortNamer(1, 1).in0 == "o1"
    sax.set_port_naming_strategy("inout")
    assert _ports(STDLIB["mmi1x2"]()) == ["o1", "o2", "o3"]  # socklight's registered stdlib: unchanged
    with pytest.raises(ValueError):
        sax.set_port_naming_strategy("bogus")


def test_models_keep_signatures_and_values():
    compat.install()
    import inspect

    import sax.models as sm

    from socklight.models import STDLIB

    for name, fn in STDLIB.items():
        assert inspect.signature(getattr(sm, name)) == inspect.signature(fn), name
    s, ref = sm.straight(wl=1.56, length=30.0), STDLIB["straight"](wl=1.56, length=30.0)
    assert s[("in0", "out0")] == ref[("o1", "o2")]


class FakeDataArray:
    """What interpolate_xarray reads of an xarray.DataArray: ``dims``, ``[dim]`` and ``__array__``."""

    def __init__(self, data, coords: dict):
        self.data, self.coords, self.dims = np.asarray(data), coords, tuple(coords)

    def __getitem__(self, dim):
        return np.asarray(self.coords[dim])

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)


def _table():
    x, y = np.array([0.0, 1.0, 3.0]), np.array([10.0, 20.0])
    mode = np.array(["te", "tm"], dtype=object)
    targets = np.array(["a", "b"], dtype=object)
    # f(x, y, mode, target), linear in x and y so bilinear interpolation is exact
    data = (
        x[:, None, None, None]
        + 0.1 * y[None, :, None, None]
        + np.array([0.0, 100.0])[None, None, :, None]
        + np.array([0.0, 1000.0])[None, None, None, :]
    )
    return FakeDataArray(data, {"x": x, "y": y, "mode": mode, "targets": targets})


def test_interpolate_xarray():
    compat.install()
    import sax

    t = _table()
    r = sax.interpolate_xarray(t, x=np.array([0.5, 2.0]), y=12.5, mode="tm")
    assert set(r) == {"a", "b"}
    np.testing.assert_allclose(r["a"], [0.5 + 1.25 + 100, 2.0 + 1.25 + 100], rtol=1e-15)
    np.testing.assert_allclose(r["b"], r["a"] + 1000, rtol=1e-15)
    r = sax.interpolate_xarray(t, x=-5.0, y=99.0, mode="te")  # clamped at the edges
    assert float(r["a"]) == pytest.approx(0.0 + 2.0)
    r = sax.interpolate_xarray(t, mode="te")  # numeric dims default to their mean
    assert float(r["a"]) == pytest.approx(4 / 3 + 1.5)
    r = sax.interpolate_xarray(t, x=1.0, y=10.0)  # an unselected string dim stays an axis
    np.testing.assert_allclose(r["a"], [2.0, 102.0])
    with pytest.raises(ValueError, match="target parameter"):
        sax.interpolate_xarray(t, targets="a")


def test_fixture_pdk_in_process():
    """The Pyodide path: no subprocess, the PDK imported after install()."""
    compat.install()
    import compat_pdk.models as pm

    from socklight.models import STDLIB

    models = pm.get_models()
    s = models["straight"](wl=1.56, cross_section="rib")
    ref = STDLIB["straight"](wl=1.56, wl0=1.55, length=10.0, loss_dB_cm=3.0, neff=2.60, ng=3.90)
    assert s[("o1", "o2")] == ref[("o1", "o2")]
    c = models["coupler"](wl=1.55)
    assert c[("o4", "o1")] == c[("o1", "o4")]
    assert abs(abs(c[("o1", "o4")]) ** 2 + abs(c[("o1", "o3")]) ** 2 - 1) < 1e-15
    assert _ports(models["mmi1x2"]()) == ["o1", "o2", "o3"]  # the PDK chose "optical"


def _run_child(mode: str, py: str, tmp_path, env: dict | None = None) -> dict:
    out = tmp_path / f"{mode}.json"
    p = subprocess.run([py, CHILD, mode, str(out)], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(out.read_text())


def _compat_env() -> dict:
    return {**os.environ, "PYTHONPATH": os.pathsep.join([SRC, HERE])}


@pytest.mark.native
def test_fixture_pdk_through_vacask(tmp_path):
    res = _run_child("compat", sys.executable, tmp_path, _compat_env())
    assert res["real_modules"] == []
    assert {"straight", "coupler", "mmi1x2", "grating_coupler", "crossing"} <= set(res["models"])
    v = res["vacask"]
    assert v["last_error"] is None
    for p, want in v["want"].items():
        assert abs(complex(*v["got"][p]) - complex(*want)) < 1e-9, (p, v)
    assert abs(complex(*v["want"]["o3"])) > 0.1  # the coupler couples


@pytest.mark.native
def test_fixture_pdk_matches_real_sax(tmp_path):
    py = os.environ.get("SOCKLIGHT_JAX_PYTHON", os.path.expanduser("~/code/gdsfactory/cspdk/.venv/bin/python"))
    if not os.path.exists(py):
        pytest.skip("no jax interpreter")
    real = _run_child("real", py, tmp_path, {**os.environ, "PYTHONPATH": HERE})
    # The shim side runs in the same interpreter, with its real jax and sax blocked.
    shim = _run_child("shim", py, tmp_path, _compat_env())
    assert shim["real_modules"] == []
    assert "coupler_table" in real["models"]  # the jax venv has xarray: the table is compared too
    assert set(real["S"]) == set(shim["S"])
    for case, sr in real["S"].items():
        ss = shim["S"][case]
        assert set(sr) == set(ss), case
        for e in sr:
            a, b = complex(*sr[e]), complex(*ss[e])
            assert abs(a - b) <= 1e-12 * max(1.0, abs(a)), (case, e, a, b)
