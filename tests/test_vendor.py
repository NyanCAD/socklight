import os
import subprocess
import sys

import numpy as np
import pytest

from socklight.models import STDLIB

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def test_stdlib_keys():
    assert set(STDLIB) == {"straight","bend","attenuator","phase_shifter","coupler","coupler_ideal",
        "grating_coupler","crossing_ideal","mmi1x2","mmi1x2_ideal","mmi2x2","mmi2x2_ideal",
        "splitter_ideal","mirror","reflector","isolator","circulator","terminator"}


def test_every_model_callable_with_defaults():
    for name, fn in STDLIB.items():
        s = fn()
        assert isinstance(s, dict) and s, name
        assert all(np.isfinite(v).all() for v in s.values()), name


def _ports(s):
    return {p for k in s for p in k}


def test_port_names_are_optical():
    assert _ports(STDLIB["mmi2x2"]()) == {"o1", "o2", "o3", "o4"}
    assert _ports(STDLIB["coupler"]()) == {"o1", "o2", "o3", "o4"}
    assert _ports(STDLIB["straight"]()) == {"o1", "o2"}


def _out_power(s, src):
    return sum(abs(v) ** 2 for (_, i), v in s.items() if i == src)


def test_straight_defaults_sane():
    s21 = abs(STDLIB["straight"]()["o1", "o2"])
    assert 0 < s21 <= 1


@pytest.mark.parametrize("name", ["coupler_ideal", "mmi2x2_ideal"])
def test_ideal_four_port_conserves_power(name):
    s = STDLIB[name]()
    for src in ("o1", "o2"):
        assert abs(_out_power(s, src) - 1) < 1e-12, (name, src)


@pytest.mark.native
def test_no_jax_in_sys_modules_after_import():
    code = "import socklight, socklight.models, sys; assert not any(m == 'jax' or m.startswith('jax.') for m in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": SRC})


@pytest.mark.native
def test_no_jax_even_when_installed():
    py = os.environ.get("SOCKLIGHT_JAX_PYTHON", os.path.expanduser("~/code/gdsfactory/cspdk/.venv/bin/python"))
    if not os.path.exists(py):
        pytest.skip("no jax interpreter")
    code = f"import sys; sys.path.insert(0, {SRC!r}); import socklight.models; assert 'jax' not in sys.modules"
    subprocess.run([py, "-I", "-c", code], check=True)
