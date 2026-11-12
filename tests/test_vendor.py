def test_stdlib_keys():
    from socklight.models import STDLIB
    assert set(STDLIB) == {"straight","bend","attenuator","phase_shifter","coupler","coupler_ideal",
        "grating_coupler","crossing_ideal","mmi1x2","mmi1x2_ideal","mmi2x2","mmi2x2_ideal",
        "splitter_ideal","mirror","reflector","isolator","circulator","terminator"}

def test_every_model_callable_with_defaults():
    from socklight.models import STDLIB
    for name, fn in STDLIB.items():
        s = fn()
        assert isinstance(s, dict) and s, name

def test_no_jax_in_sys_modules_after_import():
    import subprocess, sys
    code = "import socklight, socklight.models, sys; assert not any(m == 'jax' or m.startswith('jax.') for m in sys.modules)"
    import os
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": src})

def test_no_jax_even_when_installed():
    import os, subprocess, pytest
    py = os.environ.get("SOCKLIGHT_JAX_PYTHON", os.path.expanduser("~/code/gdsfactory/cspdk/.venv/bin/python"))
    if not os.path.exists(py): pytest.skip("no jax interpreter")
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    code = f"import sys; sys.path.insert(0, {src!r}); import socklight.models; assert 'jax' not in sys.modules"
    subprocess.run([py, "-I", "-c", code], check=True)
