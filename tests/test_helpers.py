"""The Verilog-A helpers (optical_source, optical_term, photodetector) in VACASK."""

from __future__ import annotations

import cmath
import math

import numpy as np
from conftest import elaborate
from test_osdi import wave

import socklight
from socklight import registry

MMI = "socklight.models:mmi2x2"


def deck(mod_dir: str, *lines: str) -> str:
    head = [
        "helpers test",
        "ground 0",
        f'load "{socklight.osdi_path()}"',
        f'load "{socklight.va_path("optical_source")}"',
        f'load "{socklight.va_path("optical_term")}"',
        f'load "{socklight.va_path("photodetector")}"',
        f'load "{mod_dir}/resistor.osdi"',
        "model r resistor",
        "model src optical_source",
        "model term optical_term",
        "model pd photodetector",
        f"model mmi '{MMI}'",
    ]
    return "\n".join([*head, *lines]) + "\n"


def test_source_into_term_power(run_op, vacask_mod_dir):
    v = run_op(deck(vacask_mod_dir, "s1 (a_i a_q) src p=2.0 phase=0.3", "t1 (a_i a_q) term"))
    assert abs(wave(v, "a") - math.sqrt(2) * cmath.exp(0.3j)) < 1e-12


def test_source_through_model(run_op, vacask_mod_dir):
    ports = registry._models[MMI][0].ports
    nodes = " ".join(f"{p}_i {p}_q" for p in ports)
    lines = [f"x1 ({nodes}) mmi wl=1.55", f"s1 ({ports[0]}_i {ports[0]}_q) src"]
    lines += [f"t{k} ({p}_i {p}_q) term" for k, p in enumerate(ports[1:], 2)]
    v = run_op(deck(vacask_mod_dir, *lines))
    s = registry._models[MMI][1](wl=1.55)
    for p in ports[1:]:
        want = complex(np.asarray(s.get((p, ports[0]), 0.0)).reshape(-1)[0])
        assert abs(wave(v, p) - want) < 1e-9, (p, wave(v, p), want)


def test_detector_current(run_op, vacask_mod_dir):
    v = run_op(
        deck(
            vacask_mod_dir,
            "s1 (a_i a_q) src p=0.5",
            "d1 (a_i a_q p 0) pd responsivity=0.8",
            "r1 (p 0) r r=1",
        )
    )
    assert abs(v["p"] - 0.4) < 1e-12


def test_unknown_param_rejected(run_op, vacask_mod_dir):
    import pytest
    from conftest import VacaskError

    with pytest.raises(VacaskError):
        elaborate(deck(vacask_mod_dir, "s1 (a_i a_q) src power=1"))
