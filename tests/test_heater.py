"""The thermal phase shifter (heater.va) in VACASK."""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
from test_osdi import wave

import socklight
from socklight.models import STDLIB

R = 0.375 * 320.0
LOSS_DB_CM = 3.0
LENGTH = 320.0
NEFF0 = 2.38


def t_of(vh: float, loss_db_cm: float = LOSS_DB_CM) -> complex:
    """Closed-form T of the default heater at wl = wl0 = 1.55."""
    dphi = math.pi * 12.5 * vh**2 / R
    phi = 2 * math.pi * NEFF0 * LENGTH / 1.55 + dphi
    return 10 ** (-loss_db_cm * LENGTH / 1e4 / 20) * cmath.exp(1j * phi)


def deck(mod_dir: str, *lines: str) -> str:
    head = [
        "heater test",
        "ground 0",
        f'load "{socklight.osdi_path()}"',
        *(f'load "{socklight.va_path(n)}"' for n in ("optical_source", "optical_term", "photodetector", "heater")),
        f'load "{mod_dir}/resistor.osdi"',
        "model r resistor",
        "model vsrc vsource",
        "model src optical_source",
        "model term optical_term",
        "model pd photodetector",
        "model heater heater",
        "model cpl 'socklight.models:coupler_ideal'",
        "model wg 'socklight.models:straight'",
    ]
    return "\n".join([*head, *lines]) + "\n"


def chain(vh: float, src_at: str, **params) -> list[str]:
    other = "o2" if src_at == "o1" else "o1"
    extra = "".join(f" {k}={v}" for k, v in params.items())
    return [
        f"s1 ({src_at}_i {src_at}_q) src p=1",
        f"h1 (o1_i o1_q o2_i o2_q le 0) heater{extra}",
        f"t1 ({other}_i {other}_q) term",
        f"vh (le 0) vsrc dc={vh}",
    ]


@pytest.mark.parametrize("vh", [0.0, 1.0, 2.0])
def test_heater_transmission(run_op, vacask_mod_dir, vh):
    v = run_op(deck(vacask_mod_dir, *chain(vh, "o1")))
    assert abs(wave(v, "o2") - t_of(vh)) < 1e-9


@pytest.mark.parametrize("vh", [0.0, 1.0, 2.0])
def test_heater_reverse(run_op, vacask_mod_dir, vh):
    v = run_op(deck(vacask_mod_dir, *chain(vh, "o2")))
    assert abs(wave(v, "o1") - t_of(vh)) < 1e-9


def test_heater_no_reflection(run_op, vacask_mod_dir):
    v = run_op(deck(vacask_mod_dir, *chain(1.0, "o1")))
    assert abs(wave(v, "o1") - 1.0) < 1e-9  # V = a + b with a = 1, b = 0


def test_heater_lossless_ok(run_op, vacask_mod_dir):
    v = run_op(deck(vacask_mod_dir, *chain(0.0, "o1", loss_dbcm=0)))
    assert abs(abs(wave(v, "o2")) - 1.0) < 1e-9
    assert abs(wave(v, "o2") - t_of(0.0, 0.0)) < 1e-9


def test_heater_electrical(run_op, vacask_mod_dir):
    v = run_op(deck(vacask_mod_dir, *chain(2.0, "o1")))
    assert abs(v["vh:flow(br)"]) == pytest.approx(2.0 / R, rel=1e-9)


@pytest.mark.parametrize("vh", [float(x) for x in np.linspace(0.0, 2.0, 11)])
def test_mzi_sweep(run_op, vacask_mod_dir, vh):
    sc = STDLIB["coupler_ideal"](coupling=0.5)
    sw = STDLIB["straight"](length=LENGTH, neff=NEFF0)
    c = lambda a, b: complex(sc[(a, b)])  # noqa: E731
    mid = {"m1": c("o3", "o1") * t_of(vh), "m2": c("o4", "o1") * complex(sw[("o1", "o2")])}
    want = {
        "p1": abs(c("o3", "o1") * mid["m1"] + c("o3", "o2") * mid["m2"]) ** 2,
        "p2": abs(c("o4", "o1") * mid["m1"] + c("o4", "o2") * mid["m2"]) ** 2,
    }
    v = run_op(
        deck(
            vacask_mod_dir,
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
    for k, w in want.items():
        assert abs(v[k] - w) < 1e-9, (k, v[k], w)
