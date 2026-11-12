"""Wavelength sweep: the circuit is re-elaborated per point from Python."""

from __future__ import annotations

import numpy as np
import pytest
from test_osdi import wave

import socklight
from socklight.models import STDLIB

WLS = [1.50, 1.52, 1.54, 1.55, 1.56, 1.58, 1.60]


def deck(mod_dir: str, wl: float) -> str:
    return "\n".join(
        [
            "sweep test",
            "ground 0",
            f'load "{socklight.osdi_path()}"',
            f'load "{socklight.va_path("optical_source")}"',
            f'load "{socklight.va_path("optical_term")}"',
            "model src optical_source",
            "model term optical_term",
            "model mmi 'socklight.models:mmi2x2' fwhm=0.02",
            "s1 (o1_i o1_q) src p=1",
            f"m1 (o1_i o1_q o2_i o2_q o3_i o3_q o4_i o4_q) mmi wl={wl!r}",
            "t2 (o2_i o2_q) term",
            "t3 (o3_i o3_q) term",
            "t4 (o4_i o4_q) term",
            "",
        ]
    )


def test_wl_sweep_passband(run_op, vacask_mod_dir):
    got, want = [], []
    for wl in WLS:
        v = run_op(deck(vacask_mod_dir, wl))
        got.append(abs(wave(v, "o3")) ** 2 + abs(wave(v, "o4")) ** 2)
        s = STDLIB["mmi2x2"](wl=wl, fwhm=0.02)
        want.append(abs(complex(s[("o3", "o1")])) ** 2 + abs(complex(s[("o4", "o1")])) ** 2)
    assert np.allclose(got, want, atol=1e-9, rtol=0)
    assert int(np.argmax(got)) == WLS.index(1.55)
    assert got[WLS.index(1.55)] > got[WLS.index(1.54)] + 1e-3  # a real peak, not a flat line
