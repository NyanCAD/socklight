import json
import os

import numpy as np
import pytest

from socklight.models import STDLIB

REF = json.load(open(os.path.join(os.path.dirname(__file__), "data", "sax_reference.json")))
CASES = [(m, wl) for m, d in REF.items() for wl in d]


def test_reference_covers_stdlib():
    assert set(REF) == set(STDLIB)


@pytest.mark.parametrize("name,wl", CASES)
def test_matches_sax_intended(name, wl):
    fn = STDLIB[name]
    s = fn(wl=float(wl)) if "wl" in fn.__code__.co_varnames[: fn.__code__.co_argcount + fn.__code__.co_kwonlyargcount] else fn()
    ref = REF[name][wl]
    assert {",".join(k) for k in s} == set(ref)
    for k, v in s.items():
        re, im = ref[",".join(k)]
        assert abs(complex(np.asarray(v).reshape(-1)[0]) - complex(re, im)) < 1e-12, (name, wl, k)


def test_mmi_scalar_has_dispersion():
    from socklight.models import mmi2x2

    assert abs(mmi2x2(wl=1.50)[("o1", "o3")]) < abs(mmi2x2(wl=1.55)[("o1", "o3")])
