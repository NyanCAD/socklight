"""Write tests/data/sax_reference.json from real sax (jax) at a pinned commit.

Run with a jax+sax interpreter, with sax at c22ed91 first on PYTHONPATH:

    git -C ~/code/gdsfactory/sax archive c22ed91 src/sax | tar -x -C "$TMP"
    PYTHONPATH="$TMP/src" python tools/reference_values.py

Models with a ``wl0`` parameter are evaluated at ``wl=[wl, wl0]`` and index 0 is
taken (sax's normalisation is then over a 2-point array that includes wl0).
Models without ``wl`` are evaluated once and stored under every wl key.
"""
import inspect
import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import sax

NAMES = ["straight", "bend", "attenuator", "phase_shifter", "coupler", "coupler_ideal",
         "grating_coupler", "crossing_ideal", "mmi1x2", "mmi1x2_ideal", "mmi2x2",
         "mmi2x2_ideal", "splitter_ideal", "mirror", "reflector", "isolator",
         "circulator", "terminator"]
WLS = [1.50, 1.53, 1.55, 1.57, 1.60]

sax.models.set_port_naming_strategy("optical") if hasattr(sax.models, "set_port_naming_strategy") \
    else sax.set_port_naming_strategy("optical")


def first(v):
    return np.asarray(v).reshape(-1)[0]


out = {}
for name in NAMES:
    fn = getattr(sax.models, name)
    params = inspect.signature(fn).parameters
    out[name] = {}
    for wl in WLS:
        if "wl0" in params:
            wl0 = params["wl0"].default
            s = fn(wl=np.array([wl, wl0]))
        elif "wl" in params:
            s = fn(wl=wl)
        else:
            s = fn()
        out[name][str(wl)] = {
            ",".join(k): [float(np.real(first(v))), float(np.imag(first(v)))]
            for k, v in s.items()
        }

path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "sax_reference.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print("wrote", os.path.normpath(path), sum(len(v) for v in out.values()), "model/wl entries")
