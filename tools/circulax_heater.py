"""Secondary heater cross-check: the MZI of tests/test_heater.py in circulax.

Run in an environment that has circulax, jax, sax and cspdk (not socklight):

    ~/code/gdsfactory/cspdk/.venv/bin/python tools/circulax_heater.py

Builds coupler -> (cspdk ThermalPhaseShifter | straight) -> coupler with a matched
source and 1 ohm terminations, sweeps the heater voltage, prints the max abs
difference of the two detector powers against a closed-form numpy evaluation, and
writes tests/data/circulax_heater.json. tests/test_circulax.py compares that file with
the VACASK run of the same circuit. Agreement is expected for the forward
configuration only: circulax's heater is a VCVS (non-reciprocal at o2).
"""

from __future__ import annotations

import json
import pathlib

import dataclasses

import jax
import numpy as np
import sax
from circulax import compile_circuit
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.photonic import OpticalSource
from circulax.s_transforms import sax_component
from cspdk.si220.active_models import ThermalPhaseShifter

jax.config.update("jax_enable_x64", True)
sax.set_port_naming_strategy("optical")  # o1..o4, as socklight; before any sax_component is built

VH = [float(x) for x in np.linspace(0.0, 2.0, 11)]
OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "circulax_heater.json"
LENGTH, NEFF = 320.0, 2.38  # the heater's, and the reference arm's, straight parameters


def circuit():
    models = {
        "coupler": sax_component(sax.models.coupler_ideal),
        "straight": sax_component(sax.models.straight),
        "heater": ThermalPhaseShifter,
        "source": OpticalSource,
        "res": Resistor,
        "vsrc": VoltageSource,
    }
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "s1": {"component": "source", "settings": {"power": 1.0}},
            "c1": {"component": "coupler", "settings": {"coupling": 0.5}},
            "c2": {"component": "coupler", "settings": {"coupling": 0.5}},
            "h1": {"component": "heater"},
            "w1": {"component": "straight", "settings": {"length": LENGTH, "neff": NEFF}},
            "vh": {"component": "vsrc", "settings": {"V": 0.0}},
            "rx": {"component": "res", "settings": {"R": 1.0}},
            "r1": {"component": "res", "settings": {"R": 1.0}},
            "r2": {"component": "res", "settings": {"R": 1.0}},
        },
        "connections": {
            "s1,p1": "c1,o1",
            "s1,p2": "GND,p1",
            "c1,o3": "h1,o1",
            "c1,o4": "w1,o1",
            "h1,o2": "c2,o1",
            "w1,o2": "c2,o2",
            "c1,o2": "rx,p1",
            "rx,p2": "GND,p1",
            "c2,o3": "r1,p1",
            "r1,p2": "GND,p1",
            "c2,o4": "r2,p1",
            "r2,p2": "GND,p1",
            "h1,l_e2": "vh,p1",
            "vh,p2": "GND,p1",
            "h1,r_e2": "GND,p1",
        },
        "ports": {},
    }
    return compile_circuit(net, models)


def sweep():
    """Detector powers |V(r1,p1)|^2, |V(r2,p1)|^2 per heater voltage."""
    c = circuit()
    out = []
    for vh in VH:
        y = np.asarray(c.dc(params={"vh.V": vh}))
        if c.solver.is_complex:  # the solution is stored as [real parts, imaginary parts]
            y = y[: c.sys_size] + 1j * y[c.sys_size :]
        out.append([abs(y[c.port_map[f"{r},p1"]]) ** 2 for r in ("r1", "r2")])
    return out


def closed_form(vh: float):
    """The same MZI in numpy: sax coupler/straight models and the heater's closed-form T."""
    r = 0.375 * LENGTH
    h = ThermalPhaseShifter
    d = {f.name: f.default for f in dataclasses.fields(h)}
    dphi = np.pi * d["eta_pi_per_W"] * vh**2 / r
    phi = 2 * np.pi * d["neff0"] * LENGTH / d["wl"] + dphi
    t = 10 ** (-d["loss_dBcm"] * LENGTH / 1e4 / 20) * np.exp(1j * phi)
    sc = sax.models.coupler_ideal(coupling=0.5)
    sw = sax.models.straight(length=LENGTH, neff=NEFF)
    c = lambda a, b: complex(sc[(a, b)])  # noqa: E731
    m1 = c("o3", "o1") * t
    m2 = c("o4", "o1") * complex(sw[("o1", "o2")])
    return [abs(c(o, "o1") * m1 + c(o, "o2") * m2) ** 2 for o in ("o3", "o4")]


if __name__ == "__main__":
    got = sweep()
    want = [closed_form(v) for v in VH]
    print("max |circulax - closed form| =", float(np.max(np.abs(np.array(got) - np.array(want)))))
    OUT.write_text(json.dumps({"vh": VH, "p": got}, indent=1) + "\n")
    print("wrote", OUT)
