"""Heater-tuned MZI into a photodetector and a TIA built from VACASK's bundled opamp.

Sweeps the heater voltage (one operating point per value) and prints the TIA
output. Run: python examples/mzi_tia.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import socklight


def deck(mod_dir: str, vh: float) -> str:
    return "\n".join(
        [
            "mzi tia",
            "ground 0",
            f'load "{socklight.osdi_path()}"',
            *(f'load "{socklight.va_path(n)}"' for n in ("optical_source", "optical_term", "photodetector", "heater")),
            *(f'load "{mod_dir}/{n}.osdi"' for n in ("resistor", "opamp")),
            "model r resistor",
            "model vsrc vsource",
            "model src optical_source",
            "model term optical_term",
            "model pd photodetector",
            "model heater heater",
            "model amp opamp",
            "model cpl 'socklight.models:coupler_ideal'",
            "model wg 'socklight.models:straight'",
            "s1 (a_i a_q) src p=1",
            "c1 (a_i a_q x_i x_q top_i top_q bot_i bot_q) cpl coupling=0.5",
            "t1 (x_i x_q) term",
            "h1 (top_i top_q m1_i m1_q le 0) heater",
            "w1 (bot_i bot_q m2_i m2_q) wg length=320 neff=2.38",
            "c2 (m1_i m1_q m2_i m2_q y1_i y1_q y2_i y2_q) cpl coupling=0.5",
            "t2 (y2_i y2_q) term",
            # photocurrent leaves p into the summing node `inv` (virtual ground of the TIA)
            "d1 (y1_i y1_q inv 0) pd responsivity=1",
            "a1 (0 inv out) amp",
            "rf (inv out) r r=5",
            f"vh (le 0) vsrc dc={vh!r}",
            "",
        ]
    )


def solve_op(netlist: str) -> dict[str, float]:
    from vacask import Analysis, Circuit, OpenvafCompiler, Parser, ParserTables, PTAnalysis, PTSave, Status
    from vacask import simulator as sim
    from vacask.rawfile import rawread

    sim.setup()
    s = Status()
    tab = ParserTables("models")
    if not Parser(tab).parseNetlistString(netlist) or not tab.verify(s):
        raise RuntimeError(s.message() or "netlist does not parse")
    compiler = OpenvafCompiler()
    cir = Circuit(tab, compiler, s)
    if not cir.isValid() or not cir.elaborate([], "__topdef__", "__topinst__", status=s):
        raise RuntimeError(s.message())
    desc = PTAnalysis("op1", "op")
    analysis = Analysis.create(desc, cir, s)
    if not analysis:
        raise RuntimeError(s.message())
    analysis.add(PTSave("default"))
    ok, _ = analysis.run(s)
    if not ok:
        raise RuntimeError(s.message())
    res = rawread("op1.raw").get()
    return {name: float(res[name][0]) for name in res.names}


def main() -> None:
    import vacask

    mod_dir = os.path.join(os.path.dirname(vacask.__file__), "mod")
    cwd = os.getcwd()
    print("Vheater [V]  Vtia [V]")
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # the raw file lands in the working directory
        try:
            for vh in np.linspace(0.0, 4.0, 17):
                print(f"{vh:11.2f} {solve_op(deck(mod_dir, float(vh)))['out']:10.4f}")
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    main()
