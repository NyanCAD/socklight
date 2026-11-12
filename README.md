# socklight

sax's optical models on numpy, exposed to the VACASK circuit simulator as OSDI devices. Every
model in socklight's Python registry becomes one device of `socklight.osdi`, which VACASK's
Python extension loads like any other OSDI file. The design is in `docs/design.md`; the optical
port convention is in `docs/iq-convention.md`.

## Install

```sh
pip install socklight
```

Wheels exist for Linux (x86_64, aarch64; CPython 3.10–3.14) and for Pyodide 314
(Python 3.14, the Pyodide marimo ships), where `micropip.install("socklight")` works the same.
They pull in VACASK's Python extension (`vacask`).

Building from source needs a C compiler and CMake. The Verilog-A helpers are compiled with
the `openvaf-r` that `vacask-bin` (a build requirement) ships; pass `-DOPENVAF_COMPILER=...`
through `CMAKE_ARGS` to use another one. For a development install:

```sh
pip install vacask-bin scikit-build-core pytest
pip install -e . --no-build-isolation
```

`socklight.osdi` is built against the Python headers only and does not link libpython. It
works only inside a Python process that has `vacask` imported. The standalone `vacask` binary
cannot load it.

## A minimal deck

```python
import socklight  # registers the standard library (socklight.models)

netlist = f"""mmi demo
ground 0
load "{socklight.osdi_path()}"
load "{socklight.va_path("optical_source")}"
load "{socklight.va_path("optical_term")}"
model src optical_source
model term optical_term
model mmi 'socklight.models:mmi2x2' loss_db=0.3
s1 (o1_i o1_q) src p=1
x1 (o1_i o1_q o2_i o2_q o3_i o3_q o4_i o4_q) mmi wl=1.55
t2 (o2_i o2_q) term
t3 (o3_i o3_q) term
t4 (o4_i o4_q) term
"""
```

Run it with the `vacask` Python API (parse, elaborate, run an `op` analysis); `tests/conftest.py`
shows how.

- The device name is the model's resource path, `package.module:callable`. VACASK identifiers
  cannot contain `.` or `:`, so put it in single quotes on the `model` line.
- Each optical port `p` has two terminals, `p_i p_q`. Ports are listed in `sorted()` order.
  For `mmi2x2` that is `o1 o2 o3 o4`.
- Every model parameter can be set on the `model` line or on an instance; the instance wins.
  A parameter you don't set takes the Python signature default. A parameter whose default is
  `None` is not passed to the model unless you set it.
- Every numeric parameter is a real in the deck. One whose Python default is an `int` or `bool`
  (e.g. `loss_dB_cm=3`) still accepts any real: the model gets an `int` when the value is
  integral (`loss_db_cm=2` gives `2`) and a `float` otherwise (`loss_db_cm=2.5` gives `2.5`).
- `socklight.va_path(name)` gives the Verilog-A helpers: `optical_source`, `optical_term`,
  `photodetector` and `heater` (see `docs/iq-convention.md`).

### Parameter names are lowercase

VACASK lowercases device and parameter names when it loads an OSDI file. Netlists must
write them in lowercase, including in sweeps. So sax's `loss_dB` is `loss_db` in a deck,
as is `parameter="loss_db"` in a sweep. socklight still calls the model with the original
`loss_dB`.

## Your own models

```python
import socklight

def ring(wl=1.55, kappa=0.1, loss_dB=0.0):
    ...  # return a sax SDict at scalar wl

socklight.register("mypdk.models:ring", ring)
socklight.register_models("mypdk.models", {"ring": ring, "taper": taper})  # register() for each
```

Register every model **before** the first netlist loads `socklight.osdi`. Its constructor
snapshots the registry once per process.

- After that, registering a new name, or a model whose ports or parameters (names, kinds,
  defaults) differ, raises `RuntimeError` naming what changed.
- Registering the same signature again replaces the function. The next circuit setup uses
  the new function, so a notebook can edit a model body without restarting.
- `register()` validates up front. It raises on a malformed name, a parameter without a
  default, two names or parameters that collide case-insensitively, and a probe call that
  fails.

## From a gdsfactory netlist

`socklight.netlist.netlist_to_deck` turns a flat netlist, as `Component.get_netlist()` returns it,
into a deck. It imports neither gdsfactory nor sax.

```python
from socklight.netlist import netlist_to_deck

socklight.register_models("cspdk.si220", PDK.models)  # before any deck loads socklight.osdi
deck = netlist_to_deck(c.get_netlist(), "cspdk.si220", params={"wl": 1.31})
i, q = deck.external["o1"]  # nodes of external port o1: append a source here
netlist = deck.text + f'load "{socklight.va_path("optical_source")}"\nmodel src optical_source\n' \
    + f"s1 ({i} {q}) src p=1\n"  # ...and terms or photodetectors on the other external ports
```

- `models` is a prefix (component `coupler` uses `"cspdk.si220:coupler"`) or a mapping from
  component to registered name. A component without a registered model raises `KeyError`.
- An instance's parameters are its `settings` with `info` merged over them, as `sax.circuit`
  does. Keys that are not parameters of the model are left out and listed in `deck.dropped`
  (layout-only settings such as `width` or `npoints` are normal there). `None` means "use the
  model's default".
- `params` go to every instance whose model has that parameter, unless the instance sets it
  itself: the instance wins.
- `variables=("wl",)` is `params` by reference: each instance whose model has `wl` (and does not
  set it) gets `wl=wl`, a VACASK circuit variable, so a single sweep moves every instance. A name
  in both `params` and `variables` raises `ValueError`. `deck.variables` lists them; declare each
  one before elaboration or VACASK will not resolve it:

  ```python
  deck = netlist_to_deck(c.get_netlist(), "cspdk.si220", variables=("wl",))
  # ...append sources and detectors, then the control block:
  control = """
  control
    var wl=1.55
    sweep s variable="wl" values=[1.50, 1.52, 1.55, 1.58, 1.60]
    analysis op1 op
  endc
  """
  ```

  Through the Python API there is no control block: call `circuit.setVariable("wl", Value(1.55))`
  before `circuit.elaborate`, then sweep with `PV("variable", "wl")` (see `tests/conftest.py`).
- Each net is one node pair `n<k>_i n<k>_q`. An external port `p` is `p_i p_q` (sanitized if `p`
  is not a VACASK identifier). An instance port that is on no net and not external is
  terminated with `optical_term`. The deck always loads it as model `optical_term`, so appended
  lines can use it.
- The deck has no sources, detectors or analysis. Append them.

## Errors

VACASK reports only error codes for OSDI failures, so socklight records the details on the
Python side:

- `socklight.load_error()`: why `socklight.osdi` exported no devices (a registry over capacity
  or a failed snapshot), else `None`.
- `socklight.last_error()`: the last setup failure, as a dict with `model`, `params` and
  `traceback`. Possible causes:
  - the model raised;
  - it returned a non-finite entry;
  - its ports changed;
  - `I + S` is singular;
  - the instance connects too few terminals ("connected k of 2n terminals").

  The simulation fails with "Fatal error during setup".

## Sweeping wavelength

`wl` is an ordinary parameter, so VACASK's own sweep works, whether `wl` is given on the
instance, on the model line, or not at all:

```
control
  sweep wl_sweep instance="x1" parameter="wl" values=[1.50, 1.52, 1.54, 1.55, 1.56]
  analysis op1 op
endc
```

Each point re-runs the device's setup, where the model is evaluated. `eval` and `load` never
call Python. When the sweep finishes, VACASK restores the original value.
`tests/test_native_sweep.py` builds such sweeps from Python. You can also re-elaborate one
circuit per point, as `tests/test_sweep.py` does.

## Examples and cross-checks

- `examples/mzi_tia.py`: a heater-tuned MZI feeding a photodetector and a TIA built from VACASK's
  bundled `opamp`. Sweeping the heater voltage from 0 to 4 V swings the TIA output between
  about 0 V and -4.9 V.
- `tools/circulax_heater.py`, run in a venv with jax, sax, circulax and cspdk (not socklight):
  - It builds the same heater MZI in circulax with cspdk's `ThermalPhaseShifter` and writes
    `tests/data/circulax_heater.json`. `tests/test_circulax.py` compares that file with the
    VACASK run of the socklight heater.
  - Over an 11-point sweep from 0 to 2 V, the two detector powers differ by at most 4.3e-9.
    circulax against a closed-form numpy MZI also differs by 4.3e-9.
  - Agreement is expected for the forward configuration only. circulax's heater is a VCVS and
    is non-reciprocal at `o2`, so reverse drive is not compared.
- Why the two ranges differ: the heater has R = 0.375 ohm/um × 320 um = 120 ohm, and its phase
  is `dphi = pi × 12.5 × V^2 / 120`.
  - 0 to 2 V gives up to 0.42 pi, part of one fringe. That is enough for a point-by-point
    numerical comparison and keeps the circulax run small.
  - 0 to 4 V gives up to 1.67 pi, so the demo passes through full extinction and shows the
    TIA's whole output swing.
