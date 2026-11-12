# socklight

sax's optical models on numpy, exposed to the VACASK circuit simulator as OSDI devices. Every
model in socklight's Python registry becomes one device of `socklight.osdi`, which VACASK's
Python extension loads like any other OSDI file. The design is in `docs/design.md`; the optical
port convention is in `docs/iq-convention.md`.

## Install

socklight needs VACASK's Python extension (`vacask`), and OpenVAF-reloaded (`openvaf-r`) to
compile the Verilog-A helpers. Neither is on PyPI.

```sh
pip install /path/to/vacask-*.whl                 # or build it from a VACASK checkout
PATH=/path/to/openvaf-r-dir:$PATH pip install .
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
