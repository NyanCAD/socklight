# socklight — sax-shaped photonic models as native VACASK devices

Date: 2026-11-12. Status: design, approved for planning once reviewed.

## Intent

pyosdi (this repo) proved that a hand-written OSDI device can evaluate a sax-shaped
S-parameter model in Python at setup time and stamp it into VACASK as a real
power-wave admittance, so VACASK's own solver does the photonic network solve and a
wavelength spectrum is a DC sweep of `wl`.

It got there with hacks that this design removes:

- it **embeds** its own CPython (`Py_InitializeFromConfig`, `PYOSDI_SYS_PATH`,
  promoting libpython to `RTLD_GLOBAL`) — fragile, and impossible in Pyodide;
- every device is a **generic** `pyosdi2`/`pyosdi4` with `module="pkg.mod:fn"` and a
  `params="a=1,b=2"` string, because OSDI descriptors were static. Typos in `params`
  are silent; port counts are hand-picked;
- the backing model library is **sax (jax) or socklu**, neither of which runs in a
  browser: jax does not exist in Pyodide, and socklu ships KLU.
- sax's MMI normalises its dispersion by `spectral_response.max()`, which flattens
  every scalar-wavelength call, so the bridge evaluates `[wl, wl_ref]` and indexes.

The new shape: **socklight**, a pure-Python, numpy-only curated copy of the sax
standard library (vendored directly from sax, jax swapped for numpy, no circuit
solver, and without the normalisation bug),
plus one small C `.osdi` that, when VACASK loads it **inside an already running
Python process** (VACASK's Python extension, native or Pyodide), reads a registry of
Python models and declares **one OSDI device per model**, with real typed parameters
and real port nodes.

The motivating use is photonic and optoelectronic circuit simulation in a browser
notebook (nyanodide: VACASK built for Pyodide, `~/code/nyanodide/VACASK`), with the
same code running natively.

## Goals (v1)

1. `socklight` Python package: numpy-only curated stdlib of passive optical models,
   no jax, no sax, no KLU/SuiteSparse.
2. A model registry: `import pdk; socklight.register(...)` before VACASK loads the
   `.osdi`; socklight's own stdlib registers itself on import.
3. `socklight.osdi`: generates one descriptor per registered model at `dlopen` time.
   Device name = the model's resource path (`socklight.models:mmi2x2`), parameters =
   the model's signature, nodes = the model's ports as I/Q pairs.
4. Attach to the running interpreter; never initialise Python.
5. A written-down **I/Q convention** for optical ports, shared by every device.
6. Helper devices that make decks readable: optical source, matched termination,
   photodetector.
7. One voltage-dependent device, a **thermal phase shifter (heater)**, written in
   Verilog-A, proving the convention with an electrical input.
8. Works natively (Linux) and in Pyodide (wasm side module), with tests in both.

## Non-goals (v1)

- PDK models (cspdk etc.). They come later, either curated into socklight-native
  form or through an import hook mapping `jax.numpy`/`sax` to numpy/socklight. The
  registry must not preclude either; nothing else about them is in scope.
- Composite/hierarchical models evaluated in Python (`sax.circuit`, `host.solve`).
  Composition is done in the VACASK netlist (subcircuits).
- High-speed modulators, Python evaluated inside the Newton loop, voltage-dependent
  models in Python (setup-time S(V) tables are the planned later route).
- Automatic derivatives. Passive devices need none (the Jacobian is Y).
- Standalone `vacask` binary support (no host interpreter). See *Embedding*.
- macOS and Windows builds (see *Linking*; keep them possible, do not build them).
- Mechanical translation of circulax/Python active models into Verilog-A.
- The electrical S-parameter models in sax's `rf` module. Every socklight port is an
  optical I/Q envelope pair, so they would be wrong in every analysis. Lumped parts are
  VACASK's native devices already. Lines would need rational fitting or a Verilog-A
  model to be stamped as `G + jωC`, which is a separate design.

## Anti-goals (would make this a failure even if it works)

- Python running in `eval`/`load_*` (the Newton loop, evaluated in parallel).
- Any unqualified model name in a netlist. Models are named by resource path.
- Unknown parameters being silently ignored. A typo must be a VACASK parse error.
- A Python failure silently replaced by a fallback value. (pyosdi's `t` fallback goes.)
- `import socklight` ever importing jax (socklu's `JAXFREEDOM_REPORT.md` is the
  precedent: test it in an interpreter where jax *is* installed).

## Decided

### Distribution

One wheel, `socklight`:

- `socklight/` — pure Python: `models/`, `register`, the bridge module the C side
  calls, `osdi_path()` returning the absolute path of the bundled `.osdi`.
- `socklight/socklight.osdi` — package data. Native: a shared library. Pyodide: a
  wasm side module built with the same Emscripten SDK as the VACASK Pyodide build.
- the Verilog-A helpers and heater, compiled by OpenVAF-reloaded to `.osdi` (native)
  and wasm side modules (Pyodide), also package data. VACASK's own Pyodide build
  already does this for its device library (VACASK commit `b10bd9f6`); reuse that
  toolchain.

No Cython: what remains of the C/CPython glue once embedding is gone is ~40 lines
(take the GIL, import the bridge, call one function, read floats). The bridge logic
stays in Python (today `python/pyosdi_bridge.py`).

### Repository and provenance

- **New repo**, `~/code/gdsfactory/socklight` (location is the builder's call). pyosdi
  stays as the spike it was.
- From pyosdi, take **only the bridge**: `src/osdi.h`, `src/pyosdi.c`,
  `src/device_body.inc`, `src/pybridge.c` and `python/pyosdi_bridge.py`, reshaped
  per this document. pyosdi's decks may be ported as regression tests.
- **Nothing is copied from socklu.** Its reports (`JAXFREEDOM_REPORT.md`,
  `VENDORED.md`, …) can be read as lessons, but the models come **directly from
  sax**: `src/sax/models/*.py` plus the minimum they need (`constants.py`,
  `ports.py`, `s.py::reciprocal`, type aliases), at a pinned sax commit recorded in
  the repo. The checkout is `~/code/gdsfactory/sax`; `424eba8` and the current HEAD
  have identical `src/sax/models`, so either works. Whichever commit is used, record it
  and pin the verification reference to it.
- The jax-to-numpy changes may be done by hand or by a small script. Either way they
  must be recorded in the repo. Known edits:
  - `jax.numpy` → `numpy`;
  - `@jax.jit` → a passthrough;
  - `x.at[i].set(v)` → numpy assignment (in `factories.py`);
  - imports that pull in jax (`jaxellip` in `rf.py`) removed, or the module dropped.
    `rf` is out of scope anyway.
- Apache-2.0: keep sax's licence and NOTICE, and put a §4(b) modification notice on
  every changed file.

### Model library

The vendored sax models, with these changes:

- **Remove the normalisation** in `models/mmis.py:274`:
  `amplitude = max_amplitude * spectral_response` (the true peak is 1 at `f0`).
  A scalar call now carries dispersion. Consequently the `wl_ref` machinery, the
  `[wl, wl_ref]` pair evaluation and `reference_of` are deleted; `wl0` is just an
  ordinary model parameter.
- Registered stdlib: the concrete optical models only (e.g. `straight`, `bend`,
  `attenuator`, `phase_shifter`, `coupler`, `coupler_ideal`, `grating_coupler`,
  `crossing_ideal`, `mmi1x2`, `mmi1x2_ideal`, `mmi2x2`, `mmi2x2_ideal`,
  `splitter_ideal`, `mirror`, `reflector`, `isolator`, `circulator`, `terminator`).
  Not factories (`model_2port`, `unitary`, `copier`, `passthru`, …) and not `rf`.
- socklight intentionally diverges from upstream (the normalisation fix), so later
  re-syncs from sax are manual merges, not a blind re-run.

### Registry and descriptor generation

Flow, per process:

1. `import cspdk` (later) / `import socklight` (stdlib self-registers).
2. `socklight.register(...)` adds callables keyed by resource path
   (`"package.module:callable"`, as pyosdi's `_resolve` understands it).
3. `import vacask`, build a circuit whose netlist says `load "<socklight.osdi_path()>"`.
4. `dlopen` runs a constructor in `socklight.osdi` that takes the GIL, snapshots the
   registry, probes each model, and fills `OSDI_DESCRIPTORS`/`OSDI_NUM_DESCRIPTORS`.

Facts this relies on (verified in VACASK source):

- VACASK reads `OSDI_DESCRIPTORS` (the symbol's *address* as the array base),
  `OSDI_NUM_DESCRIPTORS` (value) and `OSDI_DESCRIPTOR_SIZE` right after `dlopen`
  (`lib/osdifile.cpp:33-60`). So the array must be a statically-sized array in the
  `.osdi`'s data, filled by the constructor; choose a capacity and fail loudly if the
  registry exceeds it.
- `dlopen` of the same path returns the same handle and VACASK caches `OsdiFile` by
  handle, so the constructor runs **once per process**.
- The snapshot freezes **names, ports and parameter signatures**, not the callables.
  `setup_instance` resolves the callable from the registry by resource path on every
  call, so a notebook user who edits a model body and re-registers it picks up the new
  code without a page reload. After the snapshot, `register()` of a *new* name, or of
  an existing name whose ports/parameters differ, raises, naming the cause;
  re-registering an existing name with an identical signature replaces the callable.
  *(Builder-check call.)*
- VACASK lowercases OSDI device and parameter names on load
  (`lib/osdifile.cpp:187,252`); the netlist is case-sensitive. The constructor must
  refuse (at load, loudly) two registered names, or two parameters of one model, that
  collide after lowercasing. The C side keeps the original-case names for calling
  back into Python.
- Unquoted VACASK identifiers are `[a-zA-Z_$][0-9a-zA-Z_$]*`; single-quoted ones may
  contain any non-whitespace (`lib/dfllexer.l:869`, `docs/input-identifiers.md`). So
  the resource path is written quoted, once, on the `model` line.

Per model, the descriptor is:

- `name`: the resource path.
- Ports: from one probe call with default parameters (and `wl=1.55` if it takes
  `wl`); port names sorted, as `sax.get_ports` does. Terminals, in order: for each
  port `p`, `p_i` then `p_q`. No internal nodes.
- Parameters: from `inspect.signature`. A parameter whose default is a float/int is a
  real parameter; a str default is a string parameter; `wl` is a real parameter.
  An int (or bool) default makes the registry kind "int", but VACASK still sees a real,
  and any real value is accepted: the model gets `int(v)` when `v` is integral and
  `float(v)` otherwise (upstream PDKs write `loss_dB_cm=3` for a real quantity).
  A parameter whose default is `None` (e.g. `mmi2x2`'s `loss_dB_cross`/`loss_dB_thru`)
  is an optional real parameter: passed to the model only when given, otherwise
  omitted so the model's own `None` handling applies. *(Builder-check call.)*
  Parameters with any other default (arrays, callables, …) are not exposed;
  if the model cannot be called without them it is rejected at load with a message.
  All parameters are instance parameters, so they may be given on the `model` line
  as defaults or overridden per instance.
- Jacobian: dense 2n×2n over the terminals.

Example deck surface:

```
load "socklight.osdi"
model mmi 'socklight.models:mmi2x2' loss_db=0.3
x1 (o1_i o1_q o2_i o2_q o3_i o3_q o4_i o4_q) mmi wl=1.55
```

### Evaluation

- `setup_instance` (Python allowed): call the model with the instance's parameters at
  scalar `wl`, convert S to `Y = (I − S)(I + S)⁻¹` with the existing Kurokawa
  imaginary-z0 trick (`src/device_body.inc:225-238`: z0 = 1 + 1e-12j, conj(z0) on the diagonal), store the real
  2n×2n block `[[Re Y, −Im Y], [Im Y, Re Y]]` in the instance.
- `eval`/`load_*`: pure C, stamp the stored block. Unchanged in spirit from pyosdi.
- A Python exception in setup makes setup return an OSDI init error, so the
  simulation fails. VACASK can only report a code there. See *Errors*.
- A port set at setup that differs from the descriptor's (parameters changed the
  ports) is a setup error.
- Changing `wl` or any parameter re-enters setup (existing VACASK behaviour, which
  is how the DC `wl` sweep works today: `decks/sweep4.sim`).

### Embedding

- Never call `Py_Initialize*`. If `Py_IsInitialized()` is false at load, export zero
  descriptors and write one line to stderr (standalone `vacask` is unsupported in
  v1).
- Wrap every Python call in `PyGILState_Ensure`/`Release`. VACASK's binding never
  releases the GIL (`python/csrc` has no `gil_scoped_release`) and declares
  `mod_gil_not_used` for free-threaded builds; `PyGILState_Ensure` is correct either
  way.
- Python runs only in the load-time constructor, `setup_model` and
  `setup_instance`.

### Errors (no VACASK changes)

VACASK is not patched. Within OSDI, a `dlopen` constructor cannot fail the load, and
`osdi_log` is still null while it runs (`lib/osdifile.cpp:166-170`). Setup errors
carry only a code (`lib/osdidevice.cpp:612-650`), and `osdi_log` text goes to C++
stderr, which in Pyodide means the browser console. So errors surface on the Python
side, as quietly as that allows:

- **Validate early, in `register()`.** It raises immediately on:
  - a malformed resource path;
  - a name or parameter that collides with another after lowercasing;
  - a probe call that fails;
  - a required parameter that can't be exposed;
  - port or signature changes after the snapshot.

  Most load-time failures thus become ordinary Python exceptions in the notebook cell
  that registered the model.
- **The constructor can still fail.** Causes include no interpreter, a registry over
  capacity, or an unexpected exception. In that case it exports zero descriptors,
  writes a one-line message to stderr, and records the error so that
  `socklight.load_error()` returns it.
- **Setup failures:** the bridge stores the exception (resource path, instance
  parameters, traceback), and `socklight.last_error()` returns it. Whether socklight
  also ships a helper that wraps a VACASK run and re-raises it is the builder's call.
- Acceptance tests check these Python-side surfaces, not VACASK's messages.

### Linking

- Linux: do not link libpython; `Py*` symbols resolve from the host process.
  (pyosdi's "link libpython explicitly"/`RTLD_GLOBAL` traps applied only because it
  embedded its own interpreter.) Verify numpy imports cleanly from the bridge.
- Pyodide: side module; `Py*` resolve against the main module like any extension.
- macOS would need `-undefined dynamic_lookup`, Windows an import of the Python
  DLL. Not built in v1.

### I/Q convention (to be written as its own short document and followed by all devices)

- An optical port is two real nodes, `<port>_i` and `<port>_q`, the in-phase and
  quadrature parts of the complex baseband envelope. This is the same representation
  in DC (steady state, wavelength sweeps), AC (small-signal modulation, ω is the
  electrical frequency) and transient.
- Power waves with real reference resistance 1 Ω: `a = (V + I)/2`, `b = (V − I)/2`,
  `I` flowing into the device port, complex `V = V_i + j·V_q`.
- Optical power carried by a wave is `|a|²` (watts, by definition in this
  representation; no ½).
- Phase convention: sax's, `exp(+jφ)` for propagation.
- Rejected alternative, for the record: one node per port via AC analysis at a fixed
  ω (stamp Re Y resistively, Im Y reactively). It hijacks ω, so electrical AC and
  transient break and `|E|²` detection is impossible.

### Helper devices (Verilog-A, OpenVAF)

- `optical_source` (one optical port + nothing else): launches incident power `p`
  (W) at phase `phase` (rad) into what it connects to, with a matched 1 Ω internal
  impedance, i.e. a Thevenin source of `2·sqrt(p)·e^{j·phase}` behind 1 Ω on the I/Q
  pair, so that the wave it emits has `|a|² = p` into a matched load.
- `optical_term`: matched termination, 1 Ω on each of I and Q (replaces the two
  resistors per unused port in `decks/mmi4.sim`).
- `photodetector`: one optical port, terminated matched, plus two electrical
  terminals `(p, n)` carrying photocurrent `responsivity · P_in` (A/W), where
  `P_in = V_i² + V_q²` at the matched port. Optional `dark_current`.

### Heater (Verilog-A)

- Mirrors cspdk's `ThermalPhaseShifter` (`cspdk/si220/active_models/__init__.py`):
  ports `o1, o2` (as I/Q pairs) and electrical `l_e2, r_e2`; parameters
  `ohms_per_um=0.375, eta_pi_per_W=12.5, length=320, loss_dBcm=3.0, neff0=2.38,
  ng=4.30, wl0=1.55, wl=1.55`; `R = ohms_per_um·length` between `l_e2` and `r_e2`;
  `Δφ = π·eta_pi_per_W·V²/R`; `neff = neff0 − (wl − wl0)(ng − neff0)/wl0`;
  `φ = 2π·neff·length/wl + Δφ`; `T = 10^(−loss/20)·e^{jφ}`.
- Optical behaviour: reciprocal, reflectionless two-port, `S21 = S12 = T`,
  `S11 = S22 = 0`, so it composes with the passive socklight `straight`.
- Must not use a formulation with a `1/(1 − T²)` singularity near `|T| = 1`
  (circulax's docstring documents the problem and its VCVS-plus-matched-input fix;
  prior art, not a requirement on the formulation).

## Verification (what "done" means)

- `import socklight` does not import jax, in an interpreter that has jax installed
  (the `jaxellip` trap from socklu's `JAXFREEDOM_REPORT.md`).
- Each registered stdlib model, called at scalar `wl`, equals sax at the pinned
  vendoring commit, called on the vector `[wl, wl0]` at index 0 — the
  normalisation removal must reproduce sax's *intended* value, not its scalar bug —
  and equals sax's scalar call for models that never normalised. Sample several
  wavelengths.
- Through VACASK: a deck per representative model (2-, 3-, 4-port) whose operating
  point, read back as waves, reproduces the model's S-matrix column for a driven
  port with `optical_source` and `optical_term` elsewhere. A `wl` DC sweep of
  `mmi2x2` shows the passband (pyosdi's `README.md` has the reference numbers for
  `fwhm=0.02` to compare in shape).
- A parameter typo on a socklight device → VACASK parse error.
- Registering a new name or changed signature after load → Python exception.
- A case collision → `register()` raises.
- A model that raises in setup → the run fails, and `socklight.last_error()` names the
  model and carries the traceback.
- Heater, primary reference: the closed-form `T(V, wl)` above. A DC sweep of heater
  voltage on the heater alone (source → heater → matched detector) reproduces
  `|T|²` and, read back as waves, the phase of `T`; a second sweep in an MZI built
  from socklight `coupler`s and a `straight` matches the analytic MZI response.
  Cross-check, secondary: circulax's `ThermalPhaseShifter` in the same forward-only MZI
  (no wave enters its o2, where it is a non-reciprocal VCVS, S12 = 0, S22 = −1, and
  differs from this reciprocal heater by design). The circulax + jax + sax install with
  the cspdk checkout that has `active_models` is `~/code/gdsfactory/cspdk/.venv`; the
  builder writes the circulax MZI script. cspdk's SAX `straight_heater_metal` has a
  different parameter set (`voltage`, `vpi`, …) and is not a reference.
  *(Builder-check call.)*
- One demo deck end to end: source → heater-tuned MZI → photodetector → a TIA built
  from VACASK's bundled devices.
- All of the above that do not need sax/circulax also run in the Pyodide wheel test
  step, mirroring VACASK's `python/tests/test_models.py`.

## Left to the builder

- Python API naming beyond `register`/`osdi_path` (e.g. register a module vs a dict).
- Descriptor capacity, memory ownership of generated strings, logging via `osdi_log`.
- Exact Verilog-A formulation of the source, detector and heater.
- How the deck finds `socklight.osdi` (absolute path in `load`, or VACASK's
  `modulePath`, exposed in the binding at `python/csrc/py_simulator.cpp:52`).
- Build system for the wheel (native + Pyodide), reusing VACASK's CI patterns.
