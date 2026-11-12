# Vendored code

Source: `gdsfactory/sax`, commit `c22ed91` (`src/sax/models` is identical to `424eba8`). Apache-2.0; see `LICENSE` and `NOTICE`.
Every vendored file begins with the `# MODIFIED:` header line.

## Files

| socklight | sax |
|---|---|
| `models/{bends,couplers,crossings,factories,isolators,mmis,probes,reflectors,splitters,straight,terminators}.py` | `src/sax/models/*.py` (not `rf.py`) |
| `_sax/constants.py` | `src/sax/constants.py` |
| `_sax/ports.py` | `src/sax/ports.py` |
| `_sax/_compat.py` | `reciprocal` from `src/sax/s.py`; type aliases from `src/sax/saxtypes` reduced to plain annotations |
| `_sax/__init__.py`, `models/__init__.py` | new; `STDLIB` is the 18-model table |

`factories.py` and `probes.py` are vendored and importable but not in `STDLIB`.

## Edits (jax to numpy)

1. `import jax` removed; `import jax.numpy as jnp` becomes `import numpy as jnp` (alias kept to minimise the diff).
2. `@jax.jit` and `jax.jit(func)` dropped (no passthrough decorator needed; the call sites are just the bare function).
3. `pydantic.validate_call` import and `@validate_call` dropped (numpy-only installs have no pydantic).
4. `import sax` becomes `from socklight import _sax as sax`; `sax.X` references resolve to `socklight._sax`.
5. `S = S.at[idx].set(1)` becomes `S[idx] = 1` in `factories.py` (`S` is a freshly created local array).
6. `sax.into[sax.FloatArray](wl)` becomes `np.asarray(wl, dtype=float)` in `straight.py`.
7. `_sax/constants.py`: `FloatArray1D` imported from `_compat` instead of `sax.saxtypes.core`.
8. `_sax/ports.py`: `sax.Name` becomes `Name` from `_compat`.
9. sax's pydantic type aliases (`FloatArrayLike`, `SDict`, ...) are plain annotations, so inputs are no longer validated or coerced.

Docstring examples still refer to `sax.models.*`, `jax` plotting etc.; they are not executed.
10. `_sax/ports.py`: default `PORT_NAMING_STRATEGY` changed from `"inout"` to `"optical"` (o1, o2, ...), as gdsfactory/cspdk use.
