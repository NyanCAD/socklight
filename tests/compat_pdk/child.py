"""Evaluates the compat_pdk models in a fresh interpreter and writes JSON (tests/test_compat.py).

usage: python child.py compat|shim|real OUT.json

shim:   real jax/sax are made unimportable, then ``socklight.compat.install()``.
compat: as shim, and the models are also registered with socklight and one coupler is run
        through VACASK.
real:   real jax (x64) and sax; socklight is not imported.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
REAL = ("jax", "jaxlib", "sax", "klujax", "jaxtyping")

CASES = [
    *[(name, {"wl": wl}) for name in ("straight", "coupler", "grating_coupler", "crossing", "coupler_table") for wl in (1.53, 1.55, 1.57)],
    ("mmi1x2", {"wl": 1.55}),  # at wl0 only: away from it socklight's MMI differs (VENDORED.md edit 11)
    ("straight", {"wl": 1.56, "cross_section": "rib", "length": 25.0}),
    ("coupler", {"gap": 0.25, "length": 5.0}),
    ("coupler_table", {"wl": 1.52, "gap": 0.25}),
    ("coupler_table", {"wl": 1.70, "gap": 0.05}),  # clamped at both edges
]


class Blocker:
    """Makes the real jax/jaxlib/sax/klujax/jaxtyping unimportable (the shim's finder goes in front)."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in REAL:
            raise ImportError(f"blocked real {name}")


def _c(v) -> list[float]:
    import numpy as np

    z = complex(np.asarray(v).reshape(-1)[0])
    return [z.real, z.imag]


def evaluate(pm) -> dict:
    models = pm.get_models()
    out = {}
    for name, kw in CASES:
        if name in models:
            s = models[name](**kw)
            out[f"{name}|{json.dumps(kw, sort_keys=True)}"] = {f"{a},{b}": _c(v) for (a, b), v in s.items()}
    return out


def vacask(models) -> dict:
    """The table coupler (else the analytic one) driven at o1 in VACASK, against S/(1+S11)."""
    import vacask as vk
    from conftest import elaborate, analyse
    from test_osdi import bench, deck, wave

    import socklight

    socklight.register_models("compat_pdk.models", models)
    name = "coupler_table" if "coupler_table" in models else "coupler"
    kw = {"wl": 1.54, "gap": 0.23}
    ports = ["o1", "o2", "o3", "o4"]
    params = " ".join(f"{k}={v}" for k, v in kw.items())
    mod_dir = os.path.join(os.path.dirname(vk.__file__), "mod")
    cir, _keep = elaborate(deck(mod_dir, {"m": f"'compat_pdk.models:{name}'"}, bench("a", "m", ports, params)))
    v = {k: float(x[0]) for k, x in analyse(cir).items()}
    s = models[name](**kw)
    s11 = complex(s.get(("o1", "o1"), 0.0))
    return {
        "model": name,
        "got": {p: _c(wave(v, f"a{p}")) for p in ports[1:]},
        "want": {p: _c(complex(s.get((p, "o1"), 0.0)) / (1 + s11)) for p in ports[1:]},
        "last_error": socklight.last_error(),
    }


def main(mode: str, out: str) -> None:
    sys.path[:0] = [TESTS]
    if mode in ("compat", "shim"):
        sys.meta_path.insert(0, Blocker())
        import socklight.compat

        socklight.compat.install()
    else:
        import jax

        jax.config.update("jax_enable_x64", True)
    import compat_pdk.models as pm

    res = {"S": evaluate(pm), "models": sorted(pm.get_models())}
    if mode == "compat":
        res["vacask"] = vacask(pm.get_models())
    if mode in ("compat", "shim"):
        res["real_modules"] = sorted(
            n for n, m in sys.modules.items() if n.split(".")[0] in REAL and not getattr(m, "__socklight_compat__", n == "jax.numpy")
        )
    with open(out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
