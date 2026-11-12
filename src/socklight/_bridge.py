"""Functions the C side calls: snapshot(), evaluate(), set_load_error()."""

from __future__ import annotations

import traceback

import numpy as np

from . import registry

__all__ = ["evaluate", "record_error", "set_load_error", "snapshot"]

def _int(v: object) -> int:
    if isinstance(v, float) and not v.is_integer():
        msg = f"integer parameter got non-integral value {v!r}"
        raise ValueError(msg)
    return int(v)


_CAST = {"real": float, "int": _int, "str": str, "opt": float}


def snapshot() -> list[tuple[str, tuple[str, ...], tuple[tuple[str, str, object], ...]]]:
    """Every registered model as ``(name, ports, ((pname, kind, default), ...))``; freezes the registry."""
    registry._frozen = True
    return [
        (spec.name, spec.ports, tuple((p.name, p.kind, p.default) for p in spec.params))
        for spec, _ in registry._models.values()
    ]


def _flatten(s: dict, ports: tuple[str, ...], name: str) -> list[float]:
    got = tuple(sorted({port for key in s for port in key}))
    if got != ports:
        msg = f"{name} returned ports {got}, registered with {ports}"
        raise ValueError(msg)
    out: list[float] = []
    for i in ports:
        for j in ports:
            arr = np.asarray(s.get((i, j), 0.0))
            if arr.size != 1:
                msg = f"{name} entry {(i, j)} has {arr.size} elements, expected a scalar"
                raise ValueError(msg)
            z = complex(arr.reshape(-1)[0])
            if not (np.isfinite(z.real) and np.isfinite(z.imag)):
                msg = f"{name} entry {(i, j)} is not finite"
                raise ValueError(msg)
            out.extend((z.real, z.imag))
    return out


def evaluate(name: str, values: dict) -> list[float]:
    """S-matrix of model ``name`` as interleaved ``[re, im]``, row-major over its ports."""
    try:
        spec, fn = registry._models[name]
        unknown = sorted(set(values) - {p.name for p in spec.params})
        if unknown:
            msg = f"{name}: unknown parameters {unknown}"
            raise ValueError(msg)
        kwargs = {}
        for p in spec.params:
            if p.name in values:
                kwargs[p.name] = _CAST[p.kind](values[p.name])
        return _flatten(fn(**kwargs), spec.ports, name)
    except BaseException:
        registry._last_error = {"model": name, "params": dict(values), "traceback": traceback.format_exc()}
        raise


def set_load_error(msg: str) -> None:
    registry._load_error = msg


def record_error(name: str, message: str) -> None:
    """A setup failure found on the C side (no Python traceback): make last_error() report it."""
    registry._last_error = {"model": name, "params": {}, "traceback": message}
