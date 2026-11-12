"""Functions the C side calls: snapshot(), evaluate(), set_load_error()."""

from __future__ import annotations

import traceback

import numpy as np

from . import registry

__all__ = ["evaluate", "set_load_error", "snapshot"]

_CAST = {"real": float, "int": int, "str": str, "opt": float}


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
            z = complex(np.asarray(s.get((i, j), 0.0)).reshape(-1)[0])
            out.extend((z.real, z.imag))
    return out


def evaluate(name: str, values: dict) -> list[float]:
    """S-matrix of model ``name`` as interleaved ``[re, im]``, row-major over its ports."""
    try:
        spec, fn = registry._models[name]
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
