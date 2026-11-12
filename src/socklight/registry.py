"""Model registry: name -> callable, with a probed, comparable ModelSpec."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

__all__ = ["ModelSpec", "Param", "register", "register_models", "load_error", "last_error"]

_NAME_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
PROBE_WL = 1.55


@dataclass(frozen=True)
class Param:
    name: str
    kind: Literal["real", "int", "str", "opt"]
    default: float | int | str | None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ports: tuple[str, ...]
    params: tuple[Param, ...]
    takes_wl: bool


_models: dict[str, tuple[ModelSpec, Callable]] = {}
_frozen = False
_load_error: str | None = None
_last_error: dict | None = None


def _reset() -> None:
    """Test hook: forget every model, the frozen flag and both errors."""
    global _frozen, _load_error, _last_error
    _models.clear()
    _frozen = False
    _load_error = None
    _last_error = None


def _param(name: str, default: object) -> Param | None:
    if name == "wl":
        return Param("wl", "real", float(default) if isinstance(default, (int, float)) and not isinstance(default, bool) else PROBE_WL)
    if default is None:
        return Param(name, "opt", None)
    if isinstance(default, bool):
        return Param(name, "int", int(default))
    if isinstance(default, int):
        return Param(name, "int", default)
    if isinstance(default, float):
        return Param(name, "real", default)
    if isinstance(default, str):
        return Param(name, "str", default)
    return None  # not exposed


def _probe(name: str, fn: Callable) -> ModelSpec:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        msg = f"{name}: cannot inspect signature: {exc}"
        raise ValueError(msg) from exc
    params: list[Param] = []
    seen: dict[str, str] = {}
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is p.empty:
            msg = f"{name}: parameter {p.name!r} has no default"
            raise ValueError(msg)
        if p.name.lower() in seen:
            msg = f"{name}: parameters {seen[p.name.lower()]!r} and {p.name!r} collide case-insensitively"
            raise ValueError(msg)
        seen[p.name.lower()] = p.name
        param = _param(p.name, p.default)
        if param is not None:
            params.append(param)
    takes_wl = any(p.name == "wl" for p in params)
    s = fn(**({"wl": PROBE_WL} if takes_wl else {}))
    if not isinstance(s, Mapping) or not s:
        msg = f"{name}: probe returned {type(s).__name__}, expected a non-empty SDict"
        raise ValueError(msg)
    ports = tuple(sorted({port for key in s for port in key}))
    return ModelSpec(name, ports, tuple(params), takes_wl)


def _difference(old: ModelSpec, new: ModelSpec) -> str:
    """What makes ``new`` a different signature from ``old``."""
    if old.ports != new.ports:
        return f"ports changed from {old.ports} to {new.ports}"
    names = [p.name for p in old.params], [p.name for p in new.params]
    if names[0] != names[1]:
        return f"parameters changed from {names[0]} to {names[1]}"
    for a, b in zip(old.params, new.params):
        if a.kind != b.kind:
            return f"parameter {a.name!r} kind changed from {a.kind} to {b.kind}"
        if a.default != b.default:
            return f"parameter {a.name!r} default changed from {a.default!r} to {b.default!r}"
    return "the signature changed"


def register(name: str, fn: Callable) -> None:
    """Register ``fn`` as model ``name`` (``package.module:callable``)."""
    if not _NAME_RE.match(name):
        msg = f"model name must match 'package.module:callable', got {name!r}"
        raise ValueError(msg)
    lowered = name.lower()
    for other in _models:
        if other != name and other.lower() == lowered:
            msg = f"model name {name!r} collides case-insensitively with {other!r}"
            raise ValueError(msg)
    if _frozen and name not in _models:
        msg = f"socklight.osdi already loaded; cannot add model {name!r}; register it before loading and restart"
        raise RuntimeError(msg)
    spec = _probe(name, fn)
    if _frozen and _models[name][0] != spec:
        msg = f"socklight.osdi already loaded; {name!r}: {_difference(_models[name][0], spec)}; restart to apply"
        raise RuntimeError(msg)
    _models[name] = (spec, fn)


def register_models(module_path: str, models: Mapping[str, Callable]) -> None:
    for key, fn in models.items():
        register(f"{module_path}:{key}", fn)


def load_error() -> str | None:
    return _load_error


def last_error() -> dict | None:
    return _last_error
