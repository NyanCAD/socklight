"""The ``jax``, ``jax.typing`` and ``jaxtyping`` modules that ``socklight.compat`` serves.

None of this is jax: ``jax.numpy`` is numpy itself, ``jax.jit`` returns the function
unchanged and ``jax.ensure_compile_time_eval`` is a null context. It covers what PDK model
code written for sax uses at import and call time, nothing that traces or differentiates.
"""

from __future__ import annotations

import contextlib
import types
from typing import Any

PROVIDED = ("Array", "config", "ensure_compile_time_eval", "jit", "numpy", "typing")


class Array:
    """Stand-in for ``jax.Array`` in annotations. No value is an instance of it.

    It is deliberately not ``numpy.ndarray``: libraries that look for jax arrays with
    ``isinstance(x, jax.Array)`` (array-api-compat, scipy) must not take numpy arrays for
    jax ones.
    """


class _Config:
    """``jax.config``: stores flags, changes nothing. numpy computes in float64 anyway."""

    def __init__(self) -> None:
        self.jax_enable_x64 = True

    def update(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def read(self, name: str) -> Any:
        return getattr(self, name)


def jit(fun=None, /, **_options):
    """``jax.jit`` as the identity: ``jit(f)``, ``@jit`` and ``partial(jit, static_argnames=...)``
    return ``f`` itself, so ``inspect.signature`` sees the original parameters."""
    if fun is None:
        return lambda f: f
    return fun


def _missing(module: str, provided: tuple[str, ...]):
    def __getattr__(name: str):  # noqa: N807
        if name.startswith("__"):
            raise AttributeError(name)
        msg = (
            f"{module}.{name} is not provided by socklight.compat, which serves jax on numpy "
            f"without tracing; it provides {', '.join(provided)}"
        )
        raise AttributeError(msg)

    return __getattr__


def make_jax() -> types.ModuleType:
    m = types.ModuleType("jax", "socklight.compat: jax.numpy is numpy; jit is the identity. Not jax.")
    m.__version__ = "0.0.0+socklight.compat"
    m.Array = Array
    m.config = _Config()
    m.ensure_compile_time_eval = contextlib.nullcontext
    m.jit = jit
    m.__getattr__ = _missing("jax", PROVIDED)
    return m


def make_jax_typing() -> types.ModuleType:
    m = types.ModuleType("jax.typing", "socklight.compat: annotations only")
    m.ArrayLike = Any
    m.DTypeLike = Any
    m.__getattr__ = _missing("jax.typing", ("ArrayLike", "DTypeLike"))
    return m


class _DtypeAnnotation:
    """``Float[Array, "n"]`` and friends: subscripting gives ``Any``."""

    def __class_getitem__(cls, item):
        return Any


JAXTYPING_DTYPES = ("Bool", "Complex", "Float", "Inexact", "Int", "Integer", "Num", "Real", "Shaped", "UInt")


def make_jaxtyping() -> types.ModuleType:
    m = types.ModuleType("jaxtyping", "socklight.compat: annotations only, no runtime checks")
    m.__version__ = "0.0.0+socklight.compat"
    m.Array = Array
    m.ArrayLike = Any
    m.PyTree = Any
    for name in JAXTYPING_DTYPES:
        setattr(m, name, type(name, (_DtypeAnnotation,), {"__module__": "jaxtyping"}))
    m.jaxtyped = lambda fn=None, **_kw: fn if fn is not None else (lambda f: f)
    m.__getattr__ = _missing("jaxtyping", ("Array", "ArrayLike", "PyTree", "jaxtyped", *JAXTYPING_DTYPES))
    return m
