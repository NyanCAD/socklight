"""Run PDK model code written for jax and sax on numpy, without jax.

``install()`` puts an import hook first on ``sys.meta_path`` that serves

- ``jax``: ``jax.numpy`` *is* numpy; ``jit`` is the identity; ``ensure_compile_time_eval`` is a
  null context; ``config.update`` stores the flag and changes nothing;
- ``jax.typing`` and ``jaxtyping``: annotations only;
- ``sax``: the type aliases (unvalidated annotations), constants, ``reciprocal``, ``PortNamer``,
  ``get/set_port_naming_strategy`` and a numpy ``interpolate_xarray``;
- ``sax.models``: socklight's vendored stdlib (``socklight.models``).

so that an unmodified PDK model module (``import jax.numpy as jnp``, ``import sax``,
``@jax.jit``, ``-> sax.SDict``) imports and runs on numpy, and its models can be registered
with ``socklight.register_models``. Every other ``jax.*``, ``sax.*``, ``jaxlib`` or ``klujax``
import fails with ``ImportError``, so nothing half real loads behind the shim.

It is meant for environments without jax (Pyodide), or to run PDK models without it.
Nothing happens on ``import socklight``: only ``install()`` changes imports, and it refuses
(``RuntimeError``) when a real jax or sax is already imported. Call it before importing the
PDK. Not provided: ``sax.circuit`` (VACASK composes circuits), ``jax.grad``/``vmap``/``lax``/
``random``/``scipy``, ``.at[]`` updates, sax's RF models. Where socklight's models differ from
sax's (the MMI spectral normalisation, VENDORED.md edit 11) the shim has socklight's behaviour.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys

from . import _jax, _sax

__all__ = ["install", "installed", "uninstall"]

_FACTORIES = {
    "jax": _jax.make_jax,
    "jax.typing": _jax.make_jax_typing,
    "jaxtyping": _jax.make_jaxtyping,
    "sax": _sax.make_sax,
    "sax.models": _sax.make_sax_models,
}
_PACKAGES = ("jax", "sax", "sax.models")
_REAL = ("jax", "jaxlib", "sax")  # install() refuses if one of these is already imported
_BLOCKED = ("jax", "jaxlib", "sax", "klujax", "jaxtyping")


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        if spec.name == "jax.numpy":
            self._numpy_spec = sys.modules["numpy"].__spec__
            return sys.modules["numpy"]
        module = _FACTORIES[spec.name]()
        module.__socklight_compat__ = True
        return module

    def exec_module(self, module):
        if module.__name__ == "numpy":
            # The import system has just set numpy.__spec__ to the jax.numpy spec; undo that.
            module.__spec__ = self._numpy_spec
            sys.modules["jax"].numpy = module
            return
        parent, _, child = module.__name__.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
        if module.__name__ == "jax":
            importlib.import_module("jax.numpy")
            importlib.import_module("jax.typing")
        if module.__name__ == "sax":
            importlib.import_module("sax.models")  # `import sax; sax.models.straight` works in sax


class _Finder(importlib.abc.MetaPathFinder):
    """Serves the shims and refuses every other jax/sax/jaxlib/klujax module."""

    def __init__(self, serve_jaxtyping: bool) -> None:
        self.names = {n for n in [*_FACTORIES, "jax.numpy"] if serve_jaxtyping or n != "jaxtyping"}
        self.blocked = tuple(b for b in _BLOCKED if serve_jaxtyping or b != "jaxtyping")

    def find_spec(self, name, path=None, target=None):
        if name in self.names:
            import numpy  # noqa: F401  (jax.numpy is numpy)

            return importlib.machinery.ModuleSpec(name, _Loader(), is_package=name in _PACKAGES)
        if name.split(".")[0] in self.blocked:
            if name == "sax.circuits" or name.startswith("sax.backends"):
                msg = _sax.CIRCUIT_MSG.format(name=name.removeprefix("sax."))
            else:
                msg = f"{name} is not provided by socklight.compat (jax and sax on numpy; see socklight.compat's docstring)"
            raise ImportError(msg, name=name)
        return None


def _finder() -> _Finder | None:
    return next((f for f in sys.meta_path if isinstance(f, _Finder)), None)


def _is_shim(module: object) -> bool:
    return getattr(module, "__socklight_compat__", False) is True


def installed() -> bool:
    """Whether ``install()`` is in effect."""
    return _finder() is not None


def install() -> None:
    """Serve ``jax``, ``jax.numpy``, ``sax``, ``sax.models`` (and ``jax.typing``, ``jaxtyping``)
    from numpy and socklight. Idempotent.

    Raises ``RuntimeError`` if a real jax, jaxlib or sax is already imported: call it before
    anything imports them (in particular before importing the PDK). A ``jaxtyping`` that is
    already imported is left alone.
    """
    if installed():
        return
    real = [n for n in _REAL if n in sys.modules and not _is_shim(sys.modules[n])]
    if real:
        msg = (
            f"socklight.compat.install(): real {', '.join(real)} already imported; the shim would "
            "mix with it. Call install() before anything imports jax or sax, or use the real ones."
        )
        raise RuntimeError(msg)
    sys.meta_path.insert(0, _Finder(serve_jaxtyping="jaxtyping" not in sys.modules))


def uninstall() -> None:
    """Remove the import hook and the shim modules from ``sys.modules``.

    Modules that already imported the shims keep their references.
    """
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _Finder)]
    for name, module in list(sys.modules.items()):
        if _is_shim(module) or (name == "jax.numpy" and module is sys.modules.get("numpy")):
            del sys.modules[name]
