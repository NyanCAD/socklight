"""socklight: sax's optical models on numpy, for the VACASK circuit simulator."""

import os

from .models import STDLIB
from .registry import last_error, load_error, register, register_models

register_models("socklight.models", STDLIB)


def osdi_path() -> str:
    """Absolute path of ``socklight.osdi``, for a netlist's ``load`` line."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "socklight.osdi")


def va_path(name: str) -> str:
    """Absolute path of a Verilog-A helper's compiled OSDI, e.g. ``va_path("optical_source")``."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "va", f"{name}.osdi")


__all__ = ["last_error", "load_error", "osdi_path", "register", "register_models", "va_path"]
