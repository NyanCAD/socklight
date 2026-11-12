"""socklight: sax's optical models on numpy, for the VACASK circuit simulator."""

import os

from .models import STDLIB
from .registry import last_error, load_error, register, register_models

register_models("socklight.models", STDLIB)


def osdi_path() -> str:
    """Absolute path of ``socklight.osdi``, for a netlist's ``load`` line."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "socklight.osdi")


__all__ = ["last_error", "load_error", "osdi_path", "register", "register_models"]
