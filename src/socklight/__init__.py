"""socklight: sax's optical models on numpy, for the VACASK circuit simulator."""

from .models import STDLIB
from .registry import last_error, load_error, register, register_models

register_models("socklight.models", STDLIB)

__all__ = ["last_error", "load_error", "register", "register_models"]
