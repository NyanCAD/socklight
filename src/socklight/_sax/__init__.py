# MODIFIED: vendored into socklight from gdsfactory/sax c22ed91 (Apache-2.0); jax replaced by numpy. See VENDORED.md.
"""The minimum of the ``sax`` namespace that the vendored models reference."""

from ._compat import (
    ArrayLike,
    FloatArray,
    FloatArray1D,
    FloatArray2D,
    FloatArrayLike,
    Name,
    SCoo,
    SCooModel,
    SDict,
    SDictModel,
    reciprocal,
)
from .constants import EPS, WL_C, wl_c
from .ports import PortNamer, get_port_naming_strategy, set_port_naming_strategy

__all__ = [
    "EPS",
    "WL_C",
    "ArrayLike",
    "FloatArray",
    "FloatArray1D",
    "FloatArray2D",
    "FloatArrayLike",
    "Name",
    "PortNamer",
    "SCoo",
    "SCooModel",
    "SDict",
    "SDictModel",
    "get_port_naming_strategy",
    "reciprocal",
    "set_port_naming_strategy",
    "wl_c",
]
