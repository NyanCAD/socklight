# MODIFIED: vendored into socklight from gdsfactory/sax c22ed91 (Apache-2.0); jax replaced by numpy. See VENDORED.md.
"""Minimal stand-ins for sax types and helpers, numpy only.

Type aliases are plain annotations (sax's are pydantic-validated); ``reciprocal``
is copied from ``sax.s``.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

Name: TypeAlias = str
ArrayLike: TypeAlias = np.ndarray
FloatArray: TypeAlias = np.ndarray
FloatArray1D: TypeAlias = np.ndarray
FloatArray2D: TypeAlias = np.ndarray
FloatArrayLike: TypeAlias = Any
SDict: TypeAlias = "dict[tuple[str, str], Any]"
SDictModel: TypeAlias = Any
SCoo: TypeAlias = tuple
SCooModel: TypeAlias = Any


def reciprocal(sdict: SDict) -> SDict:
    """Make an SDict S-matrix reciprocal by ensuring S[i,j] = S[j,i].

    Copies values without conjugation. If both directions are supplied with
    conflicting values, the first direction in dictionary order defines both.
    """
    result = dict(sdict)
    seen = set()
    for (p1, p2), value in sdict.items():
        if (p1, p2) not in seen:
            result[p1, p2] = result[p2, p1] = value
            seen.update(((p1, p2), (p2, p1)))
    return result
