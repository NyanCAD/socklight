# MODIFIED: interpolate_xarray ported to numpy from gdsfactory/sax 0.18.2 and jax 0.8.3 (Apache-2.0). See VENDORED.md.
"""The ``sax`` and ``sax.models`` modules that ``socklight.compat`` serves.

``sax.models`` is socklight's vendored stdlib (``socklight.models``), each model wrapped so it
names its ports by the shim's own port naming strategy. ``sax`` holds the type aliases (as
annotations), the constants, ``reciprocal``, the port naming functions and a numpy
``interpolate_xarray``. Circuit composition (``sax.circuit``) is VACASK's job and is not here.
"""

from __future__ import annotations

import functools
import itertools
import operator
import types
from typing import Annotated, Any

import numpy as np

from socklight import _sax
from socklight._sax import constants as _constants
from socklight._sax import ports as _ports

# ── port naming ────────────────────────────────────────────────────────────────
#
# socklight's own models (``socklight.models``, registered on ``import socklight``) always
# name their ports o1, o2, ... (VENDORED.md edit 10). The shim keeps a separate strategy,
# defaulting to sax's "inout", so that PDK code calling ``sax.set_port_naming_strategy``
# changes what ``sax.models.*`` return and never the ports of socklight's registered
# stdlib devices.

_strategy: dict[str, str] = {"value": "inout"}


def set_port_naming_strategy(strategy: str) -> str:
    """``sax.set_port_naming_strategy``: "optical" (o1, o2, ...) or "inout" (in0, out0, ...)."""
    if str(strategy).lower() not in ("optical", "inout"):
        msg = f"Invalid port naming strategy: {strategy}"
        raise ValueError(msg)
    _strategy["value"] = strategy
    return strategy


def get_port_naming_strategy() -> str:
    return _strategy["value"]


class PortNamer(_ports.PortNamer):
    """``sax.PortNamer``, defaulting to the shim's strategy."""

    def __init__(self, num_inputs: int, num_outputs: int, strategy: str | None = None) -> None:
        super().__init__(num_inputs, num_outputs, strategy or get_port_naming_strategy())


def _with_strategy(fn):
    """``fn`` run with socklight's module-wide strategy set to the shim's for the call.

    Not thread-safe: two threads calling models under different strategies could see each
    other's. The signature is kept (``functools.wraps``), so ``socklight.register`` and
    ``functools.partial`` see the original parameters.
    """

    @functools.wraps(fn)
    def model(*args, **kwargs):
        saved = _ports.PORT_NAMING_STRATEGY
        _ports.PORT_NAMING_STRATEGY = _strategy["value"]
        try:
            return fn(*args, **kwargs)
        finally:
            _ports.PORT_NAMING_STRATEGY = saved

    return model


# ── type aliases ───────────────────────────────────────────────────────────────


class _Alias:
    """Metadata that keeps each alias distinct: ``sax.SDict != sax.SType``, as in sax."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"sax.{self.name}"


# sax's pydantic-validated aliases (sax 0.18 ``__all__``), as unvalidated annotations.
ALIASES = (
    "AnyNetlist ArrayLike Backend BackendLike Bool BoolArray BoolArrayLike BoolLike Complex "
    "ComplexArray ComplexArray1D ComplexArray1DLike ComplexArrayLike ComplexLike Component "
    "Connections Float FloatArray FloatArray1D FloatArray1DLike FloatArray2D FloatArray2DLike "
    "FloatArrayLike FloatLike IOLike Instance InstanceName InstancePort Instances Int IntArray "
    "IntArray1D IntArray1DLike IntArrayLike IntLike Mode Model ModelFactory ModelFactoryMM "
    "ModelFactorySM ModelMM ModelSM Models ModelsMM ModelsSM Name Net Netlist Nets "
    "Normalization Placement Placements Port PortCombination PortCombinationMM "
    "PortCombinationSM PortMap PortMapMM PortMapSM PortMode Ports RecursiveNetlist SCoo SCooMM "
    "SCooModel SCooModelFactory SCooModelFactoryMM SCooModelFactorySM SCooModelMM SCooModelSM "
    "SCooSM SDense SDenseMM SDenseModel SDenseModelFactory SDenseModelFactoryMM "
    "SDenseModelFactorySM SDenseModelMM SDenseModelSM SDenseSM SDict SDictMM SDictModel "
    "SDictModelFactory SDictModelFactoryMM SDictModelFactorySM SDictModelMM SDictModelSM "
    "SDictSM SType STypeMM STypeSM Settings SettingsValue"
).split()


# ── interpolate_xarray ─────────────────────────────────────────────────────────


def interpolate_xarray(xarr, /, **kwargs) -> dict[str, np.ndarray]:
    """``sax.interpolate_xarray`` (sax 0.18) on numpy.

    The last dimension of ``xarr`` holds the target names (strings). Numeric dimensions are
    interpolated multilinearly, clamped at the edges, and default to the mean of their
    coordinates; string dimensions are selected by value. The steps, and the order in which
    the corners are summed, are those of sax's ``_evaluate_general_corner_model`` over
    ``jax.scipy.ndimage.map_coordinates(order=1, mode="nearest")``, so float64 results agree
    with sax under ``jax_enable_x64`` to rounding.
    """
    data = np.asarray(xarr)
    params = {str(k): np.asarray(xarr[k]) for k in xarr.dims}
    new_params, strings = _extract_strings(params)
    target_name = str(xarr.dims[-1])
    targets = strings[target_name]
    if target_name in kwargs:
        msg = f"Cannot interpolate over target parameter {target_name}."
        raise ValueError(msg)
    string_kwargs = {k: v for k, v in kwargs.items() if k in strings}
    new_kwargs = {**kwargs, **{k: [str(v)] for k, v in string_kwargs.items()}}

    s, axs, pos = _evaluate_general_corner_model(data, new_params, strings, **new_kwargs)

    axs_rev = dict(sorted([(v, k) for k, v in axs.items() if k != target_name], reverse=True))
    ss = {k: s.take(pos[target_name][k], axs[target_name]) for k in targets}
    for ax, name in axs_rev.items():
        if name in string_kwargs:
            ss = {kk: s.take(pos[name][string_kwargs[name]], ax) for kk, s in ss.items()}
    return ss


def _extract_strings(params):
    string_map: dict[str, dict[str, int]] = {}
    new_params = {}
    for k, v in params.items():
        if v.dtype == object or v.dtype.kind in "US":
            string_map[k] = {str(s): i for i, s in enumerate(v)}
            new_params[k] = np.array(list(string_map[k].values()), dtype=int)
        else:
            new_params[k] = np.array(v, dtype=float)
    return new_params, string_map


def _evaluate_general_corner_model(data, params, strings, /, **kwargs):
    given_params = {k: kwargs.get(k, np.asarray(v).mean()) for k, v in params.items()}
    given_strings = {
        p: (list(v.values()) if p not in kwargs else [v[vv] for vv in kwargs[p]])
        for p, v in strings.items()
    }
    string_locations = {k: i for i, k in enumerate(given_params) if k in given_strings}
    param_locations = {k: i for i, k in enumerate(given_params) if k not in given_strings}
    data = np.transpose(data, axes=[*param_locations.values(), *string_locations.values()])
    given_params = {k: v for k, v in given_params.items() if k not in given_strings}
    num_params = len(given_params)
    for i, idxs in enumerate(reversed(list(given_strings.values())), start=1):
        data = data.take(np.asarray(idxs, dtype=int), axis=-i)
    param_shape, string_shape = data.shape[:num_params], data.shape[num_params:]
    data = data.reshape(*param_shape, -1)
    values = np.stack(np.broadcast_arrays(*[np.asarray(v) for v in given_params.values()]), 0)
    coords = [
        np.interp(v, a, np.arange(a.shape[0]))
        for a, v in zip([v for k, v in params.items() if k in given_params], values, strict=True)
    ]
    result = np.stack([_map_coordinates(data[..., j], coords) for j in range(data.shape[-1])], 0)
    axs = {k: i for i, k in enumerate(given_strings)}
    rev_strings = {k: {vv: kk for kk, vv in v.items()} for k, v in strings.items()}
    pos = {k: {rev_strings[k][vv]: i for i, vv in enumerate(v)} for k, v in given_strings.items()}
    return result.reshape(*string_shape, *result.shape[1:]), axs, pos


def _map_coordinates(arr: np.ndarray, coords: list[np.ndarray]) -> np.ndarray:
    """``jax.scipy.ndimage.map_coordinates(arr, coords, order=1, mode="nearest")``."""
    per_dim = []
    for c, size in zip(coords, arr.shape, strict=True):
        lower = np.floor(c)
        upper_weight = c - lower
        index = lower.astype(np.int32)
        per_dim.append(
            [(np.clip(index, 0, size - 1), 1 - upper_weight), (np.clip(index + 1, 0, size - 1), upper_weight)]
        )
    outputs = []
    for items in itertools.product(*per_dim):
        indices, weights = zip(*items)
        outputs.append(functools.reduce(operator.mul, weights) * arr[indices])
    result = functools.reduce(operator.add, outputs)
    if np.issubdtype(arr.dtype, np.integer):
        result = np.sign(result) * np.floor(np.abs(result) + 0.5)
    return np.asarray(result).astype(arr.dtype)


# ── not provided ───────────────────────────────────────────────────────────────

CIRCUIT_MSG = (
    "sax.{name} is not provided by socklight.compat: circuits are composed by VACASK, not "
    "sax. Register the models with socklight.register_models() and build a deck with "
    "socklight.netlist.netlist_to_deck() (see the socklight README)."
)

_CIRCUIT_NAMES = ("circuit", "evaluate_circuit", "analyze_circuit", "analyze_instances", "get_required_circuit_models")


def _circuit_stub(name: str):
    def stub(*_args, **_kwargs):
        raise NotImplementedError(CIRCUIT_MSG.format(name=name))

    stub.__name__ = stub.__qualname__ = name
    stub.__doc__ = CIRCUIT_MSG.format(name=name)
    return stub


def _missing(module: str, hint: str):
    def __getattr__(name: str):  # noqa: N807
        if name.startswith("__"):
            raise AttributeError(name)
        msg = f"{module}.{name} is not provided by socklight.compat (sax on numpy, models only). {hint}"
        raise AttributeError(msg)

    return __getattr__


# ── the modules ────────────────────────────────────────────────────────────────


def make_sax() -> types.ModuleType:
    _strategy["value"] = "inout"
    m = types.ModuleType("sax", "socklight.compat: sax's model-facing API on numpy. Not sax.")
    m.__version__ = "0.0.0+socklight.compat"
    for name in ALIASES:
        setattr(m, name, Annotated[Any, _Alias(name)])
    for name in _constants.__all__:
        setattr(m, name, getattr(_constants, name))
    m.reciprocal = _sax.reciprocal
    m.PortNamer = PortNamer
    m.get_port_naming_strategy = get_port_naming_strategy
    m.set_port_naming_strategy = set_port_naming_strategy
    m.interpolate_xarray = interpolate_xarray
    for name in _CIRCUIT_NAMES:
        setattr(m, name, _circuit_stub(name))
    m.__getattr__ = _missing(
        "sax",
        "It provides the type aliases, constants, reciprocal, PortNamer, get/set_port_naming_strategy, "
        "interpolate_xarray and sax.models.",
    )
    return m


def make_sax_models() -> types.ModuleType:
    from socklight.models import STDLIB, factories, probes

    m = types.ModuleType("sax.models", "socklight.compat: socklight.models (sax's stdlib on numpy)")
    models = {
        **STDLIB,
        **{n: getattr(factories, n) for n in ("copier", "model_2port", "model_3port", "model_4port", "passthru", "unitary")},
        "ideal_probe": probes.ideal_probe,
    }
    for name, fn in models.items():
        setattr(m, name, _with_strategy(fn))
    m.__all__ = sorted(models)
    m.__getattr__ = _missing(
        "sax.models",
        "socklight vendors sax's optical models only; the RF models (sax.models.rf) are not included.",
    )
    return m
