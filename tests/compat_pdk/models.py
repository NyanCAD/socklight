"""SAX models in the style of cspdk.si220.models: ``jax.numpy as jnp``, ``sax``, ``sax.models``,
``jax.jit``, ``sax.reciprocal``, ``-> sax.SDict`` annotations and an ``interpolate_xarray`` table
(only when xarray is installed)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import sax
import sax.models as sm
from jaxtyping import ArrayLike
from numpy.typing import NDArray

sax.set_port_naming_strategy("optical")

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

XS = {
    "strip": {"neff": 2.38, "ng": 4.30},
    "rib": {"neff": 2.60, "ng": 3.90},
}


def straight(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: str = "strip",
) -> sax.SDict:
    """sax.models.straight with the cross-section's indices."""
    return sm.straight(wl=wl, length=length, loss_dB_cm=loss_dB_cm, wl0=1.55, **XS[cross_section])


@jax.jit
def _kappa(wl: ArrayLike, gap: ArrayLike, length: ArrayLike) -> jax.Array:
    with jax.ensure_compile_time_eval():
        k0 = jnp.asarray(0.08)
    k = k0 * jnp.exp(-(gap - 0.2) / 0.1) * jnp.sqrt(wl / 1.55)
    return jnp.clip(k * length, 0.0, jnp.pi / 2)


def coupler(*, wl: sax.FloatArrayLike = 1.55, gap: float = 0.2, length: float = 10.0) -> sax.SDict:
    """A lossless directional coupler, reciprocal by ``sax.reciprocal``."""
    phi = _kappa(wl, gap, length)
    tau, kappa = jnp.cos(phi), 1j * jnp.sin(phi)
    return sax.reciprocal(
        {("o1", "o4"): tau, ("o2", "o3"): tau, ("o1", "o3"): kappa, ("o2", "o4"): kappa}
    )


mmi1x2 = partial(sm.mmi1x2, wl0=1.55, fwhm=0.2, loss_dB=0.3)
grating_coupler = partial(sm.grating_coupler, wl0=1.554, loss=0.2, bandwidth=0.04)
crossing = jax.jit(sm.crossing_ideal)

try:
    import numpy as np
    import xarray as xr
except ImportError:
    xr = None

if xr is not None:
    _WL = np.array([1.50, 1.55, 1.60])
    _GAP = np.array([0.1, 0.2, 0.3, 0.4])
    _TABLE = xr.DataArray(
        np.stack(
            [
                0.5 + 0.4 * np.cos(_GAP[None, :] * 5) * (_WL[:, None] / 1.55),  # tau
                0.1 * _WL[:, None] + _GAP[None, :] ** 2,  # phase
            ],
            axis=-1,
        ),
        coords={"wl": _WL, "gap": _GAP, "targets": np.array(["tau", "phase"], dtype=object)},
        dims=("wl", "gap", "targets"),
    )

    def coupler_table(*, wl: Float = 1.55, gap: float = 0.2) -> sax.SDict:
        """A coupler from a table, as cspdk's directional couplers are."""
        t = sax.interpolate_xarray(_TABLE, wl=wl, gap=gap)
        tau = t["tau"] * jnp.exp(1j * t["phase"])
        kappa = 1j * jnp.sqrt(1 - t["tau"] ** 2)
        return sax.reciprocal(
            {("o1", "o4"): tau, ("o2", "o3"): tau, ("o1", "o3"): kappa, ("o2", "o4"): kappa}
        )


def get_models() -> dict:
    return {
        name: fn
        for name, fn in globals().items()
        if name in ("straight", "coupler", "mmi1x2", "grating_coupler", "crossing", "coupler_table")
    }
