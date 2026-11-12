"""Optical models (numpy). ``STDLIB`` maps name to model function."""

from __future__ import annotations

from .bends import bend
from .couplers import coupler, coupler_ideal, grating_coupler
from .crossings import crossing_ideal
from .isolators import circulator, isolator
from .mmis import mmi1x2, mmi1x2_ideal, mmi2x2, mmi2x2_ideal
from .reflectors import mirror, reflector
from .splitters import splitter_ideal
from .straight import attenuator, phase_shifter, straight
from .terminators import terminator

STDLIB = {
    "straight": straight,
    "bend": bend,
    "attenuator": attenuator,
    "phase_shifter": phase_shifter,
    "coupler": coupler,
    "coupler_ideal": coupler_ideal,
    "grating_coupler": grating_coupler,
    "crossing_ideal": crossing_ideal,
    "mmi1x2": mmi1x2,
    "mmi1x2_ideal": mmi1x2_ideal,
    "mmi2x2": mmi2x2,
    "mmi2x2_ideal": mmi2x2_ideal,
    "splitter_ideal": splitter_ideal,
    "mirror": mirror,
    "reflector": reflector,
    "isolator": isolator,
    "circulator": circulator,
    "terminator": terminator,
}

__all__ = ["STDLIB", *STDLIB]
