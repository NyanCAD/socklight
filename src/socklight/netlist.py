"""A gdsfactory/sax flat netlist dict -> a VACASK deck of socklight devices.

The input is what ``Component.get_netlist()`` returns (no gdsfactory or sax is imported)::

    {"instances": {name: {"component": str, "settings": {...}, "info": {...}}},
     "nets": [{"p1": "inst,port", "p2": "inst,port"}, ...],
     "ports": {external: "inst,port"}}

Rules, chosen to match ``sax.circuit``:

- Each instance's parameters are ``{**settings, **info}`` (sax merges ``info`` over
  ``settings``). Keys that are not parameters of the registered model are left out and
  reported in ``Deck.dropped``; ``None`` values are omitted, so the model's default applies.
- ``params`` (e.g. ``{"wl": 1.55}``) go to every instance whose model has that parameter
  (matched case-insensitively), except where the instance itself sets it: the instance wins.
- ``variables`` (e.g. ``("wl",)``) works like ``params`` but writes the circuit variable's name
  (``wl=wl``) instead of a value, so one ``sweep s variable="wl" values=[...]`` moves every
  instance. The caller must declare each variable (``var wl=1.55`` in the control block)
  before elaboration; a name in both ``params`` and ``variables`` raises ``ValueError``.
- Values: bool -> 0/1, int -> int, other reals -> ``repr(float)`` (full precision),
  str -> a double-quoted VACASK string; anything else raises ``TypeError``.
- Nodes: one I/Q pair per net (``n<k>_i n<k>_q``, k the net's index); an external port
  ``p`` names its node ``p`` (``p_i p_q``). An instance port on no net and not external is
  terminated with ``optical_term`` (model name ``optical_term`` in the deck, always defined).
- ``va_devices`` maps a component to one of socklight's Verilog-A devices (``VA_DEVICES``,
  e.g. ``{"straight_heater_meander": "heater"}``) instead of a registered model. Its
  terminals follow the module's declaration order; a terminal pair ``<p>_i``/``<p>_q`` is
  optical port ``p``, every other terminal is electrical port of that name. Settings are
  matched to the module's parameters case-insensitively (written lowercase); ``params`` and
  ``variables`` apply the same way. An electrical port is one node: shared with the other
  electrical ports on its nets, named after its external port if it has one, and otherwise
  ``<instance>_<port>``, left unconnected (not terminated) so the caller can drive it.
  ``Deck.electrical`` lists every electrical port's node. A net joining an optical and an
  electrical port raises ``ValueError``.
- Instance terminals follow the registered model's sorted port order. Instances are written
  in sorted order. Names that are not valid VACASK identifiers are sanitized (non-identifier
  characters -> ``_``, a leading digit gets ``_``, line-start keywords get a trailing ``_``)
  and made unique with ``_2``, ``_3``, ...
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from . import registry

__all__ = ["TERM_MODEL", "VA_DEVICES", "Deck", "VADevice", "netlist_to_deck"]

TERM_MODEL = "optical_term"
"""Local model name of the ``optical_term`` the deck loads; usable by lines appended to it."""

# Words VACASK reads as keywords at the start of a line (dfllexer.l), plus the term model.
_RESERVED = {"model", "global", "ground", "load", "embed", "subckt", "ends", "parameters", "control", "endc"}
_NON_IDENT = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class VADevice:
    """A socklight Verilog-A device: its module terminals (declaration order) and parameters
    (lowercase, declaration order). The module and the ``va_path`` name are the dict key."""

    terminals: tuple[str, ...]
    params: tuple[str, ...]

    @property
    def optical(self) -> tuple[str, ...]:
        """Optical ports: ``p`` for each terminal pair ``p_i``, ``p_q``."""
        return tuple(t[:-2] for t in self.terminals if t.endswith("_i") and f"{t[:-2]}_q" in self.terminals)

    @property
    def electrical(self) -> tuple[str, ...]:
        """Electrical ports: the terminals that are not half of an optical pair."""
        pairs = {f"{p}{s}" for p in self.optical for s in ("_i", "_q")}
        return tuple(t for t in self.terminals if t not in pairs)


VA_DEVICES: dict[str, VADevice] = {
    "heater": VADevice(
        ("o1_i", "o1_q", "o2_i", "o2_q", "l_e2", "r_e2"),
        ("ohms_per_um", "eta_pi_per_w", "length", "loss_dbcm", "neff0", "ng", "wl0", "wl"),
    ),
    "photodetector": VADevice(("o_i", "o_q", "p", "n"), ("responsivity", "dark_current")),
    "optical_source": VADevice(("o_i", "o_q"), ("p", "phase")),
    "optical_term": VADevice(("o_i", "o_q"), ()),
}
"""socklight's Verilog-A devices (``va/*.va``, compiled to ``socklight.va_path(name)``)."""


@dataclass(frozen=True)
class Deck:
    text: str  # a complete VACASK deck body: title, ground, loads, model lines, instances
    external: dict[str, tuple[str, str]]  # optical external port -> (i node, q node)
    dropped: dict[str, set[str]]  # component -> settings not passed (not model parameters)
    variables: tuple[str, ...] = ()  # circuit variables the deck references: declare them (``var``)
    electrical: dict[str, dict[str, str]] = field(default_factory=dict)  # VA instance -> port -> node


class _Names:
    """Allocates unique VACASK identifiers. With ``suffixes``, the name is a stem: every
    ``name + suffix`` must be free, and all of them are taken (an I/Q pair is ``("_i", "_q")``)."""

    def __init__(self, taken=()):
        self.taken = set(taken)

    def __call__(self, raw: str, suffixes: tuple[str, ...] = ("",)) -> str:
        base = _NON_IDENT.sub("_", str(raw)) or "_"
        if base[0].isdigit():
            base = "_" + base
        if base in _RESERVED:
            base += "_"
        name, k = base, 2
        while any(name + x in self.taken for x in suffixes):
            name, k = f"{base}_{k}", k + 1
        self.taken.update(name + x for x in suffixes)
        return name


def _value(v: object, where: str) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):
        f = float(v)
        if not math.isfinite(f):
            msg = f"{where}: non-finite value {v!r} cannot go in a deck"
            raise ValueError(msg)
        return repr(f)
    if isinstance(v, str):
        esc = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return f'"{esc}"'
    msg = f"{where}: cannot put {type(v).__name__} {v!r} in a deck"
    raise TypeError(msg)


def _ref(ref: str) -> tuple[str, str]:
    parts = str(ref).split(",")
    if len(parts) != 2:
        msg = f"port reference {ref!r} is not 'instance,port'"
        raise ValueError(msg)
    return parts[0], parts[1]


def _resolve(component: str, models: str | Mapping[str, str]) -> str:
    if isinstance(models, str):
        name = f"{models}:{component}"
    elif component in models:
        name = models[component]
    else:
        msg = f"component {component!r} has no entry in the models mapping"
        raise KeyError(msg)
    if name not in registry._models:
        msg = f"component {component!r}: model {name!r} is not registered"
        raise KeyError(msg)
    return name


def netlist_to_deck(
    netlist: Mapping,
    models: str | Mapping[str, str],
    *,
    params: Mapping[str, object] | None = None,
    variables: Iterable[str] = (),
    va_devices: Mapping[str, str] | None = None,
    title: str = "socklight netlist",
) -> Deck:
    """Converts a gdsfactory/sax flat netlist to a VACASK deck (see the module docstring).

    ``models`` is a prefix (``"cspdk.si220"``: component ``coupler`` uses the registered
    ``"cspdk.si220:coupler"``) or a mapping component -> registered name. ``va_devices`` maps
    components to socklight Verilog-A devices instead (``{"straight_heater_meander": "heater"}``,
    names from ``VA_DEVICES``); those components need no registered model. The deck has no
    analysis or control block (``Deck.variables`` need declaring in one), and drives nothing:
    append sources and detectors to ``Deck.text`` using the nodes in ``Deck.external`` (optical)
    and ``Deck.electrical`` (the Verilog-A devices' electrical ports).
    """
    import socklight

    instances: Mapping = netlist["instances"]
    glob = {str(k).lower(): v for k, v in (params or {}).items()}
    if isinstance(variables, str):
        variables = (variables,)
    variables = tuple(variables)
    var_of = {v.lower(): v for v in variables}
    both = sorted(set(var_of) & set(glob))
    if both:
        msg = f"{both} are in both params and variables"
        raise ValueError(msg)
    va_of = dict(va_devices or {})
    for comp, dev in va_of.items():
        if dev not in VA_DEVICES:
            msg = f"component {comp!r}: {dev!r} is not a socklight Verilog-A device ({sorted(VA_DEVICES)})"
            raise KeyError(msg)

    specs: dict[str, registry.ModelSpec] = {}
    ports_of: dict[str, tuple[str, ...]] = {}
    electrical_of: dict[str, set[str]] = {}
    for inst in instances.values():
        comp = inst["component"]
        if comp in ports_of:
            continue
        if comp in va_of:
            dev = VA_DEVICES[va_of[comp]]
            ports_of[comp] = dev.optical + dev.electrical
            electrical_of[comp] = set(dev.electrical)
        else:
            specs[comp] = registry._models[_resolve(comp, models)][0]
            ports_of[comp] = tuple(specs[comp].ports)
            electrical_of[comp] = set()

    def check(ref: str, what: str) -> tuple[str, str]:
        inst, port = _ref(ref)
        if inst not in instances:
            msg = f"{what} {ref!r}: no instance {inst!r}"
            raise ValueError(msg)
        ports = ports_of[instances[inst]["component"]]
        if port not in ports:
            msg = f"{what} {ref!r}: {instances[inst]['component']!r} has no port {port!r} (ports {ports})"
            raise ValueError(msg)
        return inst, port

    def is_electrical(ref: tuple[str, str]) -> bool:
        return ref[1] in electrical_of[instances[ref[0]]["component"]]

    # Nodes: union the refs of every net; a group with an external port takes its name.
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_net: dict[tuple[str, str], int] = {}
    for k, net in enumerate(netlist.get("nets", ())):
        a, b = check(net["p1"], "net"), check(net["p2"], "net")
        if is_electrical(a) != is_electrical(b):
            msg = f"net {net['p1']!r} - {net['p2']!r} joins an optical and an electrical port"
            raise ValueError(msg)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
        first_net.setdefault(a, k)
        first_net.setdefault(b, k)
    ext_refs = {ext: check(ref, f"external port {ext!r}") for ext, ref in netlist.get("ports", {}).items()}

    iq = ("_i", "_q")
    nodes = _Names()
    node_of_group: dict[tuple[str, str], str] = {}
    external: dict[str, tuple[str, str]] = {}
    for ext in sorted(ext_refs):
        root = find(ext_refs[ext])
        if root not in node_of_group:
            node_of_group[root] = nodes(ext, ("",) if is_electrical(root) else iq)
        node = node_of_group[root]
        if not is_electrical(root):
            external[ext] = (f"{node}_i", f"{node}_q")
    group_net: dict[tuple[str, str], int] = {}
    for ref, k in first_net.items():
        root = find(ref)
        group_net[root] = min(k, group_net.get(root, k))
    for root, k in sorted(group_net.items(), key=lambda item: item[1]):
        if root not in node_of_group:
            node_of_group[root] = nodes(f"n{k}", ("",) if is_electrical(root) else iq)

    # Model lines, one per component.
    locals_ = _Names({TERM_MODEL})
    local_of = {comp: locals_(comp) for comp in sorted(ports_of)}
    lines = [
        title.replace("\n", " ").replace("\r", " "),
        "ground 0",
        f'load "{socklight.osdi_path()}"',
        f'load "{socklight.va_path("optical_term")}"',
    ]
    lines += [f'load "{socklight.va_path(d)}"' for d in sorted(set(va_of[c] for c in ports_of if c in va_of) - {"optical_term"})]
    lines.append(f"model {TERM_MODEL} optical_term")
    for c in sorted(ports_of):
        model = va_of[c] if c in va_of else f"'{specs[c].name.lower()}'"
        lines.append(f"model {local_of[c]} {model}")

    # Instances, sorted.
    inames = _Names()
    ident = {name: inames(name) for name in sorted(instances)}
    dropped: dict[str, set[str]] = {}
    terms: list[tuple[str, str]] = []
    electrical: dict[str, dict[str, str]] = {}
    for name in sorted(instances):
        inst = instances[name]
        comp = inst["component"]
        if comp in va_of:
            dev = VA_DEVICES[va_of[comp]]
            # parameter as written in the deck -> key matching it (case-insensitive)
            pnames = {p: p for p in dev.params}
            match = str.lower
            terminals_of = dev.terminals
        else:
            pnames = {p.name.lower(): p.name for p in specs[comp].params}
            match = str
            terminals_of = tuple(f"{p}{x}" for p in specs[comp].ports for x in iq)
        keys = {v: k for k, v in pnames.items()}  # matched key -> deck name
        merged = {**(inst.get("settings") or {}), **(inst.get("info") or {})}
        given: dict[str, object] = {}  # deck name -> value
        for key, v in merged.items():
            deck_name = keys.get(match(key))
            if deck_name is None:
                dropped.setdefault(comp, set()).add(key)
            elif v is not None:
                given[deck_name] = (key, v)
        for p in pnames:
            if p not in given and p in glob and glob[p] is not None:
                given[p] = (p, glob[p])
        rendered = []
        for p in pnames:
            if p in given:
                key, v = given[p]
                rendered.append(f"{p}={_value(v, f'instance {name!r} setting {key!r}')}")
            elif p in var_of:
                rendered.append(f"{p}={var_of[p]}")
        terminals = []
        for t in terminals_of:
            if comp in va_of and t in electrical_of[comp]:
                root = find((name, t))
                if root not in node_of_group:  # on no net and not external: left for the caller
                    node_of_group[root] = nodes(f"{ident[name]}_{t}")
                electrical.setdefault(name, {})[t] = node_of_group[root]
                terminals.append(node_of_group[root])
                continue
            port, suffix = t[:-2], t[-2:]
            root = find((name, port))
            if root not in node_of_group:  # on no net and not external: terminate it
                node_of_group[root] = nodes(f"{ident[name]}_{port}", iq)
                terms.append((f"t_{ident[name]}_{port}", node_of_group[root]))
            terminals.append(f"{node_of_group[root]}{suffix}")
        lines.append(" ".join([ident[name], f"({' '.join(terminals)})", local_of[comp], *rendered]))
    for tname, node in terms:
        lines.append(f"{inames(tname)} ({node}_i {node}_q) {TERM_MODEL}")
    return Deck("\n".join(lines) + "\n", external, dropped, variables, electrical)
