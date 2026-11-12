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
from dataclasses import dataclass

import numpy as np

from . import registry

__all__ = ["TERM_MODEL", "Deck", "netlist_to_deck"]

TERM_MODEL = "optical_term"
"""Local model name of the ``optical_term`` the deck loads; usable by lines appended to it."""

# Words VACASK reads as keywords at the start of a line (dfllexer.l), plus the term model.
_RESERVED = {"model", "global", "ground", "load", "embed", "subckt", "ends", "parameters", "control", "endc"}
_NON_IDENT = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class Deck:
    text: str  # a complete VACASK deck body: title, ground, loads, model lines, instances
    external: dict[str, tuple[str, str]]  # external port -> (i node, q node)
    dropped: dict[str, set[str]]  # component -> settings not passed (not model parameters)
    variables: tuple[str, ...] = ()  # circuit variables the deck references: declare them (``var``)


class _Names:
    """Allocates unique VACASK identifiers."""

    def __init__(self, taken=()):
        self.taken = set(taken)

    def __call__(self, raw: str) -> str:
        base = _NON_IDENT.sub("_", str(raw)) or "_"
        if base[0].isdigit():
            base = "_" + base
        if base in _RESERVED:
            base += "_"
        name, k = base, 2
        while name in self.taken:
            name, k = f"{base}_{k}", k + 1
        self.taken.add(name)
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
    title: str = "socklight netlist",
) -> Deck:
    """Converts a gdsfactory/sax flat netlist to a VACASK deck (see the module docstring).

    ``models`` is a prefix (``"cspdk.si220"``: component ``coupler`` uses the registered
    ``"cspdk.si220:coupler"``) or a mapping component -> registered name. The deck has no
    analysis or control block (``Deck.variables`` need declaring in one), and drives nothing: append sources and detectors to
    ``Deck.text`` using the nodes in ``Deck.external``.
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

    specs: dict[str, registry.ModelSpec] = {}
    for inst in instances.values():
        comp = inst["component"]
        if comp not in specs:
            specs[comp] = registry._models[_resolve(comp, models)][0]

    def check(ref: str, what: str) -> tuple[str, str]:
        inst, port = _ref(ref)
        if inst not in instances:
            msg = f"{what} {ref!r}: no instance {inst!r}"
            raise ValueError(msg)
        ports = specs[instances[inst]["component"]].ports
        if port not in ports:
            msg = f"{what} {ref!r}: {instances[inst]['component']!r} has no port {port!r} (ports {ports})"
            raise ValueError(msg)
        return inst, port

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
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
        first_net.setdefault(a, k)
        first_net.setdefault(b, k)
    ext_refs = {ext: check(ref, f"external port {ext!r}") for ext, ref in netlist.get("ports", {}).items()}

    nodes = _Names()
    node_of_group: dict[tuple[str, str], str] = {}
    external: dict[str, tuple[str, str]] = {}
    for ext in sorted(ext_refs):
        root = find(ext_refs[ext])
        if root not in node_of_group:
            node_of_group[root] = nodes(ext)
        node = node_of_group[root]
        external[ext] = (f"{node}_i", f"{node}_q")
    group_net: dict[tuple[str, str], int] = {}
    for ref, k in first_net.items():
        root = find(ref)
        group_net[root] = min(k, group_net.get(root, k))
    for root, k in sorted(group_net.items(), key=lambda item: item[1]):
        if root not in node_of_group:
            node_of_group[root] = nodes(f"n{k}")

    # Model lines, one per component.
    locals_ = _Names({TERM_MODEL})
    local_of = {comp: locals_(comp) for comp in sorted(specs)}
    lines = [
        title.replace("\n", " ").replace("\r", " "),
        "ground 0",
        f'load "{socklight.osdi_path()}"',
        f'load "{socklight.va_path("optical_term")}"',
        f"model {TERM_MODEL} optical_term",
    ]
    lines += [f"model {local_of[c]} '{specs[c].name.lower()}'" for c in sorted(specs)]

    # Instances, sorted.
    inames = _Names()
    ident = {name: inames(name) for name in sorted(instances)}
    dropped: dict[str, set[str]] = {}
    terms: list[tuple[str, str]] = []
    for name in sorted(instances):
        inst = instances[name]
        comp = inst["component"]
        spec = specs[comp]
        pnames = {p.name for p in spec.params}
        merged = {**(inst.get("settings") or {}), **(inst.get("info") or {})}
        given: dict[str, object] = {}
        for key, v in merged.items():
            if key not in pnames:
                dropped.setdefault(comp, set()).add(key)
            elif v is not None:
                given[key] = v
        for p in spec.params:
            if p.name not in given and p.name.lower() in glob and glob[p.name.lower()] is not None:
                given[p.name] = glob[p.name.lower()]
        rendered = []
        for p in spec.params:
            if p.name in given:
                rendered.append(f"{p.name.lower()}={_value(given[p.name], f'instance {name!r} setting {p.name!r}')}")
            elif p.name.lower() in var_of:
                rendered.append(f"{p.name.lower()}={var_of[p.name.lower()]}")
        terminals = []
        for port in spec.ports:
            root = find((name, port))
            if root not in node_of_group:  # on no net and not external: terminate it
                node_of_group[root] = nodes(f"{ident[name]}_{port}")
                terms.append((f"t_{ident[name]}_{port}", node_of_group[root]))
            node = node_of_group[root]
            terminals.append(f"{node}_i {node}_q")
        lines.append(" ".join([ident[name], f"({' '.join(terminals)})", local_of[comp], *rendered]))
    for tname, node in terms:
        lines.append(f"{inames(tname)} ({node}_i {node}_q) {TERM_MODEL}")
    return Deck("\n".join(lines) + "\n", external, dropped, variables)
