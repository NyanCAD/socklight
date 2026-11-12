"""socklight.netlist: a gdsfactory/sax flat netlist dict -> a VACASK deck."""

from __future__ import annotations

import cmath
import math
import re

import pytest
from test_osdi import wave

import socklight
from conftest import sorted_ports

from socklight import registry
from socklight.netlist import TERM_MODEL, Deck, netlist_to_deck

PREFIX = "socklight.models"
TOL = 1e-9


def mzi(coupling=0.3, l_top=10.0, l_bot=30.0) -> dict:
    """A gdsfactory-shaped MZI: two coupler_ideal and two straights, external o1..o4."""
    return {
        "instances": {
            "cp1": {"component": "coupler_ideal", "settings": {"coupling": coupling, "gap": 0.2}, "info": {}},
            "cp2": {"component": "coupler_ideal", "settings": {"coupling": coupling, "gap": 0.2}, "info": {}},
            "top": {"component": "straight", "settings": {"length": l_top, "cross_section": "strip"}, "info": {"length": l_top}},
            "bot": {"component": "straight", "settings": {"length": l_bot, "cross_section": "strip"}, "info": {"length": l_bot}},
        },
        "placements": {},
        "nets": (
            {"p1": "cp1,o4", "p2": "top,o1"},
            {"p1": "cp1,o3", "p2": "bot,o1"},
            {"p1": "top,o2", "p2": "cp2,o1"},
            {"p1": "bot,o2", "p2": "cp2,o2"},
        ),
        "ports": {"o1": "cp1,o1", "o2": "cp1,o2", "o3": "cp2,o3", "o4": "cp2,o4"},
    }


def straight_t(length: float, wl: float, wl0=1.55, neff=2.34, ng=3.4) -> complex:
    n = neff - (wl - wl0) * (ng - neff) / wl0
    return cmath.exp(2j * math.pi * n * length / wl)


def mzi_analytic(wl: float, coupling=0.3, l_top=10.0, l_bot=30.0) -> dict[str, complex]:
    """S[p, o1] of the MZI: coupler_ideal has thru (o1,o4),(o2,o3) and cross (o1,o3),(o2,o4)."""
    tau, jk = math.sqrt(1 - coupling), 1j * math.sqrt(coupling)
    a1 = tau * straight_t(l_top, wl)  # into cp2,o1
    a2 = jk * straight_t(l_bot, wl)  # into cp2,o2
    return {"o2": 0.0, "o3": jk * a1 + tau * a2, "o4": tau * a1 + jk * a2}


def drive(deck: Deck, src: str, terms=(), pds=()) -> str:
    """Appends a source on ``src``, terms and photodetectors (into 1 ohm) on other ports."""
    import vacask

    mod = f"{vacask.__file__.rsplit('/', 1)[0]}/mod"
    lines = [
        deck.text.rstrip("\n"),
        f'load "{socklight.va_path("optical_source")}"',
        f'load "{socklight.va_path("photodetector")}"',
        f'load "{mod}/resistor.osdi"',
        "model src optical_source",
        "model pd photodetector",
        "model res resistor",
    ]
    i, q = deck.external[src]
    lines.append(f"s_drive ({i} {q}) src p=1")
    for k, p in enumerate(terms):
        i, q = deck.external[p]
        lines.append(f"t_ext{k} ({i} {q}) {TERM_MODEL}")
    for p in pds:  # test port names are identifiers
        i, q = deck.external[p]
        lines += [f"pd_{p} ({i} {q} det_{p} 0) pd", f"r_{p} (det_{p} 0) res r=1"]
    return "\n".join(lines) + "\n"


def lines_of(deck: Deck, start: str) -> list[str]:
    return [ln for ln in deck.text.splitlines() if ln.startswith(start)]


# ── against VACASK ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("wl", [1.50, 1.55, 1.58])
def test_mzi_matches_analytic(run_op, wl):
    deck = netlist_to_deck(mzi(), PREFIX, params={"wl": wl})
    assert deck.dropped == {"coupler_ideal": {"gap"}, "straight": {"cross_section"}}
    v = run_op(drive(deck, "o1", terms=("o2", "o4"), pds=("o3",)))
    want = mzi_analytic(wl)
    for p in ("o2", "o3", "o4"):
        got = complex(v[deck.external[p][0]], v[deck.external[p][1]])
        assert abs(got - want[p]) < TOL, (p, got, want[p])
    assert abs(v["det_o3"] - abs(want["o3"]) ** 2) < TOL


def test_wl_circuit_variable_sweep(run_op):
    # One native sweep over the circuit variable wl moves every instance (the CLI equivalent
    # declares ``var wl=1.55`` in the control block; here Circuit.setVariable does it).
    from conftest import analyse, elaborate

    deck = netlist_to_deck(mzi(), PREFIX, variables=("wl",))
    assert deck.variables == ("wl",)
    cir, _keep = elaborate(drive(deck, "o1", terms=("o2", "o4"), pds=("o3",)), variables={"wl": 1.55})
    wls = [1.50, 1.52, 1.55, 1.58, 1.60]
    res = analyse(cir, {"variable": "wl", "values": wls})
    assert len(res["det_o3"]) == len(wls)
    for k, wl in enumerate(wls):
        want = mzi_analytic(wl)
        for p in ("o2", "o3", "o4"):
            i, q = deck.external[p]
            assert abs(complex(res[i][k], res[q][k]) - want[p]) < TOL, (wl, p)
        assert abs(res["det_o3"][k] - abs(want["o3"]) ** 2) < TOL


def test_dangling_port_terminated_in_vacask(run_op):
    # o2 of the straight is on no net and not external: the deck terminates it, so the run works
    nl = {"instances": {"s": {"component": "straight", "settings": {"length": 5.0}}}, "nets": [], "ports": {"a": "s,o1"}}
    deck = netlist_to_deck(nl, PREFIX)
    v = run_op(drive(deck, "a"))
    assert abs(wave(v, "a") - 1.0) < TOL  # matched: V = a + b = 1


def test_int_and_str_params_and_sanitized_names_elaborate(run_op):
    nl = {
        "instances": {
            "model": {"component": "sorted-ports", "settings": {"n": 2.5, "mode": 'b"\\', "wl": 1e-5}},
            "1x": {"component": "sorted-ports", "settings": {"n": 2}},
        },
        "nets": [{"p1": "model,out0", "p2": "1x,in0"}],
        "ports": {"in.0": "model,in0"},
    }
    deck = netlist_to_deck(nl, {"sorted-ports": "t.m:sorted_ports"})
    v = run_op(drive(deck, "in.0"))
    # instance "model" (a keyword) becomes "model_"; its out1 is dangling, so terminated: b = S a,
    # with S[out1,in0] = 0.25j n and n = 2.5
    assert abs(complex(v["model__out1_i"], v["model__out1_q"]) - 0.625j) < TOL


# ── text ──────────────────────────────────────────────────────────────────────


def test_deck_shape():
    deck = netlist_to_deck(mzi(), PREFIX, params={"wl": 1.55}, title="my mzi")
    lines = deck.text.splitlines()
    assert lines[0] == "my mzi"
    assert "ground 0" in lines
    assert f'load "{socklight.osdi_path()}"' in lines
    assert f'load "{socklight.va_path("optical_term")}"' in lines
    assert "model optical_term optical_term" in lines
    assert lines_of(deck, "model coupler_ideal ") == ["model coupler_ideal 'socklight.models:coupler_ideal'"]
    assert lines_of(deck, "model straight ") == ["model straight 'socklight.models:straight'"]
    assert "analysis" not in deck.text and "control" not in deck.text
    assert deck.external == {p: (f"{p}_i", f"{p}_q") for p in ("o1", "o2", "o3", "o4")}
    # sorted instances, terminals in the model's sorted port order, nets n<k>
    insts = [ln.split()[0] for ln in lines if re.match(r"^\w+ \(", ln)]
    assert insts == ["bot", "cp1", "cp2", "top"]
    assert lines_of(deck, "cp1 ") == ["cp1 (o1_i o1_q o2_i o2_q n1_i n1_q n0_i n0_q) coupler_ideal wl=1.55 coupling=0.3"]
    assert lines_of(deck, "top ") == ["top (n0_i n0_q n2_i n2_q) straight wl=1.55 length=10.0"]


def test_deterministic():
    a = mzi()
    b = mzi()
    b["instances"] = dict(reversed(list(b["instances"].items())))
    assert netlist_to_deck(a, PREFIX).text == netlist_to_deck(b, PREFIX).text


def one(component="straight", settings=None, info=None, name="s") -> dict:
    inst = {"component": component, "settings": settings or {}}
    if info is not None:
        inst["info"] = info
    return {"instances": {name: inst}, "nets": [], "ports": {"a": f"{name},o1", "b": f"{name},o2"}}


def inst_line(deck: Deck, name="s") -> str:
    (line,) = lines_of(deck, f"{name} (")
    return line


def test_info_over_settings():
    deck = netlist_to_deck(one(settings={"length": 10, "neff": 2.0}, info={"length": 16.637}), PREFIX)
    assert inst_line(deck).endswith("straight neff=2.0 length=16.637")


def test_dropped_reported_and_none_omitted():
    deck = netlist_to_deck(one(settings={"length": None, "width": 0.5, "cross_section": "x", "npoints": None}), PREFIX)
    assert deck.dropped == {"straight": {"width", "cross_section", "npoints"}}
    assert inst_line(deck) == "s (a_i a_q b_i b_q) straight"


def test_no_dropped_when_all_used():
    assert netlist_to_deck(one(settings={"length": 3.0}), PREFIX).dropped == {}


def test_global_params_fill_only_what_the_instance_left_unset():
    nl = mzi()
    nl["instances"]["top"]["settings"]["wl"] = 1.3
    deck = netlist_to_deck(nl, PREFIX, params={"wl": 1.55, "loss_db_cm": 2, "unknown": 1.0, "ng": None})
    assert inst_line(deck, "top") == "top (n0_i n0_q n2_i n2_q) straight wl=1.3 length=10.0 loss_db_cm=2"
    assert inst_line(deck, "bot") == "bot (n1_i n1_q n3_i n3_q) straight wl=1.55 length=30.0 loss_db_cm=2"
    assert inst_line(deck, "cp1").endswith("coupler_ideal wl=1.55 coupling=0.3")


def test_variables_emit_the_variable_name():
    nl = mzi()
    nl["instances"]["top"]["settings"]["wl"] = 1.3
    deck = netlist_to_deck(nl, PREFIX, params={"loss_db_cm": 2}, variables=("wl", "nosuch"))
    assert deck.variables == ("wl", "nosuch")
    assert inst_line(deck, "top") == "top (n0_i n0_q n2_i n2_q) straight wl=1.3 length=10.0 loss_db_cm=2"
    assert inst_line(deck, "bot") == "bot (n1_i n1_q n3_i n3_q) straight wl=wl length=30.0 loss_db_cm=2"
    assert inst_line(deck, "cp1").endswith("coupler_ideal wl=wl coupling=0.3")
    assert netlist_to_deck(mzi(), PREFIX).variables == ()


def test_variable_also_in_params_raises():
    with pytest.raises(ValueError, match="wl"):
        netlist_to_deck(mzi(), PREFIX, params={"WL": 1.55}, variables=["wl"])


@pytest.fixture
def sp_model(monkeypatch):
    """t.m:sorted_ports (str and int params) for text-only tests, without the session
    fixture: registering it for real here would not survive test_registry's resets."""
    monkeypatch.setitem(registry._models, "t.m:sorted_ports", (registry._probe("t.m:sorted_ports", sorted_ports), sorted_ports))


def sp(settings) -> dict:
    return {"instances": {"x": {"component": "sp", "settings": settings}}, "nets": [], "ports": {}}


def test_value_types(sp_model):
    deck = netlist_to_deck(sp({"mode": 'a"b\\c', "n": True, "wl": 1e-5}), {"sp": "t.m:sorted_ports"})
    assert inst_line(deck, "x").endswith('sp wl=1e-05 mode="a\\"b\\\\c" n=1')
    deck = netlist_to_deck(sp({"n": False}), {"sp": "t.m:sorted_ports"})
    assert inst_line(deck, "x").endswith("sp n=0")


def test_full_precision():
    deck = netlist_to_deck(one(settings={"length": 0.1 + 0.2}), PREFIX)
    assert "length=0.30000000000000004" in inst_line(deck)


@pytest.mark.parametrize("value", [[1.0, 2.0], {"a": 1}, (1, 2), object()])
def test_unsupported_setting_type_raises(value):
    with pytest.raises(TypeError, match=r"'s'.*'length'"):
        netlist_to_deck(one(settings={"length": value}), PREFIX)


def test_unsupported_type_in_dropped_setting_is_fine():
    deck = netlist_to_deck(one(settings={"bend": {"component": "x"}}), PREFIX)
    assert deck.dropped == {"straight": {"bend"}}


def test_unsupported_global_param_type_raises():
    with pytest.raises(TypeError, match="wl"):
        netlist_to_deck(one(), PREFIX, params={"wl": [1.5]})


def test_non_finite_raises():
    with pytest.raises(ValueError, match="length"):
        netlist_to_deck(one(settings={"length": float("inf")}), PREFIX)


def test_prefix_and_mapping_agree():
    mapping = {"coupler_ideal": "socklight.models:coupler_ideal", "straight": "socklight.models:straight"}
    assert netlist_to_deck(mzi(), PREFIX).text == netlist_to_deck(mzi(), mapping).text


def test_unknown_component_prefix_raises():
    with pytest.raises(KeyError, match="nosuchthing"):
        netlist_to_deck(one("nosuchthing"), PREFIX)


def test_unknown_component_mapping_raises():
    with pytest.raises(KeyError, match="coupler_ideal"):
        netlist_to_deck(mzi(), {"straight": "socklight.models:straight"})


def test_mapping_to_unregistered_model_raises():
    with pytest.raises(KeyError, match="straight"):
        netlist_to_deck(one(), {"straight": "no.such:model"})


def test_port_not_on_model_raises():
    nl = one()
    nl["ports"]["c"] = "s,o3"
    with pytest.raises(ValueError, match=r"o3"):
        netlist_to_deck(nl, PREFIX)


def test_dangling_port_terminated():
    nl = one()
    del nl["ports"]["b"]
    deck = netlist_to_deck(nl, PREFIX)
    assert inst_line(deck) == "s (a_i a_q s_o2_i s_o2_q) straight"
    assert lines_of(deck, "t_s_o2 ") == ["t_s_o2 (s_o2_i s_o2_q) optical_term"]


def test_external_port_on_a_net_names_the_net():
    nl = {
        "instances": {
            "s1": {"component": "straight", "settings": {}},
            "s2": {"component": "straight", "settings": {}},
        },
        "nets": [{"p1": "s1,o2", "p2": "s2,o1"}],
        "ports": {"a": "s1,o1", "mid": "s1,o2", "b": "s2,o2"},
    }
    deck = netlist_to_deck(nl, PREFIX)
    assert deck.external["mid"] == ("mid_i", "mid_q")
    assert inst_line(deck, "s1") == "s1 (a_i a_q mid_i mid_q) straight"
    assert inst_line(deck, "s2") == "s2 (mid_i mid_q b_i b_q) straight"


def test_identifier_sanitizing():
    nl = {
        "instances": {
            "b-1": {"component": "my comp", "settings": {}},
            "b_1": {"component": "my.comp", "settings": {}},
            "model": {"component": "optical_term", "settings": {}},
            "2x": {"component": "my comp", "settings": {}},
        },
        "nets": [],
        "ports": {"o.1": "b-1,o1", "1": "b_1,o1"},
    }
    models = {"my comp": "socklight.models:straight", "my.comp": "socklight.models:straight", "optical_term": "socklight.models:straight"}
    deck = netlist_to_deck(nl, models)
    ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    model_lines = lines_of(deck, "model ")
    locals_ = [ln.split()[1] for ln in model_lines]
    assert all(ident.match(n) for n in locals_)
    assert len(set(locals_)) == len(locals_) == 4  # optical_term + three components
    insts = [ln.split()[0] for ln in deck.text.splitlines() if re.match(r"^\S+ \(", ln)]
    assert all(ident.match(n) for n in insts)
    assert len(set(insts)) == len(insts)
    assert "model" not in insts
    for i, q in deck.external.values():
        assert ident.match(i) and ident.match(q)
    assert len(set(deck.external.values())) == 2
