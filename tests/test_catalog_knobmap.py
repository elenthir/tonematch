from tonematch.catalog import Catalog, ParamInfo, PluginInfo, tag_plugin
from tonematch.knobmap import KnobMap


def test_default_knobmap_roles_and_param_kinds():
    km = KnobMap.load()
    assert km.role_for("Archetype Nolly X") == "amp_suite"
    assert km.role_for("Archetype: Gojira X") == "amp_suite"
    assert km.role_for("ReaEQ") == "eq"
    assert km.role_for("Some IR Loader", "Fx|Cab") == "cab"
    assert km.role_for("TotallyUnknown") == "other"
    cls = lambda n: km.classify_param("Archetype Nolly X", n)[0]
    assert cls("Amp 2 Gain") == "primary"
    assert cls("Amp 2 Presence") == "primary"
    assert cls("Cab 1 Mic 1 Type") == "primary"
    assert cls("Delay Time") == "ambience"
    assert cls("Reverb Mix") == "ambience"
    assert cls("Output Gain") == "excluded"
    assert cls("Transpose Semitones") == "excluded"
    assert cls("Noise Gate Threshold") == "excluded"
    kind, val, _ = km.classify_param("Archetype Nolly X", "Noise Gate Active")
    assert kind == "fixed" and val == 1.0
    kind, val, _ = km.classify_param("Archetype Nolly X", "Doubler Active")
    assert kind == "fixed" and val == 0.0
    assert cls("Pedal 1 Level") == "secondary"


def test_user_knobmap_overrides(tmp_path):
    f = tmp_path / "mine.yaml"
    f.write_text("""
plugins:
  - match: "nolly"
    role: amp_suite
    excluded: ["Amp 1 .*"]           # only use amp 2
    ranges: {"Amp 2 Gain": [0.3, 0.7]}
    fixed: {"Amp Select": 1.0}
""")
    km = KnobMap.load([f])
    assert km.classify_param("Archetype Nolly X", "Amp 1 Gain")[0] == "excluded"
    kind, _, rng = km.classify_param("Archetype Nolly X", "Amp 2 Gain")
    assert kind == "primary" and rng == (0.3, 0.7)
    assert km.classify_param("Archetype Nolly X", "Amp Select")[:2] == ("fixed", 1.0)


def test_catalog_roundtrip_and_tagging(tmp_path):
    p = PluginInfo(id="x", name="Archetype Plini X", path="/x.vst3", params=[
        ParamInfo("Amp Gain", "amp_gain", "float", 0.5),
        ParamInfo("Cab IR", "cab_ir", "str", 0.0, n_values=200, values=None, raw_centers=None),
        ParamInfo("Pedal Mix Knob", "pedal_mix_knob", "float", 0.5),
        ParamInfo("Big List", "big_list", "str", 0.0, n_values=300),
    ])
    tag_plugin(p, KnobMap.load())
    assert p.role == "amp_suite" and p.vendor == "Neural DSP"
    kinds = {q.name: q.kind for q in p.params}
    assert kinds["Amp Gain"] == "primary"
    assert kinds["Cab IR"] == "primary"           # big discrete list, but primary by name
    assert kinds["Big List"] == "excluded"        # big discrete list with no role → not searched
    cat = Catalog({"x": p})
    cat.save(tmp_path / "c.json")
    back = Catalog.load(tmp_path / "c.json")
    assert back.plugins["x"].params[1].n_values == 200
    assert back.find("plini").id == "x"
    assert len(back.plugins["x"].searchable()) == 3
