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


def test_bass_instrument_utility_roles_are_kept_out_of_guitar_chains():
    km = KnobMap.load()
    assert km.role_for("Darkglass Ultra") == "bass"
    assert km.role_for("Parallax X") == "bass"
    assert km.role_for("Ampeg SVT-VR") == "bass"
    assert km.role_for("Bass Amp Room") == "bass"
    assert km.role_for("Superior Drummer 3") == "instrument"
    assert km.role_for("NeuralNote") == "instrument"
    assert km.role_for("Some Synth", "Instrument|Synth") == "instrument"
    assert km.role_for("Nice Amp", "", is_instrument=True) == "instrument"
    assert km.role_for("Pro-L 2 Limiter") == "utility"
    assert km.role_for("ReaTune") == "utility"
    assert km.role_for("Archetype Nolly X") == "amp_suite"
    assert km.role_for("Neural Amp Modeler") == "amp"
    assert km.role_for("Soundshed Guitar") == "other"       # unknown: listed, never chained

    from tonematch.catalog import Catalog, PluginInfo, ParamInfo
    from tonematch.search import SearchConfig, propose_chains
    cat = Catalog()
    for name in ("Archetype Nolly X", "Darkglass Ultra", "Superior Drummer 3", "ReaTune", "Neural Amp Modeler", "Some Cab IR"):
        p = PluginInfo(id=name, name=name, path=name, params=[ParamInfo("Gain", "gain", "float", 0.5)])
        tag_plugin(p, km)
        cat.plugins[name] = p
    keys = [c.key for c in propose_chains(cat, SearchConfig())]
    assert "Archetype Nolly X" in keys and "Neural Amp Modeler > Some Cab IR" in keys
    assert not any(("Darkglass" in k or "Drummer" in k or "ReaTune" in k) for k in keys)
    bass_keys = [c.key for c in propose_chains(cat, SearchConfig(), instrument="bass")]
    assert bass_keys == ["Darkglass Ultra"]


def test_probe_result_survives_stdout_noise(tmp_path, monkeypatch):
    """A plugin printing to stdout while loading must not corrupt the scan."""
    import json, sys, subprocess
    from tonematch import catalog as C
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        out = cmd[cmd.index("--out") + 1]
        open(out, "w").write(json.dumps([{"id": "X", "name": "X", "path": cmd[cmd.index("--probe") + 1],
                                          "vendor": "", "category": "", "format": "vst3", "role": "other",
                                          "params": [], "latency": 0, "is_instrument": False, "error": None}]))
        return subprocess.CompletedProcess(cmd, 0, stdout="LICENSE OK [garbage\n", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    infos = C.probe_plugin_file(tmp_path / "X.vst3")
    assert len(infos) == 1 and infos[0].name == "X" and not infos[0].error

    def crash_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="boom")
    monkeypatch.setattr(subprocess, "run", crash_run)
    infos = C.probe_plugin_file(tmp_path / "Y.vst3")
    assert infos[0].error and "boom" in infos[0].error


def test_find_plugin_files_dedups_windows_bundles(tmp_path):
    from tonematch.catalog import find_plugin_files
    b = tmp_path / "Archetype Nolly X.vst3" / "Contents" / "x86_64-win"
    b.mkdir(parents=True)
    (b / "Archetype Nolly X.vst3").write_bytes(b"")
    (tmp_path / "Flat.vst3").write_bytes(b"")
    found = sorted(p.name for p in find_plugin_files([tmp_path]))
    assert found == ["Archetype Nolly X.vst3", "Flat.vst3"]
    assert all(p.parent == tmp_path for p in find_plugin_files([tmp_path]))
